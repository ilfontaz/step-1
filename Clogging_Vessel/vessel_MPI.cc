/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 
 *
 * Author: Wolfgang Bangerth, University of Texas at Austin, 2000, 2004
 *         Wolfgang Bangerth, Texas A&M University, 2016
 */

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h> // May not be needed
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>  // may not be needed

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>   // may not be needed
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>  // may not be needed

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/fe/fe_nothing.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <fstream>
#include <iostream>
#include <sstream>


namespace Fluid_quantities
{
    using namespace dealii;
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_vf (unsigned int q,
            std::vector<Vector<double> > old_solution_values)
    {
        Tensor<1,dim> v;
        v[0] = old_solution_values[q](0);
        v[1] = old_solution_values[q](1);
        
        return v;
    }
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_uf (unsigned int q,
            std::vector<Vector<double> > old_solution_values)
    {
        Tensor<1,dim> u;
        u[0] = old_solution_values[q](dim);
        u[1] = old_solution_values[q](dim+1);
        
        return u;
    }
    
    template <int dim>
    inline
    double
    get_pf (unsigned int q,
            std::vector<Vector<double> > old_solution_values)
    {
        double pf  ;
        
        pf = old_solution_values[q](dim+dim);
        
        return pf;
    }
    
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_vf (unsigned int q,
                 std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
        Tensor<2,dim> grad_v;
        grad_v[0][0] =  old_solution_grads[q][0][0];
        grad_v[0][1] =  old_solution_grads[q][0][1];
        grad_v[1][0] =  old_solution_grads[q][1][0];
        grad_v[1][1] =  old_solution_grads[q][1][1];
        
        return grad_v;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_vf_T (unsigned int q,
                   std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
        Tensor<2,dim> grad_v_transpose;
        grad_v_transpose[0][0] =  old_solution_grads[q][0][0];
        grad_v_transpose[0][1] =  old_solution_grads[q][1][0];
        grad_v_transpose[1][0] =  old_solution_grads[q][0][1];
        grad_v_transpose[1][1] =  old_solution_grads[q][1][1];
        
        return grad_v_transpose;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_uf (unsigned int q,
                 std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
        Tensor<2,dim> gradU;
        gradU[0][0] = old_solution_grads[q][dim][0];
        gradU[0][1] = old_solution_grads[q][dim][1];
        gradU[1][0] = old_solution_grads[q][dim+1][0];
        gradU[1][1] = old_solution_grads[q][dim+1][1];
        
        return gradU;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_uf_T (const Tensor<2,dim> grad_uf)
    {
        
        
        return transpose(grad_uf);
    }
    
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_f (unsigned int q,
             std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
        Tensor<2,dim> F;
        F[0][0] = 1.0 +  old_solution_grads[q][dim][0];
        F[0][1] = old_solution_grads[q][dim][1];
        F[1][0] = old_solution_grads[q][dim+1][0];
        F[1][1] = 1.0 + old_solution_grads[q][dim+1][1];
        
        return F;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_f_T (Tensor<2,dim> F_f)
    {
        return transpose(F_f);
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_f_Inv (const Tensor<2,dim> F_f)
    {
        return invert (F_f);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_f_Inv_T (const Tensor<2,dim> F_f_Inv)
    {
        return transpose (F_f_Inv);
    }
    
    template <int dim>
    inline
    double
    get_J_f (const Tensor<2,dim> F_f)
    {
        return determinant (F_f);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_Identity ()
    {
        Tensor<2,dim> identity;
        identity[0][0] = 1.0;
        identity[0][1] = 0.0;
        identity[1][0] = 0.0;
        identity[1][1] = 1.0;
        
        return identity;
    }
    
    template <int dim>   // -------> This form has been obtained doing explicitely the calculations, there is a Mathematica file for it!!!
    inline
    double
    get_J_f_LinU (unsigned int q,
                  const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
                  const Tensor<2,dim> phi_i_grads_uf)
    {
        return (phi_i_grads_uf[0][0] * (1 + old_solution_grads[q][dim+1][1]) +
                (1 + old_solution_grads[q][dim][0]) * phi_i_grads_uf[1][1] -
                phi_i_grads_uf[0][1] * old_solution_grads[q][dim+1][0] -
                old_solution_grads[q][dim][1] * phi_i_grads_uf[1][0]);
    }
    
    
    template < int dim>
    inline
    Tensor<2,dim>
    get_F_f_LinU ( const  Tensor<2,dim> phi_i_grads_uf)
    {
        Tensor<2,dim>  tmp;
        
        tmp[0][0] = phi_i_grads_uf[0][0];
        tmp[0][1] = phi_i_grads_uf[0][1];
        tmp[1][0] = phi_i_grads_uf[1][0];
        tmp[1][1] = phi_i_grads_uf[1][1];
        
        return tmp;
        
    }
    
    
    
    template <int dim>     // -------> This form has been obtained doing explicitely the calculations, there is a Mathematica file for it!!!
    inline                 // -------> Already contains the - sign!
    Tensor<2,dim>
    get_F_f_Inv_LinU (const Tensor<2,dim> phi_i_grads_uf,
                      const double J_f,
                      const double J_f_LinU,
                      unsigned int q,
                      std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
        Tensor<2,dim> F_tilde;
        F_tilde[0][0] = 1.0 + old_solution_grads[q][dim+1][1];
        F_tilde[0][1] = -old_solution_grads[q][dim][1];
        F_tilde[1][0] = -old_solution_grads[q][dim+1][0];
        F_tilde[1][1] = 1.0 + old_solution_grads[q][dim][0];
        
        Tensor<2,dim> F_tilde_LinU;
        F_tilde_LinU[0][0] = phi_i_grads_uf[1][1];
        F_tilde_LinU[0][1] = -phi_i_grads_uf[0][1];
        F_tilde_LinU[1][0] = -phi_i_grads_uf[1][0];
        F_tilde_LinU[1][1] = phi_i_grads_uf[0][0];
        
        return (-1.0/(J_f*J_f) * J_f_LinU * F_tilde +
                1.0/J_f * F_tilde_LinU);
        
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_f_Inv_T_LinU (const Tensor<2,dim> F_f_Inv_LinU)    // ----------> We used [-F^(-1)*grad_uf*F^(-1)]^T = [F^(-T)*grad_uf^(T)*-F^(-T)]
    {
        Tensor<2,dim> tmp;
        tmp = transpose(F_f_Inv_LinU);
        
        return tmp;
    }
    
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_vf_LinV (const Tensor<1,dim> phi_i_vf)
    
    {
        Tensor<1,dim> tmp;
        tmp[0] = phi_i_vf[0];
        tmp[1] = phi_i_vf[1];
        
        return tmp;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_vf_LinV (const Tensor<2,dim> phi_i_grads_vf)
    {
        Tensor<2,dim> tmp;
        tmp[0][0] = phi_i_grads_vf[0][0];    //∂xVx
        tmp[0][1] = phi_i_grads_vf[0][1];    //∂yVx
        tmp[1][0] = phi_i_grads_vf[1][0];    //∂xVy
        tmp[1][1] = phi_i_grads_vf[1][1];    //dyVy
        
        return tmp;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_vf_T_LinV (const Tensor<2,dim> phi_i_grads_vf)
    {
        Tensor<2,dim> tmp;
        tmp[0][0] = phi_i_grads_vf[0][0];    //∂xVx
        tmp[0][1] = phi_i_grads_vf[1][0];    //∂yVx
        tmp[1][0] = phi_i_grads_vf[0][1];    //∂xVy
        tmp[1][1] = phi_i_grads_vf[1][1];    //dyVy
        
        return tmp;
    }
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_uf_LinU (const Tensor<1,dim> phi_i_uf)
    {
        Tensor<1,dim> tmp;
        tmp[0] = phi_i_uf[0];
        tmp[1] = phi_i_uf[1];
        
        return tmp;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_uf_LinU (const Tensor<2,dim> phi_i_grads_uf)
    {
        Tensor<2,dim> tmp;
        tmp[0][0] = phi_i_grads_uf[0][0];    //∂xUx
        tmp[0][1] = phi_i_grads_uf[0][1];    //∂yUx
        tmp[1][0] = phi_i_grads_uf[1][0];    //∂xUy
        tmp[1][1] = phi_i_grads_uf[1][1];    //dyUy
        
        return tmp;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_uf_T_LinU (const Tensor<2,dim> phi_i_grads_uf)
    {
        Tensor<2,dim> tmp;
        tmp[0][0] = phi_i_grads_uf[0][0];    //∂xUx
        tmp[0][1] = phi_i_grads_uf[1][0];    //∂yUx
        tmp[1][0] = phi_i_grads_uf[0][1];    //∂xUy
        tmp[1][1] = phi_i_grads_uf[1][1];    //dyUy
        
        return tmp;
    }
    
    
    template <int dim>
    inline
    double
    get_pf_LinP(const double phi_i_p )
    {return phi_i_p; }
    
    
    
    template <int dim>
    inline
    double
    get_tr_grad_vf_LinV_F_f_Inv(const Tensor<2,dim> grad_vf_LinV,     // ---------> USED IN THE "incompressibility_term_LinV"
                                const Tensor<2,dim> F_f_Inv)
    {
        Tensor<2,dim> tmp;
        double trace;
        tmp = grad_vf_LinV * F_f_Inv;
        trace = tmp[0][0] + tmp[1][1];
        
        return trace;
        
    }
    
    template <int dim>
    inline
    double
    get_tr_grad_vf_F_f_Inv(Tensor<2,dim> grad_vf,      // -------> USED IN "incompressibility_term_LinU_1"
                           Tensor<2,dim> F_f_Inv)
    {
        Tensor<2,dim> tmp;
        double trace;
        tmp = grad_vf * F_f_Inv;
        trace = tmp[0][0] + tmp[1][1];
        return trace;
    }
    
    template <int dim>
    inline
    double
    get_tr_grad_vf_F_Inv_LinU(Tensor<2,dim> grad_vf,        //------> USED IN "incompressibility_term_LinU_2"
                              Tensor<2,dim> F_f_Inv_LinU)
    {
        Tensor<2,dim> tmp;
        double trace;
        tmp = grad_vf * F_f_Inv_LinU;
        trace = tmp[0][0] + tmp[1][1];
        return trace;
    }
}


namespace Solid_quantities

{
    
    using namespace dealii;
    
    
    
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_us (unsigned int q,
            std::vector<Vector<double> > old_solution_values)
    {
        Tensor<1,dim> u;
        u[0] = old_solution_values[q](dim+dim+1);
        u[1] = old_solution_values[q](dim+dim+dim);
        
        return u;
    }
    
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_us (unsigned int q,
                 std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
        Tensor<2,dim> gradU;
        gradU[0][0] = old_solution_grads[q][dim+dim+1][0];
        gradU[0][1] = old_solution_grads[q][dim+dim+1][1];
        gradU[1][0] = old_solution_grads[q][dim+dim+dim][0];
        gradU[1][1] = old_solution_grads[q][dim+dim+dim][1];
        
        return gradU;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_s (unsigned int q,
             std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
             const double g_growth)
    {
        Tensor<2,dim> F;
        F[0][0] = 1.0 +  old_solution_grads[q][dim+dim+1][0];
        F[0][1] = old_solution_grads[q][dim+dim+1][1];
        F[1][0] = old_solution_grads[q][dim+dim+dim][0];
        F[1][1] = 1.0 + old_solution_grads[q][dim+dim+dim][1];
        
        return (1.0/g_growth) * F;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_s_T (Tensor<2,dim> F_s)
    {
        return transpose(F_s);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_s_Inv (const Tensor<2,dim> F_s)
    {
        return invert (F_s);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_s_Inv_T (const Tensor<2,dim> F_s_Inv)
    {
        return transpose (F_s_Inv);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_s_LinU (unsigned int q,
                  const  Tensor<2,dim> phi_i_grads_us,
                  const double g_growth)
    {
        Tensor<2,dim> tmp;
        tmp[0][0] = phi_i_grads_us[0][0];
        tmp[0][1] = phi_i_grads_us[0][1];
        tmp[1][0] = phi_i_grads_us[1][0];
        tmp[1][1] = phi_i_grads_us[1][1];
        
        return  (1.0/g_growth) * tmp;
    }
    
    
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_F_s_T_LinU(unsigned int q,
                   const Tensor<2,dim> phi_i_grads_us,
                   const double g_growth)
    
    {
        Tensor<2,dim> tmp;
        tmp[0][0] = phi_i_grads_us[0][0];
        tmp[0][1] = phi_i_grads_us[1][0];
        tmp[1][0] = phi_i_grads_us[0][1];
        tmp[1][1] = phi_i_grads_us[1][1];
        
        return  (1.0/g_growth) * tmp;
        
    }
    
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_us_LinU (const Tensor<1,dim> phi_i_us)
    {
        Tensor<1,dim> tmp;
        tmp[0] = phi_i_us[0];
        tmp[1] = phi_i_us[1];
        
        return tmp;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_us_LinU (const Tensor<2,dim> phi_i_grads_us)
    {
        Tensor<2,dim> tmp;
        
        tmp[0][0] = phi_i_grads_us[0][0];
        tmp[0][1] = phi_i_grads_us[0][1];
        tmp[1][0] = phi_i_grads_us[1][0];
        tmp[1][1] = phi_i_grads_us[1][1];
        
        return tmp;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_grad_us_T_LinU (const Tensor<2,dim> phi_i_grads_us)
    {
        Tensor<2,dim> tmp;
        
        tmp[0][0] = phi_i_grads_us[0][0];
        tmp[0][1] = phi_i_grads_us[1][0];
        tmp[1][0] = phi_i_grads_us[0][1];
        tmp[1][1] = phi_i_grads_us[1][1];
        
        return tmp;
    }
    
    
    
    
    
}

namespace Fluid_Terms
{
    
    using namespace dealii;
    
    
    //Here we build a tensor that will help us build entire terms
    
    
    template <int dim>     // <--------- This term has been modified with regard to the newtonian case
    inline
    Tensor<2,dim>
    get_cauchy_f_vu(const double viscosity,
                    const double density_fluid,
                    const Tensor<2,dim> grad_vf,
                    const Tensor<2,dim> grad_vf_T,
                    const Tensor<2,dim> F_f_Inv,
                    const Tensor<2,dim> F_f_Inv_T)
    {
        return 0.5 * (grad_vf * F_f_Inv + F_f_Inv_T * grad_vf_T);
    }
    
    
    
    template <int dim>    // <--------- This term has been modified with regard to the newtonian case
    inline
    Tensor<2,dim>
    get_cauchy_f_vu_LinV (const double density,
                          const double viscosity,
                          const Tensor<2,dim>  grad_vf_LinV,
                          const Tensor<2,dim>  grad_vf_T_LinV,
                          const Tensor<2,dim>  F_f_Inv,
                          const Tensor<2,dim>  F_f_Inv_T)
    {
        return  0.5 * (grad_vf_LinV * F_f_Inv + F_f_Inv_T * grad_vf_T_LinV);
    }
    
    
    template <int dim>  // <--------- This term has been modified with regard to the newtonian case
    inline
    Tensor<2,dim>
    get_cauchy_f_vu_LinU (const double density,
                          const double viscosity,
                          const Tensor<2,dim>  grad_vf,
                          const Tensor<2,dim>  grad_vf_T,
                          const Tensor<2,dim>  F_f_Inv_LinU,
                          const Tensor<2,dim>  F_f_Inv_T_LinU)
    {
        return  0.5 * (grad_vf * F_f_Inv_LinU + F_f_Inv_T_LinU * grad_vf_T);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_E_f_Linear (const Tensor<2,dim> grad_uf,
                    const Tensor<2,dim> grad_uf_T)
    {
        return 0.5 * (grad_uf + grad_uf_T);
    }
    
    
    // Here we start to build the single terms of the jacobean for the fluid
    
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_E_f_Linear_LinU (const Tensor<2,dim> grad_uf_LinU,
                         const Tensor<2,dim> grad_uf_T_LinU)
    {
        return 0.5 * (grad_uf_LinU + grad_uf_T_LinU);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_sigma_lin_el_LinU1 (const double J_f,
                            const Tensor<2,dim> F_f_LinU,
                            const Tensor<2,dim> E_f_Linear,
                            const Tensor <2,dim> Identity)
    {
        return F_f_LinU * ((0.00000001/J_f) * trace(E_f_Linear) * Identity + 2 * (0.00000001/J_f) * E_f_Linear);
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_sigma_lin_el_LinU2 (const double J_f,
                            const double J_f_LinU,
                            const Tensor<2,dim> F_f,
                            const Tensor<2,dim> E_f_Linear,
                            const Tensor<2,dim> E_f_Linear_LinU,
                            const Tensor <2,dim> Identity)
    {
        return F_f * (- (0.00000001 * J_f_LinU/(J_f*J_f)) * trace(E_f_Linear) * Identity
                      + (0.00000001/J_f) * trace(E_f_Linear_LinU) * Identity
                      -  2 * (0.00000001 * J_f_LinU/(J_f * J_f)) * E_f_Linear
                      +  2 * (0.00000001/J_f) * E_f_Linear_LinU);
        
    }
    
    
    
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_acceleration_term_LinV (const Tensor<1,dim> vf_LinV,
                                const double J_f,
                                const double old_timestep_J_f,
                                const double theta)
    
    {
        return  (theta * J_f + (1.0-theta) * old_timestep_J_f) * vf_LinV;
    }
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_convective_mesh_term_LinV (const Tensor<2,dim> grad_vf_LinV,
                                   const double J_f,
                                   const Tensor<2,dim> F_f_Inv,
                                   const Tensor<1,dim> old_timestep_uf,
                                   const Tensor<1,dim> uf)
    
    {
        return J_f * grad_vf_LinV * F_f_Inv * (uf - old_timestep_uf);
    }
    
    template<int dim>
    inline
    Tensor<1,dim>
    get_convective_term_LinV_1 (const double J_f,
                                const Tensor<2,dim> grad_vf_LinV,
                                const Tensor<2,dim> F_f_Inv,
                                const Tensor<1,dim> vf)
    {
        return J_f * (grad_vf_LinV * F_f_Inv) * vf;
    }
    
    template<int dim>
    inline
    Tensor<1,dim>
    get_convective_term_LinV_2 (const double J_f,
                                const Tensor<2,dim> grad_vf,
                                const Tensor<2,dim> F_f_Inv,
                                const Tensor<1,dim> vf_LinV)
    {
        return J_f * (grad_vf * F_f_Inv) * vf_LinV;
    }
    
    template <int dim> // <--------- This term has been modified with regard to the newtonian case
    inline
    Tensor<2,dim>
    get_fluid_stress_vu_term_LinV_1 (const double epsilon,
                                     const double p,
                                     const double density_fluid,
                                     const double viscosity,
                                     const double J_f,
                                     const Tensor<2,dim> cauchy_f_vu,
                                     const Tensor<2,dim> cauchy_f_vu_LinV,
                                     const Tensor<2,dim> F_f_Inv_T)
    {
        
        double modulus = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
        
        Tensor <2,dim> cauchy_pow = 2.0 * density_fluid * viscosity * ((p-2.0)/2.0) * std::pow((epsilon * epsilon + modulus * modulus),((p-2.0)/2.0) - 1.0) *
        2.0 * modulus * modulus * cauchy_f_vu_LinV;
        
        
        return J_f * cauchy_pow * F_f_Inv_T;
    }
    
    template <int dim> // <--------- This term has been modified with regard to the newtonian case
    
    inline
    Tensor<2,dim>
    get_fluid_stress_vu_term_LinV_2(const double epsilon,
                                    const double p,
                                    const double density_fluid,
                                    const double viscosity,
                                    const double J_f,
                                    const Tensor<2,dim> cauchy_f_vu,
                                    const Tensor<2,dim> cauchy_f_vu_LinV,
                                    const Tensor<2,dim> F_f_Inv_T
                                    )
    {
        
        double modulus = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
        
        Tensor <2,dim> cauchy_pow = 2.0 * density_fluid * viscosity * std::pow((epsilon * epsilon + modulus * modulus), ((p-2.0)/2.0)) * cauchy_f_vu_LinV;
        
        return J_f * cauchy_pow * F_f_Inv_T;
    }
    
    
    inline
    double
    get_incompressibility_term_LinV(const double J_f,
                                    const double tr_grad_vf_LinV_F_f_Inv)
    {
        return J_f * tr_grad_vf_LinV_F_f_Inv;
    }
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_acceleration_term_LinU(const double theta,
                               const double J_f_LinU,
                               const Tensor<1,dim> vf,
                               const Tensor<1,dim> old_timestep_vf)
    
    {
        return theta * J_f_LinU * (vf - old_timestep_vf);
    }
    
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_convective_mesh_term_LinU_1(const double J_f_LinU,
                                    const Tensor<2,dim> grad_vf,
                                    const Tensor<1,dim> uf,
                                    const Tensor<1,dim> old_timestep_uf,
                                    const Tensor<2,dim> F_f_Inv)
    
    {
        return J_f_LinU * grad_vf * F_f_Inv * (uf - old_timestep_uf);
    }
    
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_convective_mesh_term_LinU_2(const double J_f,
                                    const Tensor<2,dim> grad_vf,
                                    const Tensor<2,dim> F_f_Inv_LinU,
                                    const Tensor<1,dim> uf,
                                    const Tensor<1,dim> old_timestep_uf)
    
    {
        return J_f * grad_vf * F_f_Inv_LinU * (uf - old_timestep_uf);
    }
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_convective_mesh_term_LinU_3(const double J_f,
                                    const Tensor<2,dim> grad_vf,
                                    const Tensor<2,dim> F_f_Inv,
                                    const Tensor<1,dim> phi_i_uf)
    
    {
        return J_f * grad_vf * F_f_Inv * phi_i_uf ;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_fluid_stress_pf_term_LinU_1(const double J_f_LinU,
                                    const double pf,
                                    const Tensor<2,dim> Identity,
                                    const Tensor<2,dim> F_f_Inv_T)
    {
        return J_f_LinU * (-pf * Identity) * F_f_Inv_T;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_fluid_stress_pf_term_LinU_2(const double J_f,
                                    const double pf,
                                    const Tensor<2,dim> Identity,
                                    const Tensor<2,dim> F_f_Inv_T_LinU)
    {
        return J_f * (-pf * Identity) * F_f_Inv_T_LinU;
    }
    
    
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_convective_term_LinU_1(const double J_f_LinU,
                               const Tensor<2,dim> grad_vf,
                               const Tensor<2,dim> F_f_Inv,
                               const Tensor<1,dim> vf)
    {
        return J_f_LinU * (grad_vf * F_f_Inv) * vf;
    }
    
    template <int dim>
    inline
    Tensor<1,dim>
    get_convective_term_LinU_2(const double J_f,
                               const Tensor<2,dim> grad_vf,
                               const Tensor<2,dim> F_f_Inv_lin_U,
                               const Tensor<1,dim> vf)
    {
        return J_f * (grad_vf * F_f_Inv_lin_U) * vf;
    }
    
    
    template <int dim> // <--------- This term has been modified with regard to the newtonian case
    inline
    Tensor<2,dim>
    get_fluid_stress_vu_term_LinU_1 (const double epsilon,
                                     const double p,
                                     const double density_fluid,
                                     const double viscosity,
                                     const double J_f,
                                     const Tensor<2,dim> cauchy_f_vu,
                                     const Tensor<2,dim> cauchy_f_vu_LinU,
                                     const Tensor<2,dim> F_f_Inv_T)
    {
        
        double modulus = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
        
        Tensor <2,dim> cauchy_pow = 2.0 * density_fluid * viscosity * ((p-2.0)/2.0) * std::pow((epsilon * epsilon + modulus * modulus),((p-2.0)/2.0) - 1.0) *
        2.0 * modulus * modulus * (cauchy_f_vu_LinU);
        
        
        return J_f * cauchy_pow * F_f_Inv_T;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_fluid_stress_vu_term_LinU_2(const double epsilon,
                                    const double p,
                                    const double density_fluid,
                                    const double viscosity,
                                    const double J_f,
                                    const Tensor<2,dim> cauchy_f_vu,
                                    const Tensor<2,dim> cauchy_f_vu_LinU,
                                    const Tensor<2,dim> F_f_Inv_T
                                    )
    {
        double modulus = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
        
        Tensor <2,dim> cauchy_pow = 2.0 * density_fluid * viscosity * std::pow((epsilon * epsilon + modulus * modulus), ((p-2.0)/2.0)) * cauchy_f_vu_LinU;
        
        return J_f * cauchy_pow * F_f_Inv_T;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_fluid_stress_vu_term_LinU_3(const double epsilon,
                                    const double p,
                                    const double density_fluid,
                                    const double viscosity,
                                    const double J_f_LinU,
                                    const Tensor<2,dim> cauchy_f_vu,
                                    const Tensor<2,dim> F_f_Inv_T,
                                    const double pf,
                                    const Tensor<2,dim> Id)
    {
        double modulus = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
        
        Tensor<2,dim> tmp = 2.0 * density_fluid * viscosity * std::pow((epsilon * epsilon + modulus * modulus ),(p-2.0)/2.0) * cauchy_f_vu;
        
        return J_f_LinU *( - pf * Id + tmp )* F_f_Inv_T;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_fluid_stress_vu_term_LinU_4(const double epsilon,
                                    const double p,
                                    const double density_fluid,
                                    const double viscosity,
                                    const double J_f,
                                    const Tensor<2,dim> cauchy_f_vu,
                                    const Tensor<2,dim> F_f_Inv_T_LinU,
                                    const Tensor<2,dim> Id,
                                    const double pf)
    {
        double modulus = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
        
        Tensor<2,dim> tmp = 2.0 * density_fluid * viscosity * std::pow((epsilon * epsilon + modulus * modulus),(p-2.0)/2.0) * cauchy_f_vu;
        
        return J_f * ( -pf * Id + tmp) * F_f_Inv_T_LinU;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_harmonic_mmpde_term_LinU_1(const double J_f,
                                   const double J_f_LinU,
                                   const Tensor<2,dim> grad_uf)
    {
        return -0.00000001 * (1.0/(J_f*J_f)) * J_f_LinU * grad_uf;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_harmonic_mmpde_term_LinU_2(const double J_f,
                                   const Tensor<2,dim> grad_uf_LinU)
    {
        return 0.00000001 * (1.0/J_f)* grad_uf_LinU;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_linear_elastic_mmpde_term_LinU_1 (const double alpha_nu_s,
                                          const double alpha_EY,
                                          const double J_f,
                                          const double J_f_LinU,
                                          const Tensor<2,dim> E_f_Linear,
                                          const Tensor <2,dim> Identity)
    {
        return  - (alpha_nu_s * alpha_EY * J_f_LinU)/((J_f*J_f)*(1.0 + alpha_nu_s)*(1.0 - 2.0 * alpha_nu_s)) * trace(E_f_Linear) * Identity;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_linear_elastic_mmpde_term_LinU_2 (const double alpha_nu_s,
                                          const double alpha_EY,
                                          const double J_f,
                                          const Tensor<2,dim> E_f_Linear_LinU,
                                          const Tensor <2,dim> Identity)
    {
        return  + (alpha_nu_s * alpha_EY)/(J_f * (1.0 + alpha_nu_s)*(1.0 - 2.0 * alpha_nu_s)) * trace(E_f_Linear_LinU) * Identity;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_linear_elastic_mmpde_term_LinU_3 (const double alpha_nu_s,
                                          const double alpha_EY,
                                          const double J_f,
                                          const double J_f_LinU,
                                          const Tensor<2,dim> E_f_Linear,
                                          const Tensor <2,dim> Identity)
    {
        return  - (alpha_EY * J_f_LinU)/((J_f * J_f)*(1.0 + alpha_nu_s)) * E_f_Linear;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_linear_elastic_mmpde_term_LinU_4 (const double alpha_nu_s,
                                          const double alpha_EY,
                                          const double J_f,
                                          const Tensor<2,dim> E_f_Linear_LinU,
                                          const Tensor <2,dim> Identity)
    {
        return  + alpha_EY/(J_f * (1.0 + alpha_nu_s)) * E_f_Linear_LinU;
    }
    
    
    inline
    double
    get_incompressibility_term_LinU_1(const double J_f_LinU,
                                      const double tr_grad_vf_F_f_Inv)
    {
        return J_f_LinU * tr_grad_vf_F_f_Inv;
    }
    
    
    inline
    double
    get_incompressibility_term_LinU_2(const double J_f,
                                      const double tr_grad_vf_F_f_Inv_LinU)
    {
        return J_f * tr_grad_vf_F_f_Inv_LinU;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_fluid_stress_pf_term_LinP(const double J_f,
                                  const double pf_LinP,
                                  const Tensor<2,dim> Identity,
                                  const Tensor<2,dim> F_f_Inv_T)
    {
        return J_f * (-pf_LinP * Identity) * F_f_Inv_T;
    }
    
    
    // Here we build the single terms to build the jacobean of the do-nothing condition ( which is in the fluid! )
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_do_nothing_term_LinU_1 (const double J_f_LinU,
                                const Tensor<2,dim> F_f_Inv_T,
                                const Tensor<2,dim> grad_vf_T)
    {
        return J_f_LinU * F_f_Inv_T * grad_vf_T * F_f_Inv_T;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_do_nothing_term_LinU_2 (const double J_f,
                                const Tensor<2,dim> F_f_Inv_T_LinU,
                                const Tensor<2,dim> F_f_Inv_T,
                                const Tensor<2,dim> grad_vf_T)
    {
        return J_f * F_f_Inv_T_LinU * grad_vf_T * F_f_Inv_T;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_do_nothing_term_LinU_3 (const double J_f,
                                const Tensor<2,dim> F_f_Inv_T_LinU,
                                const Tensor<2,dim> F_f_Inv_T,
                                const Tensor<2,dim> grad_vf_T)
    {
        return J_f * F_f_Inv_T * grad_vf_T * F_f_Inv_T_LinU;
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_do_nothing_term_LinV (const double J_f,
                              const Tensor<2,dim> F_f_Inv_T,
                              const Tensor<2,dim> grad_vf_T_LinV)
    {
        return J_f * F_f_Inv_T * grad_vf_T_LinV * F_f_Inv_T;
    }
    
    template <int dim>
    Tensor<1,dim>
    get_acceleration_term(const double theta,
                          const double J_f,
                          const double old_timestep_J_f,
                          const Tensor<1,dim> old_timestep_vf,
                          const Tensor<1,dim> vf)
    {
        return  (theta * J_f + (1.0-theta) * old_timestep_J_f) * (vf-old_timestep_vf);
    }
    
    
    template <int dim>
    Tensor<1,dim>
    get_convective_mesh_term(const double J_f,
                             const Tensor<1,dim> old_timestep_uf,
                             const Tensor<1,dim> uf,
                             const Tensor<2,dim> F_f_Inv,
                             const Tensor<2,dim> grad_v)
    {
        return J_f * (grad_v * F_f_Inv * (uf - old_timestep_uf)) ;
    }
    
    template <int dim>
    Tensor<1,dim>
    get_convective_term(const double J_f,
                        Tensor<2,dim> grad_vf,
                        Tensor<2,dim> F_f_Inv,
                        Tensor<1,dim> vf)
    {
        return J_f * (grad_vf * F_f_Inv) * vf;
        
    }
    
    template <int dim>
    Tensor<2,dim>
    get_fluid_vu_stress_term(const double epsilon,
                             const double p,
                             const double density_fluid,
                             const double viscosity,
                             const double J_f,
                             const Tensor<2,dim> cauchy_f_vu,
                             const Tensor<2,dim> F_f_Inv_T)
    {
        
        double modulus = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
        
        Tensor<2,dim> tmp = 2.0 * density_fluid * viscosity * std::pow((epsilon * epsilon + modulus * modulus),(p-2.0)/2.0) * cauchy_f_vu;
        
        return J_f * tmp * F_f_Inv_T;
    }
    
    template <int dim>
    double
    get_incompressibility_term(const double J_f,
                               const Tensor<2,dim> grad_vf,
                               const Tensor<2,dim> F_f_Inv)
    {
        return J_f * trace(grad_vf * F_f_Inv);
    }
    
    template <int dim>
    Tensor<2,dim>
    get_harmonic_mmpde_term(const double J_f,
                            const Tensor<2,dim> grad_uf)
    {
        return 0.00000001 * (1.0/J_f) * grad_uf;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_linear_elastic_mmpde_term (const double alpha_EY,
                                   const double alpha_nu_s,
                                   const double J_f,
                                   const Tensor<2,dim> E_f_Linear,
                                   const Tensor <2,dim> Identity)
    {
        return ( (alpha_nu_s * alpha_EY)/(J_f*(1 + alpha_nu_s)*(1- 2*alpha_nu_s))) * trace(E_f_Linear) * Identity + 2 * (alpha_EY/(J_f * (1 + alpha_nu_s))) * E_f_Linear;
    }
    
    
    template <int dim>
    Tensor<2,dim>
    get_fluid_pf_stress_term(const double J_f,
                             const double pf,
                             const Tensor<2,dim> Identity,
                             const Tensor<2,dim> F_f_Inv_T)
    
    {
        return J_f * (-pf * Identity) * F_f_Inv_T;
    }
    
    template <int dim>
    Tensor<2,dim>
    get_do_nothing_term (const double J_f,
                         const Tensor<2,dim> F_f_Inv_T,
                         const Tensor<2,dim> grad_vf_T)
    {
        return J_f * F_f_Inv_T * grad_vf_T * F_f_Inv_T;
    }
    
    
    
}

namespace Solid_Terms
{
    
    
    // At first we build some tensors usefull to build the terms in the solid domain
    
    using namespace dealii;
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_E (const Tensor<2,dim> F_s_T,
           const Tensor<2,dim> F_s,
           const Tensor<2,dim> Identity)
    {
        return 0.5 * (F_s_T * F_s - Identity);
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_SIG (const double lame_coefficient_mu,
             const double lame_coefficient_lambda,
             Tensor<2,dim> E,
             Tensor<2,dim> Identity)
    {
        return lame_coefficient_lambda * trace(E) * Identity + 2 * lame_coefficient_mu * E;
    }
    
    // Here we build linearized tensors that will help us buils the terms in the solid jacobean
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_E_LinU (const Tensor<2,dim> F_s,
                const Tensor<2,dim> F_s_T,
                const Tensor<2,dim> F_s_LinU,
                const Tensor<2,dim> F_s_T_LinU)
    
    {
        return 0.5 * (F_s_T_LinU * F_s + F_s_T * F_s_LinU);
    }
    
    
    
    /*
     template <int dim>  // <---------- TO BE CHECKED, POSSIBLE FACTOR 0.5 MISSING!!!!!!!!!!!!
     inline
     double
     get_tr_E_LinU (unsigned int q,
     const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
     const Tensor<2,dim> phi_i_grads_us,
     const double g_growth)
     {
     return (1.0/(g_growth * g_growth))*((1 + old_solution_grads[q][dim+dim+1][0]) *
     phi_i_grads_us[0][0] +
     old_solution_grads[q][dim+dim+1][1] *
     phi_i_grads_us[0][1] +
     (1 + old_solution_grads[q][dim+dim+dim][1]) *
     phi_i_grads_us[1][1] +
     old_solution_grads[q][dim+dim+dim][0] *
     phi_i_grads_us[1][0]);
     }
     */
    
    template <int dim>  // <---------- TO BE CHECKED, POSSIBLE FACTOR 0.5 MISSING!!!!!!!!!!!!
    inline
    double
    get_tr_E_LinU (const Tensor<2,dim> E_LinU)
    {
        return  trace(E_LinU);
    }
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_SIG_LinU (const double lame_coefficient_mu,
                  const double lame_coefficient_lambda,
                  const Tensor<2,dim> E_LinU,
                  const double tr_E_LinU,
                  const Tensor<2,dim> Identity)
    {
        return lame_coefficient_lambda * tr_E_LinU * Identity  + 2 * lame_coefficient_mu * E_LinU;
    }
    
    // Here we build the single terms to build the jacobean of the solid
    
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_solid_stress_term_LinU_1 (const Tensor<2,dim> F_s_LinU,
                                  const Tensor<2,dim> SIG)
    {
        return   F_s_LinU * SIG;
    }
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_solid_stress_term_LinU_2 (const Tensor<2,dim> F_s,
                                  const Tensor<2,dim> SIG_LinU)
    {
        return   F_s * SIG_LinU;
    }
    
    // Here we build terms to make up the solid residual
    
    template <int dim>
    inline
    Tensor<2,dim>
    get_solid_stress_term(const Tensor<2,dim> SIG,
                          const Tensor<2,dim> F_s)
    {
        return F_s * SIG;
    }
    
}


namespace ALE_Problem
{
    using namespace dealii;
    
    
    template <int dim>
    class FsiProblem
    {
    public:
        FsiProblem (const unsigned int stokes_FE_degree,
                    const unsigned int elasticity_FE_degree,
                    std::string MMPDE,
                    unsigned int refinement_level);
        
        ~FsiProblem ();
        void run ();
        
    private:
        
        // First listing the member functions
        
        static bool
        cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
        static bool
        cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
        
        void import_grid();
        void set_runtime_parameters();
        void interface_dof_matcher();
        void setup_system ();
        void set_active_fe_indices ();
        void set_initial_bc(const double time);
        void assemble_jacobean_matrix ();
        void assemble_residual ();
        void set_newton_bc();
        void newton_iteration (const double time);
        void update_u_y();
        void solve ();
        void refine_grid ();
        void output_results (unsigned int timestep_number );
        double compute_point_value (Point<dim> p,
                                    const unsigned int component) const;
        void compute_functional_values(double time);
        void output_results();
        
        
         std::ofstream data_file;
         std::string MMPDE;
         unsigned int refinement_level;
        
        // Listing the variables
        
        enum
        {
            fluid_domain_id,
            solid_domain_id
        };

    // Variables needed for parallelization
        
        MPI_Comm mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;
        ConditionalOStream pcout;
        PETScWrappers::MPI::SparseMatrix system_matrix;
        PETScWrappers::MPI::Vector       solution;
        PETScWrappers::MPI::Vector       system_rhs;
        PETScWrappers::MPI::Vector       newton_update;
        PETScWrappers::MPI::Vector       old_timestep_solution;
        
       
   
        // Usual variables needed to write FE codes in deal.ii
        
        Triangulation<dim>       triangulation;
        hp::DoFHandler<dim>      dof_handler;
        FESystem<dim>            stokes_fe;
        FESystem<dim>            elasticity_fe;
        hp::FECollection<dim>    fe_collection;
        ConstraintMatrix         constraints;
        
        
        const unsigned int    stokes_FE_degree;
        const unsigned int    elasticity_FE_degree;
        
        // Vectors to store dof to constrain in order to achieve
        // continuity of velocities and displacements at interface
        std::vector<unsigned int> fluid_velo,fluid_displ,solid_displ;
        std::vector<unsigned int> total_interface_dofs_us;
        std::vector<unsigned int> total_interface_dofs_vf;
        std::vector<unsigned int> total_interface_dofs_uf;
        
        // Global variables for timestepping scheme
        unsigned int timestep_number;
        unsigned int max_no_timesteps;
        double timestep, theta, time;
        std::string time_stepping_scheme;
        
        // Fluid parameters
        double density_fluid, viscosity;
        
        // Structure parameters
        double density_structure;
        double lame_coefficient_mu, lame_coefficient_lambda, poisson_ratio_nu;
        
        // Artificial Pparameters
        double alpha_EY;          //------> Avoids artificial mesh stiffness
        double alpha_nu_s;
        
        
        const long double pi = 3.141592653589793238462643;
        
        double pressure_fluid_x;
        double alpha_growth;   //    <----- alpha_growth is labelled as Cs(tau) on the paper!!!!!!!!
        double stop_growth;
        double u_y;
        double drag_summed;
        double final_drag_summed;
        double gamma_zero;
        double growth_initial;
        double drag;
        double p;
        double epsilon;
        double bad_term_switch = 1.0;
        double breaker = 0.0;
        unsigned int p_switch = 0;
        double max_newton_iterations = 0;
    
        
        
        // Variable to compute the time needed for each section to run
        TimerOutput                               computing_timer;
    };
    
////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                            //
//    At first we define the helper functions (the one that are not member of the class)      //
//                                                                                            //
////////////////////////////////////////////////////////////////////////////////////////////////
   
    

    
    template <int dim>
    class BoundaryParabolic : public Function<dim>
    {
    public:
        BoundaryParabolic (const double time,
                           const double u_y)
        : Function<dim>(dim+dim+1+dim)
        {
            _time = time;       // Is this an alternative way to initialize variables
            _u_y = u_y;         // in the constructor????
        }
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
        
    private:
        double _time;
        double _u_y;
    };
    
    /* The boundary values are given to component
     with number 0 which is the velocity along x
     */
    
    template <int dim>
    double
    BoundaryParabolic<dim>::value (const Point<dim>  &p,
                                   const unsigned int component) const
    {
        Assert (component < this->n_components,
                ExcIndexRange (component, 0, this->n_components));
        
        const long double pi = 3.141592653589793238462643;
        
        
        double inflow_velocity = 1.5;   // This is the factor 3/2 in the definition of v^in (t,x,y)
        double beta_inflow = 1.0e-1;
        
        if (component == 0)
        {
            return   ( (p(0) == -5) && (p(1) <= 1.0) && (p(1) >= -1.0) ? inflow_velocity *
                      ((beta_inflow + 10.0 * (1.0 -  _u_y)) * (1.0 - p(1)*p(1))) : 0);
        }
        
        return 0;
    }
    
    
    template <int dim>
    void
    BoundaryParabolic<dim>::vector_value (const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        for (unsigned int c=0; c<this->n_components; ++c)
            values (c) = BoundaryParabolic<dim>::value (p, c);
    }

    
    // Here's the constructor of the FluidStructureProblem class, we initialize here the FESystem
    // for the fluid (stokes_fe) and for the solid (elasticity_fe), we set the maximum degree for
    // the FE in fluid (stokes_degree) and solid(elasticity_degree). This is needed in order to
    // properly choose the degree of the quadrature formula when integrating on cells. Inside the
    // function we build the hp_fe_collection, the first element (element 0) is made by the FESystem
    // "stokes_fe", the second (element 1) is made by "elasticity_fe"
    
    //     (  | vf |  )   |
    //     (  | uf |  )   | --> element 0 = stokes_fe
    //     (  | pf |  )   |
    //     (
    //     (  | us |  )   | --> element 1 = elasticity_fe
    //

    
    
    template <int dim>
    FsiProblem<dim>::FsiProblem (const unsigned int stokes_FE_degree,
                                         const unsigned int elasticity_FE_degree,
                                         std::string         MMPDE,
                                         unsigned int refinement_level)
    :
    MMPDE(MMPDE),
    refinement_level(refinement_level),
    mpi_communicator (MPI_COMM_WORLD),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout (std::cout,
           (this_mpi_process == 0)),
    dof_handler (triangulation),
    stokes_fe (FE_Q<dim>(stokes_FE_degree), dim,
               FE_Q<dim>(stokes_FE_degree), dim,
               FE_Q<dim>(stokes_FE_degree-1), 1,
               FE_Nothing<dim>(), dim),
    elasticity_fe (FE_Nothing<dim>(), dim,
                   FE_Nothing<dim>(), dim,
                   FE_Nothing<dim>(), 1,
                   FE_Q<dim>(elasticity_FE_degree), dim),
    stokes_FE_degree(stokes_FE_degree),
    elasticity_FE_degree(elasticity_FE_degree),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::cpu_and_wall_times)
    {
        
        DeclException1 (ExcBadParameter,
                        std::string,
                        << "Parameter " << arg1 << " is not a possible choice" );
        
       
        AssertThrow (MMPDE == "linear_elastic" || MMPDE == "harmonic" ,ExcBadParameter(MMPDE));

        
        
        fe_collection.push_back (stokes_fe);
        fe_collection.push_back(elasticity_fe);
    }
    
    
    
    
    template <int dim>
    FsiProblem<dim>::~FsiProblem ()
    {
        dof_handler.clear ();
    }
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////
    //                                                                                            //
    //         Then we define the member functions (the one that are member of the class)         //
    //                                                                                            //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    template <int dim>
    bool
    FsiProblem<dim>::
    cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
    {
        return (cell->material_id() == fluid_domain_id);
    }
    
    template <int dim>
    bool
    FsiProblem<dim>::
    cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
    {
        return (cell->material_id() == solid_domain_id);
    }
    
    template <int dim>
    void FsiProblem<dim>::set_runtime_parameters ()
    {
        drag_summed = 0.0;
        final_drag_summed = 0.0;
        gamma_zero = 5.0e-7;
        growth_initial = 100.0;
        stop_growth = 1.0e+12;
        pressure_fluid_x = 1.0;
        
        alpha_growth = 0.0; // 0.02
        
        density_fluid = 1.0;
        density_structure = 1.0;
        viscosity = 0.3;  // 1.0
        lame_coefficient_mu = 1.0e+4;  // 1.0e+3
        poisson_ratio_nu = 0.2;
        
        lame_coefficient_lambda =  4.0e+4; //(2 * poisson_ratio_nu * lame_coefficient_mu)/(1.0 - 2 * poisson_ratio_nu);
        
        alpha_EY = 1e-8;
        alpha_nu_s = -0.1;
        
        epsilon = 1.0;
        p = 2.0;
        
        
        // Timestepping schemes
        //BE, CN, CN_shifted
        time_stepping_scheme = "BE";
        
        // Timestep size:
        // TODO
        timestep = 43200.0; //86400.0;
        
        
        // A variable to count the number of time steps
        timestep_number = 0;
        
        // Counts total time
        time = 0;
        
        // Here, we choose a time-stepping scheme that
        // is based on finite differences:
        // BE         = backward Euler scheme
        // CN         = Crank-Nicolson scheme
        // CN_shifted = time-shifted Crank-Nicolson scheme
        
        if (time_stepping_scheme == "BE")
            theta = 1.0;
        else if (time_stepping_scheme == "CN")
            theta = 0.5;
        else if (time_stepping_scheme == "CN_shifted")
            theta = 0.5 + timestep;
        else 
            std::cout << "No such timestepping scheme" << std::endl;

    }

    
    template <int dim>
    void FsiProblem<dim>::interface_dof_matcher()
    {
        
        std::vector<types::global_dof_index> local_face_dof_indices (stokes_fe.dofs_per_face);
        std::vector<types::global_dof_index> neighbor_dof_indices (elasticity_fe.dofs_per_face);
        
        unsigned int faces_at_interface = 0;
        unsigned int fluid_velocity_dofs = 0;
        unsigned int fluid_displacement_dofs = 0;
        unsigned int solid_displacement_dofs = 0;
        unsigned int counter = 0;
        
        
        // We perform a cylce over the cells to retrieve information
        // about the number of faces that are at the interface
        // and to get how many dofs the variables vf,vs,uf and us
        // have at the interface.
        
        for (typename hp::DoFHandler<dim>::active_cell_iterator
             cell = dof_handler.begin_active();
             cell != dof_handler.end(); ++cell)
        {
            if (cell_is_in_fluid_domain (cell))
            {
                for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                {
                    if(!cell->at_boundary(f))
                    {
                        if(cell_is_in_solid_domain(cell->neighbor(f)))
                        {
                            faces_at_interface += 1;
                            
                            if(faces_at_interface == 1)
                            {
                                cell->face(f)->get_dof_indices(local_face_dof_indices,0);
                                cell->neighbor(f)->face(cell->neighbor_of_neighbor(f))->get_dof_indices(neighbor_dof_indices,1);
                                for(unsigned int i=0;i<local_face_dof_indices.size();++i)
                                {
                                    if(stokes_fe.face_system_to_component_index(i).first < dim)
                                        fluid_velocity_dofs+= 1;
                                    if(stokes_fe.face_system_to_component_index(i).first >= dim && stokes_fe.face_system_to_component_index(i).first < dim+dim)fluid_displacement_dofs+= 1;
                                }
                                for(unsigned int i=0;i<neighbor_dof_indices.size();++i)
                                {
                                    if(elasticity_fe.face_system_to_component_index(i).first > dim+dim)
                                        solid_displacement_dofs+= 1;
                                }
                            }
                            
                        }
                    }
                }
            }
        }
        
        // std::cout<<faces_at_interface<<" "<<fluid_velocity_dofs<<" "<<solid_velocity_dofs<<" "<<fluid_displacement_dofs<<" "<<solid_displacement_dofs<< std::endl;
        
        std::vector<unsigned int> temporary_fluid_velo  (faces_at_interface * fluid_velocity_dofs);
        std::vector<unsigned int> temporary_fluid_displ (faces_at_interface * fluid_displacement_dofs);
        std::vector<unsigned int> temporary_solid_displ (faces_at_interface * solid_displacement_dofs);
        unsigned int p=0,q=0,s=0,l=0,n=0,o=0;
        
        
        // Here we fill the vectors with the right dofs, namely vf gets filled with vf dofs and so on...
        for (typename hp::DoFHandler<dim>::active_cell_iterator
             cell = dof_handler.begin_active();
             cell != dof_handler.end(); ++cell)
        {
            if (cell_is_in_fluid_domain (cell))
            {
                for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                    if (!cell->at_boundary(f))
                    {
                        if (cell_is_in_solid_domain (cell->neighbor(f)))
                        {
                            
                            cell->face(f)->get_dof_indices(local_face_dof_indices,0);
                            cell->neighbor(f)->face(cell->neighbor_of_neighbor(f))->get_dof_indices(neighbor_dof_indices,1);
                            
                            // Here we copy into a the vectors the dofs (vf,uf,vs,us) at each vertex.
                            
                            
                            
                            for(unsigned int i=0;i<local_face_dof_indices.size();++i)
                            {
                                if(stokes_fe.face_system_to_component_index(i).first<dim)
                                {
                                    temporary_fluid_velo[p] = local_face_dof_indices[i];
                                    p+=1;
                                }
                                
                                if(stokes_fe.face_system_to_component_index(i).first >= dim && stokes_fe.face_system_to_component_index(i).first <dim+dim )
                                {
                                    temporary_fluid_displ[q] = local_face_dof_indices[i];
                                    q+=1;
                                }
                            }
                            
                            for(unsigned int i=0;i<neighbor_dof_indices.size();++i)
                            {
                                
                                if(elasticity_fe.face_system_to_component_index(i).first > dim+dim)
                                {
                                    temporary_solid_displ[s] = neighbor_dof_indices[i];
                                    s+=1;
                                }
                            }
                            
                            
                        }
                        
                    }
            } // Closes cell is in fluid domain
            
        } // Closes loop on cells
        
        
        
        
        total_interface_dofs_us = temporary_solid_displ;
        total_interface_dofs_vf = temporary_fluid_velo;
        total_interface_dofs_uf = temporary_fluid_displ;
        
        
        // We take care here of the fact that we could in principle
        // use different degrees for the FE on the solid and fluid side
        // for the "complementary variables" uf-us and vf_vs
        
        if(fluid_displacement_dofs < solid_displacement_dofs)
        {
            for(unsigned int i=0; (i*6)<temporary_solid_displ.size();++i)
            {
                temporary_solid_displ[4 + i*6] = 0;
                temporary_solid_displ[5 + i*6] = 0;
            }
            
            // std::cout<<" We are actually using different degrees for uf and us"<<std::endl;
            
        }
        
        if(solid_displacement_dofs < fluid_displacement_dofs)
        {
            for(unsigned int i=0; (i*6)<temporary_fluid_displ.size();++i)
            {
                temporary_fluid_displ[4 + i*6] = 0;
                temporary_fluid_displ[5 + i*6] = 0;
            }
            
            // std::cout<<" We are actually using different degrees for uf and us"<<std::endl;
            
        }

        
        
        
        // Here we take care of the fact that up to now the
        // vectors contain double terms. Thid is clear if
        // we think of how we got the dofs. We tracked the interface and
        // extracted the dofs (on both sides fluid & solid) at the vertices.
        // Therefore if we move from a face to it´s neighboring face this two
        // will have a vertex in common and we would store twice the dofs at that vertex
        
        
        
        
        
        for(unsigned int i=0;i<faces_at_interface*fluid_velocity_dofs;++i)
        {
            for(unsigned int j=0;j<faces_at_interface*fluid_velocity_dofs;++j) if(temporary_fluid_velo[i] == temporary_fluid_velo[j] && i!=j)   temporary_fluid_velo[j] = 0;
        }
        
        
        for(unsigned int i=0;i<faces_at_interface*fluid_displacement_dofs;++i)
        {
            for(unsigned int j=0;j<faces_at_interface*fluid_displacement_dofs;++j) if(temporary_fluid_displ[i] == temporary_fluid_displ[j] && i!=j)   temporary_fluid_displ[j] = 0;
        }
        
        for(unsigned int i=0;i<faces_at_interface*solid_displacement_dofs;++i)
        {
            for(unsigned int j=0;j<faces_at_interface*solid_displacement_dofs;++j) if(temporary_solid_displ[i] == temporary_solid_displ[j] && i!=j)   temporary_solid_displ[j] = 0;
        }
        
        
        
        // We finally fill the vf vector
        for(unsigned int i=0;i<(faces_at_interface*fluid_velocity_dofs);++i)
        {
            if(temporary_fluid_velo[i]!=0)
            {
                counter+=1;
            }
        }
        
        fluid_velo.resize(counter);
        counter =0;
        
        for(unsigned int i=0;i<(faces_at_interface*fluid_velocity_dofs);++i)
        {
            if(temporary_fluid_velo[i]!=0)
            {
                fluid_velo[l] = temporary_fluid_velo[i];
                l+=1;
            }
        }
        
        // We finally fill the uf vector
        for (unsigned int i=0; i<(faces_at_interface*fluid_displacement_dofs);++i)
        {
            if(temporary_fluid_displ[i]!=0)
            {
                counter +=1;
            }
        }
        
        
        fluid_displ.resize(counter);
        counter =0;
        
        for (unsigned int i=0; i<(faces_at_interface*fluid_displacement_dofs);++i)
        {
            if(temporary_fluid_displ[i]!=0)
            {
                fluid_displ[n] = temporary_fluid_displ[i];
                n+=1;
            }
        }
        
        
        
        // We finally fill the us vector
        for (unsigned int i=0; i<(faces_at_interface*solid_displacement_dofs);++i)
        {
            if(temporary_solid_displ[i]!=0)
            {
                counter +=1;
            }
        }
        
        solid_displ.resize(counter);
        counter =0;
        
        for (unsigned int i=0; i<(faces_at_interface*solid_displacement_dofs);++i)
        {
            if(temporary_solid_displ[i]!=0)
            {
                solid_displ[o] = temporary_solid_displ[i];
                o+=1;
            }
        }
        
        
        // for(unsigned int i=0; i<fluid_displ.size();++i) std::cout<<i<<" "<<fluid_displ[i]<<" "<<solid_displ[i]<<std::endl;
        
    }
    

    
    template <int dim>
    void FsiProblem<dim>::import_grid()
    {
        GridIn<2> grid_in;
        grid_in.attach_triangulation(triangulation);
        
        std::ifstream input_file("channel_growth_Sep_2013.inp");
        grid_in.read_ucd(input_file);
        
        triangulation.refine_global(refinement_level);
        
        std::ofstream out ("vessel.eps");
        GridOut grid_out;
        grid_out.write_eps (triangulation, out);
        pcout << "Grid written to vessel.eps" << std::endl;
        
        GridTools::partition_triangulation (n_mpi_processes, triangulation);
      
    }
    
    template <int dim>
    void FsiProblem<dim>::set_active_fe_indices ()
    {
        int solid_cells_number = 0;
        int fluid_cells_number = 0;
     
        for (typename hp::DoFHandler<dim>::active_cell_iterator
             cell = dof_handler.begin_active();
             cell != dof_handler.end(); ++cell)
        {
           
               if (cell_is_in_fluid_domain(cell))
              {
               cell->set_active_fe_index (0);
               fluid_cells_number +=1;
              }
            
              else if (cell_is_in_solid_domain(cell))
              {
                cell->set_active_fe_index (1);
                solid_cells_number +=1;
              }
            
              else
                Assert (false, ExcNotImplemented());
     
        }
      

        pcout<<"-----------------------------------------"<<std::endl;
        pcout<<"# SOLID CELLS "<<solid_cells_number<<std::endl;
        pcout<<"# FLUID CELLS "<<fluid_cells_number<<std::endl;
        

        
    }

    
    template <int dim>
    void FsiProblem<dim>::setup_system ()
    {
        
        TimerOutput::Scope t(computing_timer, "setup");
        
        dof_handler.distribute_dofs (fe_collection);
        DoFRenumbering::subdomain_wise (dof_handler);
        
        
        const types::global_dof_index n_local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler,
                                                                                                      this_mpi_process);
        
        interface_dof_matcher();
        
        
        {
            constraints.clear ();
            set_newton_bc();
            
            for(unsigned int i=0; i<fluid_displ.size();++i)
            {
              constraints.add_line(fluid_displ[i]);
              constraints.add_entry(fluid_displ[i],solid_displ[i],1.0);
            }
            
            if(total_interface_dofs_us.size() > total_interface_dofs_uf.size())
            {
                pcout<<std::endl;
                pcout<<"We are in Us > Uf"<<std::endl;
                pcout<<std::endl;
                
                for (unsigned int i=4; i< total_interface_dofs_us.size(); i+=6)
                {
                    constraints.add_line(total_interface_dofs_us[i]);
                    constraints.add_entry(total_interface_dofs_us[i],total_interface_dofs_us[i-4],0.5);
                    constraints.add_entry(total_interface_dofs_us[i],total_interface_dofs_us[i-2],0.5);
                    
                    constraints.add_line(total_interface_dofs_us[i+1]);
                    constraints.add_entry(total_interface_dofs_us[i+1],total_interface_dofs_us[i+1-4],0.5);
                    constraints.add_entry(total_interface_dofs_us[i+1],total_interface_dofs_us[i+1-2],0.5);
                }
            }

            
        }
        
        constraints.close ();
       
        
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, dsp);
        constraints.condense(dsp);
        dsp.compress();
        
        
        std::vector<IndexSet> dofs_per_subdomain;
        
        dofs_per_subdomain = DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
      
        system_matrix.reinit(dofs_per_subdomain[this_mpi_process],
                             dofs_per_subdomain[this_mpi_process],
                             dsp,
                             mpi_communicator);
       
        
        
        
        solution.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
        old_timestep_solution.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
        system_rhs.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
        newton_update.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
        
        
        pcout<<"#    DOFS    "<<dof_handler.n_dofs()
             << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
            pcout << (p==0 ? ' ' : '+')
            << (DoFTools::
                count_dofs_with_subdomain_association (dof_handler,
                                                       p));
        pcout << ")" << std::endl;
        pcout<<"-----------------------------------------"<<std::endl;
      
        
    }

    template <int dim>
    void
    FsiProblem<dim>::
    assemble_jacobean_matrix ()
    {
        
         TimerOutput::Scope t(computing_timer, "matrix assembly");
        
        system_matrix = 0;
        
        const QGauss<dim> stokes_quadrature(stokes_FE_degree + 1);
        const QGauss<dim> elasticity_quadrature(elasticity_FE_degree + 1);
        
        hp::QCollection<dim>  q_collection;
        
        q_collection.push_back (stokes_quadrature);
        q_collection.push_back (elasticity_quadrature);
        
        hp::FEValues<dim> hp_fe_values (fe_collection, q_collection,
                                        update_values    |
                                        update_quadrature_points  |
                                        update_JxW_values |
                                        update_gradients);
        
        const QGauss<dim-1> face_quadrature(std::max(stokes_FE_degree + 1, elasticity_FE_degree + 1));
        
        FEFaceValues<dim>    stokes_fe_face_values (stokes_fe,
                                                    face_quadrature,
                                                    update_values   |
                                                    update_quadrature_points |
                                                    update_JxW_values |
                                                    update_normal_vectors |
                                                    update_gradients);
        
        FEFaceValues<dim>    elasticity_fe_face_values (elasticity_fe,
                                                        face_quadrature,
                                                        update_values   |
                                                        update_quadrature_points |
                                                        update_JxW_values |
                                                        update_normal_vectors |
                                                        update_gradients);
        
        
        
        const unsigned int   stokes_dofs_per_cell       = stokes_fe.dofs_per_cell;
        const unsigned int   elasticity_dofs_per_cell   = elasticity_fe.dofs_per_cell;
        
        const unsigned int  n_face_q_points  = face_quadrature.size();
        const unsigned int   n_q_points      =  q_collection.max_n_quadrature_points();
        
        
        FullMatrix<double> local_matrix(std::max (stokes_dofs_per_cell,elasticity_dofs_per_cell),
                                        std::max (stokes_dofs_per_cell,elasticity_dofs_per_cell));
        
        
        std::vector<types::global_dof_index> local_dof_indices(std::max (stokes_dofs_per_cell,elasticity_dofs_per_cell));
 
        const FEValuesExtractors::Vector fluid_velocities (0);
        const FEValuesExtractors::Vector fluid_displacements (dim);
        const FEValuesExtractors::Scalar pressure (dim+dim);
        const FEValuesExtractors::Vector solid_displacements (dim+dim+1);
        
        
        // We declare Vectors and Tensors for the solutions at the previous Newton iteration:
        
        std::vector<Vector<double> > old_solution_values (n_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        
        std::vector<Vector<double> >  old_solution_face_values (n_face_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        
        // We declare Vectors and Tensors for the solution at the previous time step
        
        std::vector<Vector<double> > old_timestep_solution_values (n_q_points, Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points,std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        
        Tensor<2,dim> Identity = Fluid_quantities
        ::get_Identity<dim> ();
        
        std::vector<Tensor<1,dim> > phi_i_vf (stokes_dofs_per_cell);
        std::vector<Tensor<2,dim> > phi_i_grads_vf(stokes_dofs_per_cell);
        std::vector<Tensor<1,dim> > phi_i_uf (stokes_dofs_per_cell);
        std::vector<Tensor<2,dim> > phi_i_grads_uf(stokes_dofs_per_cell);
        std::vector<double>         phi_i_p(stokes_dofs_per_cell);
        std::vector<Tensor<1,dim> > phi_i_us (elasticity_dofs_per_cell);
        std::vector<Tensor<2,dim> > phi_i_grads_us (elasticity_dofs_per_cell);
        
        
        Vector<double> localized_solution(solution);
        Vector<double> localized_old_timestep_solution(old_timestep_solution);
        
        
        
        typename hp::DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            if(cell->subdomain_id() == this_mpi_process)
            {
               hp_fe_values.reinit (cell);
            
               const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            
               local_matrix.reinit (cell->get_fe().dofs_per_cell,
                                    cell->get_fe().dofs_per_cell);
            
            
               // Old Newton iteration values
               fe_values.get_function_values (localized_solution, old_solution_values);
               fe_values.get_function_gradients (localized_solution, old_solution_grads);
            
               // Old_timestep_solution values
               fe_values.get_function_values (localized_old_timestep_solution, old_timestep_solution_values);
               fe_values.get_function_gradients (localized_old_timestep_solution, old_timestep_solution_grads);
            
               // Next, we run over all cells for the fluid equations
            
            
            
            if(cell_is_in_fluid_domain(cell))
            {
                const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
                
                Assert (dofs_per_cell == stokes_dofs_per_cell,
                        ExcInternalError());
                
                
                // We build here the Jacobean on the fluid cells
                for(unsigned int q=0; q<fe_values.n_quadrature_points; q++)
                {
                    for (unsigned int k=0; k<stokes_dofs_per_cell; ++k)
                    {
                        phi_i_vf[k]       = fe_values[fluid_velocities].value (k, q);
                        phi_i_grads_vf[k] = fe_values[fluid_velocities].gradient (k, q);
                        phi_i_uf[k]       = fe_values[fluid_displacements].value (k, q);
                        phi_i_grads_uf[k] = fe_values[fluid_displacements].gradient (k, q);
                        phi_i_p[k]        = fe_values[pressure].value (k, q);
                    }
                    
                    
                    
                    // Here we get the quantities dependent only on the quadrature points and
                    // not on the shape functions. We do it now for the fluid variables that
                    // appear in the part of the bilinear form which lives on the
                    // fluid domain, namely (Vf,Uf,P) and derived quantities like gradients a.s.o
                    // old_solution_values refers to the newton_step.
                    
                    const double pf = Fluid_quantities
                    ::get_pf<dim>(q, old_solution_values);
                    
                    const Tensor<1,dim> vf = Fluid_quantities
                    ::get_vf<dim> (q, old_solution_values);
                    
                    const Tensor<1,dim> uf = Fluid_quantities
                    ::get_uf<dim> (q,old_solution_values);
                    
                    const Tensor<2,dim> grad_vf = Fluid_quantities
                    ::get_grad_vf<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> grad_uf = Fluid_quantities
                    ::get_grad_uf<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> grad_uf_T = Fluid_quantities
                    ::get_grad_uf_T<dim> (grad_uf);
                    
                    const Tensor<2,dim> grad_vf_T = Fluid_quantities
                    ::get_grad_vf_T<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> F_f = Fluid_quantities
                    ::get_F_f<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> F_f_Inv = Fluid_quantities
                    ::get_F_f_Inv<dim> (F_f);
                    
                    const Tensor<2,dim> F_f_Inv_T = Fluid_quantities
                    ::get_F_f_Inv_T<dim> (F_f_Inv);
                    
                    const double J_f = Fluid_quantities
                    ::get_J_f<dim> (F_f);
                    
                    // Stress tensor for the fluid in ALE notation (not linearized)
                    
                    const Tensor<2,dim> cauchy_f_vu = Fluid_Terms
                    ::get_cauchy_f_vu(viscosity, density_fluid, grad_vf, grad_vf_T, F_f_Inv, F_f_Inv_T);
                    
                    const Tensor <2,dim> E_f_Linear = Fluid_Terms
                    ::get_E_f_Linear(grad_uf,grad_uf_T);
                    
                    // Further, we also need some information from the previous time steps
                    
                    const Tensor<1,dim> old_timestep_vf = Fluid_quantities
                    ::get_vf<dim> (q, old_timestep_solution_values);
                    
                    const Tensor<1,dim> old_timestep_uf = Fluid_quantities
                    ::get_uf<dim>(q, old_timestep_solution_values);
                    
                    const Tensor<2,dim> old_timestep_grad_uf = Fluid_quantities
                    ::get_grad_uf<dim> (q, old_timestep_solution_grads);
                    
                    const Tensor<2,dim> old_timestep_F_f = Fluid_quantities
                    ::get_F_f<dim> (q, old_timestep_solution_grads);
                    
                    const double old_timestep_J_f = Fluid_quantities
                    ::get_J_f<dim> (old_timestep_F_f);
                    
                    // Outer loop for dofs, we need to do this on the dofs becouse
                    // the variations are made up of test functions so we need to do
                    // it in the loops on the dofs, we do it on the outer one to not perform
                    // this twice.
                    
                    for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                    {
                        
                        
                        
                        //Here we build the variated terms that will be used in building the single terms in the Jacoben on the fluid domain
                        
                        const Tensor<1,dim> vf_LinV = Fluid_quantities
                        ::get_vf_LinV(phi_i_vf[i]);
                        
                        const Tensor<1,dim> uf_LinU = Fluid_quantities
                        ::get_uf_LinU(phi_i_uf[i]);
                        
                        const double pf_LinP = Fluid_quantities
                        ::get_pf_LinP<dim> (phi_i_p[i]);
                        
                        const Tensor<2,dim> grad_vf_LinV = Fluid_quantities
                        ::get_grad_vf_LinV<dim> (phi_i_grads_vf[i]);
                        
                        const Tensor<2,dim> grad_uf_LinU = Fluid_quantities
                        ::get_grad_uf_LinU(phi_i_grads_uf[i]);
                        
                        const Tensor<2,dim> grad_uf_T_LinU = Fluid_quantities
                        ::get_grad_uf_T_LinU(phi_i_grads_uf[i]);
                        
                        const Tensor<2,dim> grad_vf_T_LinV = Fluid_quantities
                        ::get_grad_vf_T_LinV<dim> (phi_i_grads_vf[i]);
                        
                        const double J_f_LinU =  Fluid_quantities
                        ::get_J_f_LinU<dim> (q, old_solution_grads, phi_i_grads_uf[i]);
                        
                        const Tensor<2,dim> F_f_LinU = Fluid_quantities
                        ::get_F_f_LinU(phi_i_grads_uf[i]);
                        
                        const Tensor<2,dim> F_f_Inv_LinU = Fluid_quantities
                        ::get_F_f_Inv_LinU (phi_i_grads_uf[i], J_f, J_f_LinU, q, old_solution_grads);
                        
                        const Tensor<2,dim> F_f_Inv_T_LinU  = Fluid_quantities
                        ::get_F_f_Inv_T_LinU(F_f_Inv_LinU);
                        
                        const Tensor<2,dim> cauchy_f_vu_LinV = Fluid_Terms
                        ::get_cauchy_f_vu_LinV(density_fluid, viscosity, grad_vf_LinV, grad_vf_T_LinV, F_f_Inv, F_f_Inv_T);
                        
                        const Tensor<2,dim> cauchy_f_vu_LinU = Fluid_Terms
                        ::get_cauchy_f_vu_LinU(density_fluid, viscosity, grad_vf, grad_vf_T, F_f_Inv_LinU, F_f_Inv_T_LinU);
                        
                        const double tr_grad_vf_LinV_F_Inv = Fluid_quantities
                        ::get_tr_grad_vf_LinV_F_f_Inv(grad_vf_LinV, F_f_Inv);
                        
                        const double tr_grad_vf_F_Inv = Fluid_quantities
                        ::get_tr_grad_vf_F_f_Inv(grad_vf, F_f_Inv);
                        
                        const double tr_grad_vf_F_Inv_LinU = Fluid_quantities
                        ::get_tr_grad_vf_F_Inv_LinU(grad_vf, F_f_Inv_LinU);
                        
                        const Tensor<2,dim> E_f_Linear_LinU = Fluid_Terms
                        ::get_E_f_Linear_LinU(grad_uf_LinU,grad_uf_T_LinU);
                        
                        
                        
                        
                        // Now we build the single terms that compose the Jacobean on the fluid domain
                        
                        const Tensor<2,dim> sigma_lin_el_LinU1 = Fluid_Terms
                        ::get_sigma_lin_el_LinU1(J_f,F_f_LinU,E_f_Linear,Identity);
                        
                        const Tensor<2,dim> sigma_lin_el_LinU2 = Fluid_Terms
                        ::get_sigma_lin_el_LinU2(J_f,J_f_LinU,F_f,E_f_Linear,E_f_Linear_LinU,Identity);
                        
                        const Tensor<1,dim> acceleration_term_LinV = Fluid_Terms
                        ::get_acceleration_term_LinV(vf_LinV, J_f, old_timestep_J_f, theta);
                        
                        const Tensor<1,dim> convective_mesh_term_LinV = Fluid_Terms
                        ::get_convective_mesh_term_LinV(grad_vf_LinV, J_f, F_f_Inv, old_timestep_uf, uf);
                        
                        const Tensor<1,dim> convective_term_LinV_1 = Fluid_Terms
                        ::get_convective_term_LinV_1(J_f, grad_vf_LinV, F_f_Inv, vf);
                        
                        const Tensor<1,dim> convective_term_LinV_2 = Fluid_Terms
                        ::get_convective_term_LinV_2(J_f, grad_vf, F_f_Inv, vf_LinV);
                        
                        const Tensor<2,dim> fluid_stress_vu_term_LinV_1 = Fluid_Terms
                        ::get_fluid_stress_vu_term_LinV_1(epsilon, p, density_fluid, viscosity, J_f, cauchy_f_vu, cauchy_f_vu_LinV, F_f_Inv_T);
                        
                        const Tensor<2,dim> fluid_stress_vu_term_LinV_2 = Fluid_Terms
                        ::get_fluid_stress_vu_term_LinV_2(epsilon, p, density_fluid, viscosity, J_f, cauchy_f_vu, cauchy_f_vu_LinV, F_f_Inv_T);
                        
                        const double incompressibility_term_LinV =Fluid_Terms
                        ::get_incompressibility_term_LinV(J_f, tr_grad_vf_LinV_F_Inv);
                        
                        const Tensor<1,dim> acceleration_term_LinU = Fluid_Terms
                        ::get_acceleration_term_LinU(theta, J_f_LinU, vf, old_timestep_vf);
                        
                        const Tensor<1,dim> convective_mesh_term_LinU_1 = Fluid_Terms
                        ::get_convective_mesh_term_LinU_1(J_f_LinU, grad_vf, uf, old_timestep_uf, F_f_Inv);
                        
                        const Tensor<1,dim> convective_mesh_term_LinU_2 = Fluid_Terms
                        ::get_convective_mesh_term_LinU_2(J_f, grad_vf, F_f_Inv_LinU, uf, old_timestep_uf);
                        
                        const Tensor<1,dim> convective_mesh_term_LinU_3 = Fluid_Terms
                        ::get_convective_mesh_term_LinU_3(J_f, grad_vf, F_f_Inv, phi_i_uf[i]);
                        
                        const Tensor<2,dim> fluid_stress_pf_term_LinU_1 = Fluid_Terms
                        ::get_fluid_stress_pf_term_LinU_1(J_f_LinU, pf, Identity, F_f_Inv_T);
                        
                        const Tensor<2,dim> fluid_stress_pf_term_LinU_2 = Fluid_Terms
                        ::get_fluid_stress_pf_term_LinU_2(J_f, pf, Identity, F_f_Inv_T_LinU);
                        
                        const Tensor<1,dim> convective_term_LinU_1 = Fluid_Terms
                        ::get_convective_term_LinU_1(J_f_LinU, grad_vf, F_f_Inv, vf);
                        
                        const Tensor<1,dim> convective_term_LinU_2 = Fluid_Terms
                        ::get_convective_term_LinU_2(J_f, grad_vf, F_f_Inv_LinU, vf);
                        
                        const Tensor<2,dim> fluid_stress_vu_term_LinU_1 = Fluid_Terms
                        ::get_fluid_stress_vu_term_LinU_1(epsilon, p, density_fluid, viscosity, J_f, cauchy_f_vu, cauchy_f_vu_LinU, F_f_Inv_T);
                        
                        const Tensor<2,dim> fluid_stress_vu_term_LinU_2 = Fluid_Terms
                        ::get_fluid_stress_vu_term_LinU_2(epsilon, p, density_fluid, viscosity, J_f, cauchy_f_vu, cauchy_f_vu_LinU, F_f_Inv_T);
                        
                        const Tensor<2,dim> fluid_stress_vu_term_LinU_3 = Fluid_Terms
                        ::get_fluid_stress_vu_term_LinU_3(epsilon, p, density_fluid, viscosity, J_f_LinU, cauchy_f_vu, F_f_Inv_T, pf, Identity);
                        
                        const Tensor<2,dim> fluid_stress_vu_term_LinU_4 = Fluid_Terms
                        ::get_fluid_stress_vu_term_LinU_4(epsilon, p, density_fluid, viscosity, J_f, cauchy_f_vu, F_f_Inv_T_LinU, Identity, pf);
                        
                        
                        const Tensor<2,dim> harmonic_mmpde_term_LinU_1 = Fluid_Terms
                        ::get_harmonic_mmpde_term_LinU_1(J_f, J_f_LinU, grad_uf);
                        
                        const Tensor<2,dim> harmonic_mmpde_term_LinU_2 = Fluid_Terms
                        ::get_harmonic_mmpde_term_LinU_2(J_f, grad_uf_LinU);
                        
                        const Tensor<2,dim> linear_elastic_mmpde_term_LinU_1 = Fluid_Terms
                        ::get_linear_elastic_mmpde_term_LinU_1(alpha_nu_s, alpha_EY, J_f, J_f_LinU, E_f_Linear, Identity);
                        
                        const Tensor<2,dim> linear_elastic_mmpde_term_LinU_2 = Fluid_Terms
                        ::get_linear_elastic_mmpde_term_LinU_2(alpha_nu_s, alpha_EY, J_f, E_f_Linear_LinU, Identity);
                        
                        const Tensor<2,dim> linear_elastic_mmpde_term_LinU_3 = Fluid_Terms
                        ::get_linear_elastic_mmpde_term_LinU_3(alpha_nu_s, alpha_EY, J_f, J_f_LinU, E_f_Linear, Identity);
                        
                        const Tensor<2,dim> linear_elastic_mmpde_term_LinU_4 = Fluid_Terms
                        ::get_linear_elastic_mmpde_term_LinU_4(alpha_nu_s, alpha_EY, J_f, E_f_Linear_LinU, Identity);
                        
                        const double incompressibility_term_LinU_1 = Fluid_Terms
                        ::get_incompressibility_term_LinU_1(J_f_LinU, tr_grad_vf_F_Inv);
                        
                        const double incompressibility_term_LinU_2 = Fluid_Terms
                        ::get_incompressibility_term_LinU_2(J_f,tr_grad_vf_F_Inv_LinU);
                        
                        const Tensor<2,dim> fluid_stress_pf_term_LinP = Fluid_Terms
                        ::get_fluid_stress_pf_term_LinP(J_f, pf_LinP, Identity, F_f_Inv_T);
                        
                        // Now we are set to build the Jacobean regarding the fluid domain
                        
                        for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                        {
                            
                            
                            
                            local_matrix(j,i) +=  (
                                                   
                                                   + timestep * theta * density_fluid * convective_term_LinV_1 * phi_i_vf[j]
                                                   + timestep * theta * density_fluid * convective_term_LinV_2 * phi_i_vf[j]
                                                   + timestep * theta * density_fluid * convective_term_LinU_1 * phi_i_vf[j]
                                                   + timestep * theta * density_fluid * convective_term_LinU_2 * phi_i_vf[j]
                                                   
                                                   
                                                   + timestep * scalar_product(fluid_stress_pf_term_LinU_1,phi_i_grads_vf[j])
                                                   + timestep * scalar_product(fluid_stress_pf_term_LinU_2,phi_i_grads_vf[j])
                                                   + timestep * scalar_product(fluid_stress_pf_term_LinP,phi_i_grads_vf[j])
                                                   
                                                   + timestep * theta * scalar_product(/*bad_term_switch */ fluid_stress_vu_term_LinV_1,phi_i_grads_vf[j])
                                                   + timestep * theta * scalar_product(fluid_stress_vu_term_LinV_2,phi_i_grads_vf[j])
                                                   + timestep * theta * scalar_product(fluid_stress_vu_term_LinU_1,phi_i_grads_vf[j])
                                                   + timestep * theta * scalar_product(fluid_stress_vu_term_LinU_2,phi_i_grads_vf[j])
                                                   + timestep * theta * scalar_product(fluid_stress_vu_term_LinU_3,phi_i_grads_vf[j])
                                                   + timestep * theta * scalar_product(fluid_stress_vu_term_LinU_4,phi_i_grads_vf[j])
                                                   
                                                   
                                                   
                                                   
                                                   + incompressibility_term_LinV * phi_i_p[j]
                                                   + incompressibility_term_LinU_1 * phi_i_p[j]
                                                   + incompressibility_term_LinU_2 * phi_i_p[j]
                                                   
                                                   ) * fe_values.JxW(q);
                            
                            if(MMPDE == "harmonic") local_matrix(j,i) +=  (
                                
                                                                               scalar_product(harmonic_mmpde_term_LinU_1, phi_i_grads_uf[j])
                                                                             + scalar_product(harmonic_mmpde_term_LinU_2, phi_i_grads_uf[j])
                                
                                                                           ) * fe_values.JxW(q);
                            
                            
                            if(MMPDE == "linear_elastic") local_matrix(j,i) +=  (
                                                                                                 
                                                                                     scalar_product(linear_elastic_mmpde_term_LinU_1,phi_i_grads_uf[j])
                                                                                   + scalar_product(linear_elastic_mmpde_term_LinU_2,phi_i_grads_uf[j])
                                                                                   + scalar_product(linear_elastic_mmpde_term_LinU_3,phi_i_grads_uf[j])
                                                                                   + scalar_product(linear_elastic_mmpde_term_LinU_4,phi_i_grads_uf[j])
                                                                                                 
                                                                                ) * fe_values.JxW(q);

                            
                            
                        } //ends the j loop
                        
                    }  //ends the i loop
                    
                }  //Closes the quadrature points
                
                for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                {
                    if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 1))
                    {
                        stokes_fe_face_values.reinit (cell, face);
                        
                        stokes_fe_face_values.get_function_values (localized_solution,  old_solution_face_values);
                        stokes_fe_face_values.get_function_gradients (localized_solution,  old_solution_face_grads);
                        
                        
                        for (unsigned int q=0; q< stokes_fe_face_values.get_quadrature().size(); ++q)
                        {
                            for (unsigned int k=0; k<stokes_dofs_per_cell; ++k)
                            {
                                phi_i_vf[k]       =  stokes_fe_face_values[fluid_velocities].value (k, q);
                                phi_i_grads_vf[k] =  stokes_fe_face_values[fluid_velocities].gradient (k, q);
                                phi_i_uf[k]       =  stokes_fe_face_values[fluid_displacements].value (k, q);
                                phi_i_grads_uf[k] =  stokes_fe_face_values[fluid_displacements].gradient (k, q);
                                
                            }
                            
                            
                            // Again befor looping on the dofs we define the quantities that depend only on the quadrature points
                            
                            const Tensor<1,dim> normal_vector =  stokes_fe_face_values.normal_vector(q);
                            
                            const Tensor<2,dim> grad_vf_T = Fluid_quantities
                            ::get_grad_vf_T(q,  old_solution_face_grads);
                            
                            const Tensor<2,dim> F_f = Fluid_quantities
                            ::get_F_f(q,  old_solution_face_grads);
                            
                            const Tensor<2,dim> F_f_Inv = Fluid_quantities
                            ::get_F_f_Inv(F_f);
                            
                            const Tensor<2,dim> F_f_Inv_T = Fluid_quantities
                            ::get_F_f_Inv_T(F_f_Inv);
                            
                            const double J_f = Fluid_quantities
                            ::get_J_f(F_f);
                            
                            
                            for (unsigned int i=0; i<stokes_fe_face_values.dofs_per_cell; ++i)
                            {
                                
                                
                                // Here we can build the linearized quantities becouse those involve test functions "i", we do it in
                                
                                const double J_f_LinU = Fluid_quantities
                                ::get_J_f_LinU(q,  old_solution_face_grads,  phi_i_grads_uf[i]);
                                
                                const Tensor<2,dim> F_f_Inv_LinU = Fluid_quantities
                                ::get_F_f_Inv_LinU( phi_i_grads_uf[i], J_f, J_f_LinU, q,  old_solution_face_grads);
                                
                                const Tensor<2,dim> F_f_Inv_T_LinU = Fluid_quantities
                                ::get_F_f_Inv_T_LinU(F_f_Inv_LinU);
                                
                                const Tensor<2,dim> grad_vf_T_LinV = Fluid_quantities
                                ::get_grad_vf_T_LinV( phi_i_grads_vf[i]);
                                
                                // We construct the terms that build up the jacobean of the do-nothing condition part
                                
                                const Tensor<2,dim> do_nothing_term_LinU_1 = Fluid_Terms
                                ::get_do_nothing_term_LinU_1(J_f_LinU, F_f_Inv_T, grad_vf_T);
                                
                                const Tensor<2,dim> do_nothing_term_LinU_2 = Fluid_Terms
                                ::get_do_nothing_term_LinU_2(J_f, F_f_Inv_T_LinU, F_f_Inv_T, grad_vf_T);
                                
                                const Tensor<2,dim> do_nothing_term_LinU_3 = Fluid_Terms
                                ::get_do_nothing_term_LinU_3(J_f, F_f_Inv_T_LinU, F_f_Inv_T, grad_vf_T);
                                
                                const Tensor<2,dim> do_nothing_term_LinV = Fluid_Terms
                                ::get_do_nothing_term_LinV(J_f, F_f_Inv_T, grad_vf_T_LinV);
                                
                                
                                // We are now ready to build the jacobean of the do nothing term
                                
                                for (unsigned int j=0; j<stokes_fe_face_values.dofs_per_cell; ++j)
                                {
                                    
                                    
                                    local_matrix(j,i) +=  - timestep * theta * density_fluid * viscosity * (
                                                                                                            
                                                                                                            
                                                                                                            (
                                                                                                               do_nothing_term_LinV
                                                                                                             + do_nothing_term_LinU_1
                                                                                                             + do_nothing_term_LinU_2
                                                                                                             + do_nothing_term_LinU_3
                                                                                                             
                                                                                                             ) * normal_vector *  phi_i_vf[j]
                                                                                                            
                                                                                                            
                                                                                                            ) *  stokes_fe_face_values.JxW(q);
                                    
                                    
                                } // Closes the loop on the j dofs
                                
                                
                            } // Closes the loop on i dofs
                            
                        } // Closes the loop on face quadrature points
                        
                    } // Closes the if that choses the outflow boundary
                    
                } // Closes the loop on fluid faces
                
                
                
            }  // CLoses if the cell is in fluid domain
            
            else
            {
                
                 const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
                
                Assert (dofs_per_cell == elasticity_dofs_per_cell,
                        ExcInternalError());
                
                
                
                for(unsigned int q=0; q<n_q_points; q++)
                {
                    
                    double g_growth = 0.0;
                    
                    if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && (time <=stop_growth))
                    {
                        g_growth = 1.0 + alpha_growth *
                        std::exp(-fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0])
                        * (2.0 - std::abs(fe_values.quadrature_point(q)[1]));
                        //g_growth = 1.0;
                    }
                    else if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && (time > stop_growth))
                    {
                        abort();
                        
                    }
                    else 
                        g_growth = 1.0;
                
                
            
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {
                        phi_i_us[k]       = fe_values[solid_displacements].value (k, q);
                        phi_i_grads_us[k] = fe_values[solid_displacements].gradient (k, q);
                        
                    }
                    
                    
                    // Here again as in the fluid case we define the quantities that are dependent on the quadrature points only
                    
                   
                    
                    const Tensor<1,dim> us = Solid_quantities
                    ::get_us<dim> (q,old_solution_values);
                    
                    const Tensor<2,dim> grad_us = Solid_quantities
                    ::get_grad_us<dim> (q,old_solution_grads);
                    
                    const Tensor<2,dim> F_s = Solid_quantities
                    ::get_F_s<dim>(q, old_solution_grads, g_growth);

                    
                    const Tensor<2,dim> F_s_T = Solid_quantities
                    ::get_F_s_T<dim> (F_s);
                    
                    const Tensor<2,dim> E = Solid_Terms
                    ::get_E(F_s_T,F_s,Identity);
                    
                    const Tensor<2,dim> SIG = Solid_Terms
                    ::get_SIG(lame_coefficient_mu, lame_coefficient_lambda, E, Identity);
                    
                    // Outer loop for dofs, we need to do this on the dofs becouse
                    //the variations are made up of test functions so we need to do
                    //it in the loops on the dofs, we do it on the outer one to not perform
                    //this twice.
                    
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        const unsigned int comp_i = stokes_fe.system_to_component_index(i).first;
                        
                        //Here we build the variated terms that will be used in building the single terms in the Jacobean
                        
                        const Tensor<1,dim> us_LinU = Solid_quantities
                        ::get_us_LinU(phi_i_us[i]);
                        
                        const Tensor<2,dim> grad_us_LinU = Solid_quantities
                        ::get_grad_us_LinU(phi_i_grads_us[i]);
                        
                        const Tensor<2,dim> grad_us_T_LinU = Solid_quantities
                        ::get_grad_us_T_LinU(phi_i_grads_us[i]);
                        
                        const Tensor<2,dim> F_s_LinU = Solid_quantities
                        ::get_F_s_LinU(q, phi_i_grads_us[i],g_growth);
                        
                        const Tensor<2,dim> F_s_T_LinU = Solid_quantities
                        ::get_F_s_T_LinU(q, phi_i_grads_us[i],g_growth);
                        
                        const Tensor<2,dim> E_LinU = Solid_Terms
                        ::get_E_LinU(F_s, F_s_T, F_s_LinU, F_s_T_LinU);
                        
                        const double tr_E_LinU = Solid_Terms
                        ::get_tr_E_LinU(E_LinU);
                        
                        const Tensor<2,dim> SIG_LinU = Solid_Terms
                        ::get_SIG_LinU(lame_coefficient_mu,lame_coefficient_lambda,E_LinU,tr_E_LinU,Identity);
                        
                        // Now we build the terms we need for the jacobean of the solid
                        
                        Tensor<2,dim> solid_stress_term_LinU_1 = Solid_Terms
                        ::get_solid_stress_term_LinU_1(F_s_LinU, SIG);
                        
                        const Tensor<2,dim> solid_stress_term_LinU_2 = Solid_Terms
                        ::get_solid_stress_term_LinU_2(F_s, SIG_LinU);
                        
                        // Now begins the inner loop for the dofs, here we assemble the local matrix
                        
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            
                            
                            
                            
                            local_matrix(j,i) +=  (
                                                  
                                                   + timestep * theta * scalar_product(solid_stress_term_LinU_1, phi_i_grads_us[j])
                                                   + timestep * theta * scalar_product(solid_stress_term_LinU_2, phi_i_grads_us[j])
                                                   
                                                   ) * fe_values.JxW(q);
                        } // closes the j loop
                        
                    } // closes the i loop
                    
                } // closes the quadrature points loop
                
                
                
            }    // Closes if CELL_IS_IN_SOLID_DOMAIN
            
            
            local_dof_indices.resize (cell->get_fe().dofs_per_cell);
            cell->get_dof_indices (local_dof_indices);
            constraints.distribute_local_to_global (local_matrix, local_dof_indices,
                                                    system_matrix);
            
            } // Closes If subdomain identification
            
        }  //closes the loop over cells
        
        system_matrix.compress(VectorOperation::add);
        
    }
    
    template <int dim>
    void
    FsiProblem<dim>::
    assemble_residual ()
    {
        TimerOutput::Scope t(computing_timer, "residuum assembly");
       
        system_rhs=(0) ;
        
        const QGauss<dim> stokes_quadrature(stokes_FE_degree + 1);
        const QGauss<dim> elasticity_quadrature(elasticity_FE_degree + 1);
        
        hp::QCollection<dim>  q_collection;
        
        q_collection.push_back (stokes_quadrature);
        q_collection.push_back (elasticity_quadrature);
        
        hp::FEValues<dim> hp_fe_values (fe_collection, q_collection,
                                        update_values    |
                                        update_quadrature_points  |
                                        update_JxW_values |
                                        update_gradients);
        
        const QGauss<dim-1> face_quadrature(std::max(stokes_FE_degree + 1,elasticity_FE_degree + 1));
        
        FEFaceValues<dim>    stokes_fe_face_values (stokes_fe,
                                                    face_quadrature,
                                                    update_values         |
                                                    update_quadrature_points  |
                                                    update_JxW_values  |
                                                    update_normal_vectors |
                                                    update_gradients);
        
        FEFaceValues<dim>    elasticity_fe_face_values (elasticity_fe,
                                                        face_quadrature,
                                                        update_values         |
                                                        update_quadrature_points  |
                                                        update_JxW_values  |
                                                        update_normal_vectors |
                                                        update_gradients);
        
        const unsigned int   stokes_dofs_per_cell   = stokes_fe.dofs_per_cell;
        const unsigned int   elasticity_dofs_per_cell   = elasticity_fe.dofs_per_cell;
        
        
        Vector<double>  local_rhs(std::max (stokes_dofs_per_cell, elasticity_dofs_per_cell));
        
        std::vector<types::global_dof_index> local_dof_indices(std::max (stokes_dofs_per_cell, elasticity_dofs_per_cell));
        
        const FEValuesExtractors::Vector fluid_velocities (0);
        const FEValuesExtractors::Vector fluid_displacements (dim);
        const FEValuesExtractors::Scalar pressure (dim+dim);
        const FEValuesExtractors::Vector solid_displacements (dim+dim+1);
        
        unsigned int  n_face_q_points = face_quadrature.size();
        const unsigned int   n_q_points     = q_collection.max_n_quadrature_points();
        
        
        // We declare Vectors and Tensors for the solutions at the previous Newton iteration:
        
        std::vector<Vector<double> > old_solution_values (n_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points,std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        std::vector<Vector<double> > old_solution_face_values (n_face_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points,std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        
        
        // We declare Vectors and Tensors for the solution at the previous time step:
        
        std::vector<Vector<double> > old_timestep_solution_values (n_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points,std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        std::vector<Vector<double> > old_timestep_solution_face_values (n_face_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_face_grads (n_face_q_points,std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        
        
        const Tensor<2,dim> Identity = Fluid_quantities
        ::get_Identity<dim> ();
        
        Vector<double> localized_solution (solution);  // Check if this causes a bottleneck, alternatively move the get_function_values function outside the If clause!
        Vector<double> localized_old_timestep_solution (old_timestep_solution);  // Check if this causes a bottleneck, alternatively move the get_function_values function outside the If clause!
       
     
       
        
        
        typename hp::DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            
        if(cell->subdomain_id() == this_mpi_process)
          {
              
              hp_fe_values.reinit (cell);
              const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
              local_rhs.reinit (cell->get_fe().dofs_per_cell);
              
              // Old Newton iteration values
              fe_values.get_function_values (localized_solution, old_solution_values);
              fe_values.get_function_gradients (localized_solution, old_solution_grads);
              
              // Old_timestep_solution values
              fe_values.get_function_values (localized_old_timestep_solution, old_timestep_solution_values);
              fe_values.get_function_gradients (localized_old_timestep_solution, old_timestep_solution_grads);

              
               if(cell_is_in_fluid_domain(cell))
               {
                const unsigned int actual_dofs_per_cell = cell->get_fe().dofs_per_cell;
                Assert (actual_dofs_per_cell == stokes_fe.dofs_per_cell,
                        ExcInternalError());
              
                // We build here the residual on the cells in the fluid domain
                for(unsigned int q=0; q<n_q_points; ++q)
                {
        
                    // Here we get the quantities dependent only on the quadrature points and
                    // not on the shape functions. We do it now for the fluid variables that
                    // appear in the part of the bilinear form which lives on the
                    // fluid domain, namely (Vf,Uf,P) and derived quantities like gradients a.s.o
                    // old_solution_values refers to the newton_step.
        
                    const double pf = Fluid_quantities
                    ::get_pf<dim>(q, old_solution_values);
                    
                    const Tensor<1,dim> vf = Fluid_quantities
                    ::get_vf<dim> (q, old_solution_values);
                    
                    const Tensor<1,dim> uf = Fluid_quantities
                    ::get_uf<dim> (q,old_solution_values);
                    
                    const Tensor<2,dim> grad_vf = Fluid_quantities
                    ::get_grad_vf<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> grad_uf = Fluid_quantities
                    ::get_grad_uf<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> grad_uf_T = Fluid_quantities
                    ::get_grad_uf_T<dim> (grad_uf);
                    
                    const Tensor<2,dim> grad_vf_T = Fluid_quantities
                    ::get_grad_vf_T<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> F_f = Fluid_quantities
                    ::get_F_f<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> F_f_Inv = Fluid_quantities
                    ::get_F_f_Inv<dim> (F_f);
                    
                    const Tensor<2,dim> F_f_Inv_T = Fluid_quantities
                    ::get_F_f_Inv_T<dim> (F_f_Inv);
                    
                    const double J_f = Fluid_quantities
                    ::get_J_f<dim> (F_f);
                    
                    // Stress tensor for the fluid in ALE notation (not linearized)
                    
                    const Tensor<2,dim> cauchy_f_vu = Fluid_Terms
                    ::get_cauchy_f_vu(viscosity, density_fluid, grad_vf, grad_vf_T, F_f_Inv, F_f_Inv_T);
                    
                    const Tensor<2,dim> E_f_Linear = Fluid_Terms
                    ::get_E_f_Linear(grad_uf,grad_uf_T);
                    
                    
                    
                    // Further, we also need some information from the previous time steps
                    
                    const Tensor<1,dim> old_timestep_vf = Fluid_quantities
                    ::get_vf<dim> (q, old_timestep_solution_values);
                    
                    const Tensor<1,dim> old_timestep_uf = Fluid_quantities
                    ::get_uf<dim>(q, old_timestep_solution_values);
                    
                    const Tensor<2,dim> old_timestep_grad_vf = Fluid_quantities
                    ::get_grad_vf<dim> (q, old_timestep_solution_grads);
                    
                    const Tensor<2,dim> old_timestep_grad_vf_T = Fluid_quantities
                    ::get_grad_vf_T(q, old_timestep_solution_grads);
                    
                    const Tensor<2,dim> old_timestep_grad_uf = Fluid_quantities
                    ::get_grad_uf<dim> (q, old_timestep_solution_grads);
                    
                    const Tensor<2,dim> old_timestep_F_f = Fluid_quantities
                    ::get_F_f (q, old_timestep_solution_grads);
                    
                    const Tensor<2,dim> old_timestep_F_f_Inv = Fluid_quantities
                    ::get_F_f_Inv(old_timestep_F_f);
                    
                    const Tensor<2,dim> old_timestep_F_f_Inv_T = Fluid_quantities
                    ::get_F_f_Inv_T(old_timestep_F_f_Inv);
                    
                    const double old_timestep_J_f = Fluid_quantities
                    ::get_J_f<dim> (old_timestep_F_f);
                    
                    const Tensor<2,dim> old_timestep_cauchy_f_vu = Fluid_Terms
                    ::get_cauchy_f_vu(viscosity, density_fluid, old_timestep_grad_vf, old_timestep_grad_vf_T, old_timestep_F_f_Inv, old_timestep_F_f_Inv_T);
                    
                    // Now we build the terms that make up the Residual of the Newton's method
                    
                    
                    const Tensor<1,dim> acceleration_term = Fluid_Terms
                    ::get_acceleration_term(theta, J_f, old_timestep_J_f, old_timestep_vf, vf);
                    
                    const Tensor<1,dim> convective_mesh_term = Fluid_Terms
                    ::get_convective_mesh_term(J_f, old_timestep_uf, uf, F_f_Inv, grad_vf);
                    
                    const Tensor<1,dim> convective_term = Fluid_Terms
                    ::get_convective_term(J_f, grad_vf, F_f_Inv, vf);
                    
                    const Tensor<2,dim> fluid_stress_term = Fluid_Terms
                    ::get_fluid_vu_stress_term(epsilon, p, density_fluid, viscosity, J_f, cauchy_f_vu, F_f_Inv_T);
                    
                    const double incompressibility_term = Fluid_Terms
                    ::get_incompressibility_term(J_f, grad_vf, F_f_Inv);
                    
                    const Tensor<2,dim> harmonic_mmpde_term = Fluid_Terms
                    ::get_harmonic_mmpde_term(J_f, grad_uf);
                    
                    const Tensor<2,dim> linear_elastic_mmpde_term = Fluid_Terms
                    ::get_linear_elastic_mmpde_term(alpha_EY, alpha_nu_s, J_f, E_f_Linear, Identity);
                    
                    const Tensor<2,dim> pressure_stress_term = Fluid_Terms
                    ::get_fluid_pf_stress_term(J_f, pf, Identity, F_f_Inv_T);
                    
                    // Now we build the terms at the old_timestep that make up the Residual of the Newton's method
                    
                    const Tensor<1,dim> old_timestep_convective_term = Fluid_Terms
                    ::get_convective_term(old_timestep_J_f, old_timestep_grad_vf, old_timestep_F_f_Inv, old_timestep_vf);
                    
                    const Tensor<2,dim> old_timestep_fluid_stress_term = Fluid_Terms::
                    get_fluid_vu_stress_term(epsilon, p, density_fluid, viscosity, old_timestep_J_f, old_timestep_cauchy_f_vu, old_timestep_F_f_Inv_T);
                    
                    

                    
                    for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                    {
                        
                        const Tensor<1,dim> phi_i_vf = fe_values[fluid_velocities].value (i, q);
                        const Tensor<2,dim> phi_i_grads_vf = fe_values[fluid_velocities].gradient (i, q);
                        const Tensor<2,dim> phi_i_grads_uf = fe_values[fluid_displacements].gradient (i, q);
                        const double phi_i_p = fe_values[pressure].value (i, q);
                     
                        local_rhs(i) -= (
                                          + timestep * theta * density_fluid * convective_term * phi_i_vf
                                          + timestep * (1.0-theta) * density_fluid * old_timestep_convective_term * phi_i_vf
                                         
                                         
                                         
                                          + timestep * scalar_product(pressure_stress_term, phi_i_grads_vf)
                                         
                                          + timestep * theta * scalar_product(fluid_stress_term,phi_i_grads_vf)
                                          + timestep * (1.0-theta) * scalar_product(old_timestep_fluid_stress_term, phi_i_grads_vf)
                                         
                                          + incompressibility_term * phi_i_p
                                         
                                         
                                         
                                        ) * fe_values.JxW(q);
                        
                        if(MMPDE == "harmonic")        local_rhs(i) -=   (scalar_product(harmonic_mmpde_term, phi_i_grads_uf)) * fe_values.JxW(q);
                        if(MMPDE == "linear_elastic") local_rhs(i) -= (scalar_product(linear_elastic_mmpde_term, phi_i_grads_uf)) * fe_values.JxW(q);
                    
                    } // Closes loop on i
                    
                    
         
                } // Closes Loop over quadrature points
         
         
              
                for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                {
                    // As said here we build the face integrals on the outflow boundary
                    if (cell->face(face)->at_boundary() &&
                        (cell->face(face)->boundary_id() == 1))
                    {
                        stokes_fe_face_values.reinit (cell, face);
                        
                        stokes_fe_face_values.get_function_values (localized_solution,old_solution_face_values);
                        stokes_fe_face_values.get_function_gradients (localized_solution,old_solution_face_grads);
                        
                        stokes_fe_face_values.get_function_values (localized_old_timestep_solution,old_timestep_solution_face_values);
                        stokes_fe_face_values.get_function_gradients (localized_old_timestep_solution,old_timestep_solution_face_grads);
                        
                        
                        
                        for (unsigned int q=0; q< stokes_fe_face_values.get_quadrature().size(); ++q)
                        {
                            
                            
                            // Again before looping on the dofs we define the quantities that depend only on the quadrature points
                            
                            const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);
                            
                            const Tensor<2,dim> grad_vf_T = Fluid_quantities
                            ::get_grad_vf_T(q, old_solution_face_grads);
                            
                            const Tensor<2,dim> F_f = Fluid_quantities
                            ::get_F_f(q, old_solution_face_grads);
                            
                            const double J_f = Fluid_quantities
                            ::get_J_f<dim> (F_f);
                            
                            const Tensor<2,dim> F_f_Inv = Fluid_quantities
                            ::get_F_f_Inv(F_f);
                            
                            const Tensor<2,dim> F_f_Inv_T = Fluid_quantities
                            ::get_F_f_Inv_T(F_f_Inv);
                            
                            // We need to have some information about ols_timestep
                            
                            const Tensor<2,dim> old_timestep_grad_vf_T = Fluid_quantities
                            ::get_grad_vf_T(q, old_timestep_solution_face_grads);
                            
                            const Tensor<2,dim> old_timestep_F_f = Fluid_quantities
                            ::get_F_f(q, old_timestep_solution_face_grads);
                            
                            const double old_timestep_J_f = Fluid_quantities
                            ::get_J_f<dim> (old_timestep_F_f);
                            
                            const Tensor<2,dim> old_timestep_F_f_Inv = Fluid_quantities
                            ::get_F_f_Inv(old_timestep_F_f);
                            
                            const Tensor<2,dim> old_timestep_F_f_Inv_T = Fluid_quantities
                            ::get_F_f_Inv_T(old_timestep_F_f_Inv);
                            
                            const Tensor<2,dim> do_nothing_term = Fluid_Terms
                            ::get_do_nothing_term(J_f, F_f_Inv_T, grad_vf_T);
                            
                            const Tensor<2,dim> old_timestep_do_nothing_term = Fluid_Terms
                            ::get_do_nothing_term(old_timestep_J_f, old_timestep_F_f_Inv_T, old_timestep_grad_vf_T);
                            
                            
                            
                            
                            
                            
                            for (unsigned int i=0; i<stokes_fe_face_values.dofs_per_cell; ++i)
                            {
                                const Tensor<1,dim> phi_i_vf =  stokes_fe_face_values[fluid_velocities].value (i, q);
                           
                               local_rhs(i) -= - timestep * density_fluid * viscosity * (
                                                                                                   theta * do_nothing_term * normal_vector * phi_i_vf
                                                                                                   + (1.0-theta) * old_timestep_do_nothing_term * normal_vector * phi_i_vf
                                                                                                   
                                                                                                   ) * stokes_fe_face_values.JxW(q);
                            
                                
                                
                                
                                
                            } // Closes the loop on i dofs
                            
                        } // Closes the loop on face quadrature points
                         
                         
                        
                    } // Closes the if that choses the outflow boundary
                } // Closes the loop on the faces
                
              
                
            } // Closes if cell is in fluid domain
            
               else    //------> We are now selecting the solid domain
               {
                
                const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
                Assert (dofs_per_cell == elasticity_fe.dofs_per_cell,
                        ExcInternalError());
                
                
                //We build here the residual  on the cells in the solid domain
                for(unsigned int q=0; q<n_q_points; ++q)
                {
                    // Growth
                    //double g_growth = 1.0 * (1.0 + 0.01 * time);
                    double g_growth = 0.0;
                    
                    if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && (time <=stop_growth))
                    {
                        g_growth = 1.0 + alpha_growth *
                        std::exp(-fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0])
                        * (2.0 - std::abs(fe_values.quadrature_point(q)[1]));
                        //g_growth = 1.0;
                    }
                    
                    else if ((std::abs(fe_values.quadrature_point(q)[1]) > 1.0) && (time > stop_growth))
                    {
                        abort();
                        
                    }
                    
                    else
                    {
                        g_growth = 1.0;
                        
                        pcout<<std::endl;
                        pcout<<"-----------------------"<<std::endl;
                        pcout<<"SOMEHOW WE ENDED IN THE OPTION g_growth = 1.0"<<std::endl;
                        pcout<<"-----------------------"<<std::endl;
                        pcout<<std::endl;
                        
                        
                        
                    }

                    
                    // Here again as in the fluid case we define the quantities that are dependent on the quadrature points only
                    
                    
                    const Tensor<1,dim> us = Solid_quantities
                    ::get_us<dim> (q, old_solution_values);
                    
                    const Tensor<2,dim> grad_us = Solid_quantities
                    ::get_grad_us<dim> (q, old_solution_grads);
                    
                    const Tensor<2,dim> F_s = Solid_quantities
                    ::get_F_s<dim> (q, old_solution_grads,g_growth);
                    
                    const Tensor<2,dim> F_s_T = Solid_quantities
                    ::get_F_s_T(F_s);
                    
                    const Tensor<2,dim> E = Solid_Terms
                    ::get_E(F_s_T, F_s, Identity);
                    
                    const Tensor<2,dim> SIG = Solid_Terms
                    ::get_SIG(lame_coefficient_mu, lame_coefficient_lambda, E, Identity);
                    
                    const Tensor<2,dim> solid_stress_term = Solid_Terms
                    ::get_solid_stress_term(SIG,F_s);
                    
                    // We need also some information about the previous timestep
                    
                    
                    const Tensor<1,dim> old_timestep_us = Solid_quantities
                    ::get_us<dim>(q, old_timestep_solution_values);
                    
                    const Tensor<2,dim> old_timestep_F_s = Solid_quantities
                    ::get_F_s<dim> (q, old_timestep_solution_grads,g_growth);
                    
                    const Tensor<2,dim> old_timestep_F_s_T = Solid_quantities
                    ::get_F_s_T(old_timestep_F_s);
                    
                    const Tensor<2,dim> old_timestep_grad_us = Solid_quantities
                    ::get_grad_us(q, old_timestep_solution_grads);
                    
                    const Tensor<2,dim> old_timestep_E = Solid_Terms
                    ::get_E(old_timestep_F_s_T, old_timestep_F_s, Identity);
                    
                    const Tensor<2,dim> old_timestep_SIG = Solid_Terms
                    ::get_SIG(lame_coefficient_mu, lame_coefficient_lambda, old_timestep_E, Identity);
                    
                    const Tensor<2,dim> old_timestep_solid_stress_term = Solid_Terms
                    ::get_solid_stress_term(old_timestep_SIG, old_timestep_F_s);
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        
                        
                     
                        const Tensor<2,dim> phi_i_grads_us = fe_values[solid_displacements].gradient (i, q);
                        const Tensor<1,dim> phi_i_us = fe_values[solid_displacements].value (i, q);
                        
                        
                        local_rhs(i) -= (
                                          timestep * theta * scalar_product(solid_stress_term,phi_i_grads_us)   // vs -> us
                                         + timestep * (1.0-theta) * scalar_product(old_timestep_solid_stress_term,phi_i_grads_us)  // vs -> us
                                         
                                         ) * fe_values.JxW(q);
                        
                        
                        
                        
                    } // Closes the i loop
                    
                    
                } // Closes the quadrature points
                
                
                
            }  // closes the else which restricts us to the solid domain
            
             
               local_dof_indices.resize (cell->get_fe().dofs_per_cell);
               cell->get_dof_indices (local_dof_indices);
               constraints.distribute_local_to_global (local_rhs, local_dof_indices,
                                                       system_rhs);
          
             
          
          
          }     //Closes the if on the MPI process involved
        }   // Closes the loop over cells
        
       
          system_rhs.compress(VectorOperation::add);
        
     }


    template <int dim>
    void FsiProblem<dim>::set_initial_bc (const double time)
    {
        
        std::map<unsigned int,double> boundary_values;
        std::vector<bool> component_mask (dim+dim+1+dim,false);
        
        // Fluid Velocities
        component_mask[0] = true;
        component_mask[1] = true;
        //Fluid Displacements
        component_mask[2] = true;
        component_mask[3] = true;
        
        /*
         We set here the inhomogeneous Dirichlet condition on the  fluid velocity
         at the inflow boundary, namely a parabolic velocity profile plus we set
         to zero the fluid displacement
         */
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  0,
                                                  BoundaryParabolic<dim>(time,u_y),
                                                  boundary_values,
                                                  component_mask);
        
        
        
        
        // Fluid Velocities
        component_mask[0] = false;
        component_mask[1] = false;
        
        /*
         We impose here no mesh motion at the outflow interface
         */
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  boundary_values,
                                                  component_mask);
        
        
        // Fluid Displacements
        component_mask[2] = false;
        component_mask[3] = false;
        
        
        // We impose clamping condition on the external part of the walls long-side
        
        component_mask[5] = true;
        component_mask[6] = true;
        
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  2,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  boundary_values,
                                                  component_mask);
        
        // We impose clamping condition on the external part of the walls short-side
        
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  80,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  boundary_values,
                                                  component_mask);
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  82,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  boundary_values,
                                                  component_mask);
        
        for(unsigned int i=0; i< fluid_velo.size() ; ++i) solution[fluid_velo[i]] = 0.0;
        
        
        
        for (typename std::map<unsigned int, double>::const_iterator
             z = boundary_values.begin();
             z != boundary_values.end();
             ++z)
            solution(z->first) = z->second;
        


        
    }
    
    
    template <int dim>
    void FsiProblem<dim>::set_newton_bc ()
    {
        
        std::vector<bool> component_mask (dim+dim+1+dim,false);
        
        // Fluid Velocities
        component_mask[0] = true;
        component_mask[1] = true;
        //FLuid Displacements
        component_mask[2] = true;
        component_mask[3] = true;
        
        
        /*
         The filter now selects only fluid velocities and displacements
         in the first step we apply homogeneous Dirichlet conditions to the
         fluid velocities and displacements on the boundary "0" which represents
         inflow boundary
         */
        
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  0,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  constraints,
                                                  component_mask);
        
        
        // Fluid Velocities
        component_mask[0] = false;
        component_mask[1] = false;
        
        // We set no mesh motion on the outflow boundary
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  constraints,
                                                  component_mask);
        
        // Fluid Displacements
        component_mask[2] = false;
        component_mask[3] = false;
        // Solid Displacements
        component_mask[5] = true;
        component_mask[6] = true;
        
        // We impose clamping condition on the external part of the walls long-side
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  2,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  constraints,
                                                  component_mask);
        
        // We impose clamping condition on the external part of the walls short-side
        
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  80,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  constraints,
                                                  component_mask);
        
        
        // We impose clamping condition on the external part of the walls short-side
        
        
        
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  82,
                                                  ZeroFunction<dim>(dim+dim+1+dim),
                                                  constraints,
                                                  component_mask);
        
        
        std::vector<types::global_dof_index> local_face_dof_indices (stokes_fe.dofs_per_face);
        
        for (typename hp::DoFHandler<dim>::active_cell_iterator
             cell = dof_handler.begin_active();
             cell != dof_handler.end(); ++cell)
        {
            
            if (cell_is_in_fluid_domain (cell))
            {
                for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                {
                    if (!cell->at_boundary(f))
                    {
                        if (cell_is_in_solid_domain (cell->neighbor(f))) cell->face(f)->get_dof_indices (local_face_dof_indices, 0);
                        for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
                        {
                            if (stokes_fe.face_system_to_component_index(i).first == 0 || stokes_fe.face_system_to_component_index(i).first == 1 )
                                constraints.add_line (local_face_dof_indices[i]);
                        }
                        
                    }
                }
            }
        }
        
        
    }


     template <int dim>
    void FsiProblem<dim>::solve ()
    {
        
        TimerOutput::Scope t(computing_timer, "solving");
        
        SolverControl           solver_control;
       
        PETScWrappers::MPI::Vector sol;
        PETScWrappers::MPI::Vector rhs;
        
        sol = newton_update;
        rhs = system_rhs;
        
        PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
        solver.solve(system_matrix, sol, rhs);
        
        Vector<double> tmp_solution(sol);
        constraints.distribute(tmp_solution);
        newton_update = tmp_solution;
 
    }
    
   
     template <int dim>
    void FsiProblem<dim>::output_results (unsigned int timestep_number)
    {
        const Vector<double> localized_solution (solution);
        
        if (this_mpi_process == 0)
        {
            std::vector<std::string> solution_names (dim, "fluid_velocity");
            solution_names.push_back ("fluid_displacement");
            solution_names.push_back ("fluid_displacement");
            solution_names.push_back ("fluid_pressure");
            solution_names.push_back ("solid_displacement");
            solution_names.push_back ("solid_displacement");
            
            std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation
            (dim, DataComponentInterpretation::component_is_part_of_vector);
            
            data_component_interpretation
            .push_back (DataComponentInterpretation::component_is_part_of_vector);
            data_component_interpretation
            .push_back (DataComponentInterpretation::component_is_part_of_vector);
            
            data_component_interpretation
            .push_back (DataComponentInterpretation::component_is_scalar);
            
            data_component_interpretation
            .push_back (DataComponentInterpretation::component_is_part_of_vector);
            data_component_interpretation
            .push_back (DataComponentInterpretation::component_is_part_of_vector);
            
          
    
    
            DataOut<dim,hp::DoFHandler<dim> > data_out;
            data_out.attach_dof_handler (dof_handler);
            
            data_out.add_data_vector (localized_solution, solution_names,DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
                                      data_component_interpretation);
            
            std::vector<unsigned int> partition_int (triangulation.n_active_cells());
            GridTools::get_subdomain_association (triangulation, partition_int);
            
            const Vector<double> partitioning(partition_int.begin(),
                                              partition_int.end());
            
            data_out.add_data_vector (partitioning, "partitioning");
            
            data_out.build_patches ();
            
            std::string filename_basis;
            filename_basis  = "vessel_MPI_solution_";
            
            std::ostringstream filename;
            
            pcout << "------------------" << std::endl;
            pcout << "Write solution" << std::endl;
            pcout << "------------------" << std::endl;
            pcout << std::endl;
            filename << filename_basis
            <<refinement_level
            <<"_"
            << Utilities::int_to_string (timestep_number,5)
            << ".vtk";
            
            std::ofstream output (filename.str().c_str());
            
            data_out.write_vtk (output);
        }
    }
    
    
    
    template <int dim>
    void FsiProblem<dim>::newton_iteration (const double time)
    
    {
        Timer timer_newton;
        const double lower_bound_newton_residuum = 1.0e-4;
        const unsigned int max_no_newton_steps  = 155;
        
        // Decision whether the system matrix should be build
        // at each Newton step
        const double nonlinear_rho = 0.1;
        
        // Line search parameters
        unsigned int line_search_step;
        const unsigned int  max_no_line_search_steps = 50;
        const double line_search_damping = 0.6;
        double new_newton_residuum;
        double Qn;
        

        
        // Application of the initial boundary conditions to the
        // variational equations:
        set_initial_bc (time);
        assemble_residual();
        
        double newton_residuum = system_rhs.linfty_norm();
        
        pcout<<"The norm of RHS is" <<" " <<system_rhs.linfty_norm() <<std::endl;
        pcout<<std::endl;
      
        double old_newton_residuum = newton_residuum;
        
        unsigned int newton_step = 1;
        
        if (newton_residuum < lower_bound_newton_residuum)
        {
               pcout << '\t'
            << std::scientific
            << newton_residuum
            << std::endl;
        }
     
        while (newton_residuum > lower_bound_newton_residuum &&
               newton_step < max_no_newton_steps)
        {
            
            timer_newton.start();
            
            
            if(newton_step > 3 && newton_residuum/old_newton_residuum < 1.00000001 && newton_residuum/old_newton_residuum > 0.99999999 ) break;
            
            old_newton_residuum = newton_residuum;
            
            assemble_residual();
            newton_residuum = system_rhs.linfty_norm();
            
            if (newton_residuum < lower_bound_newton_residuum)
            {
                
                if(p_switch !=0)
                {
                    pcout << '\t'
                    << std::scientific
                    << newton_residuum << std::endl;
                    
                }
                break;
            }
         
        
            if (newton_residuum/old_newton_residuum > nonlinear_rho)
                assemble_jacobean_matrix();
            
            // Solve Ax = b
            solve ();
            
            
            line_search_step = 0;
            for ( ;
                 line_search_step < max_no_line_search_steps;
                 ++line_search_step)
            {
                
                
                solution +=(newton_update);
                assemble_residual();
                new_newton_residuum = system_rhs.linfty_norm();
                
                if (new_newton_residuum < newton_residuum)
                    break;
                else
                    solution -=(newton_update);
                
                newton_update *= line_search_damping;
            }
            
            
            Qn = newton_residuum/old_newton_residuum;
            
            bad_term_switch = 0.2 * bad_term_switch + (2*bad_term_switch)/(0.7 + exp(1.5*Qn));

            
            timer_newton.stop();
            
            if (p_switch !=0)
            {
                pcout << std::setprecision(5) <<newton_step << '\t'
                << std::scientific << newton_residuum << '\t'
                << std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
                if (newton_residuum/old_newton_residuum > nonlinear_rho)
                    pcout << "r" << '\t' ;
                else
                    pcout << " " << '\t' ;
                pcout << line_search_step  << '\t'
                << std::scientific << timer_newton ()
                <<" " << bad_term_switch
                << std::endl;
            }
            
            // Updates
            timer_newton.reset();
            newton_step++;
            
            
        } // Closes the While loop
        
        breaker = newton_residuum;
        
        if(newton_step > max_newton_iterations) max_newton_iterations = newton_step;
        

        
    }


    
    template <int dim>
    double FsiProblem<dim>::compute_point_value (Point<dim> p,
                                                    const unsigned int component) const
    {
        
        Vector<double> tmp_vector(dim+dim+1+dim);
       
        Vector<double> tmp_solution(solution);
        
        
        VectorTools::point_value (dof_handler,
                                  tmp_solution,
                                  p,
                                  tmp_vector);
        
        return tmp_vector(component);
        
       
        
    }
    
    template<int dim>
    void FsiProblem<dim>::update_u_y()
    {
        u_y = std::abs(compute_point_value(Point<dim>(0.0,-1.0), dim+dim+dim));  //<------------ check this out
    }

    
    template <int dim>
    void FsiProblem<dim>::compute_functional_values(double time)
    {
        TimerOutput::Scope t(computing_timer, "Functional values computation");
        
        
        std::vector<double> global_drag_lift_tensor(2,0);
        std::vector<double> drag_lift_vec_on_process(2,0);
        Tensor<1,dim> drag_lift_on_process;
        drag_lift_on_process = 0.0;
        
        std::vector<double> global_drag_lift_tensorFS(2,0);
        std::vector<double> drag_lift_vec_on_processFS(2,0);
        Tensor<1,dim> drag_lift_on_processFS;
        drag_lift_on_processFS = 0.0;
        

        double Jf_on_q_point;
        double min_Jf_on_process = 2.0;
        double global_min_Jf = 2.0;
        
        double outflow_on_process = 0.0;
        double global_outflow;
        
        double vorticity_on_process = 0.0;
        double global_vorticity;
        
        double us_x;
        double us_y;
        double width;
        
        Point<dim> growth_point(0.0,-1.0);   // This poin represents the A(t) control point mentioned in the paper
        
        const QGauss<dim> stokes_quadrature(stokes_FE_degree + 1);
        const QGauss<dim> elasticity_quadrature(elasticity_FE_degree + 1);
        
        hp::QCollection<dim>  q_collection;
        
        q_collection.push_back (stokes_quadrature);
        q_collection.push_back (elasticity_quadrature);
        
        hp::FEValues<dim> hp_fe_values (fe_collection, q_collection,
                                        update_values    |
                                        update_quadrature_points  |
                                        update_JxW_values |
                                        update_gradients);
        
        const QGauss<dim-1> face_quadrature(std::max(stokes_FE_degree + 1,elasticity_FE_degree + 1));
        
        FEFaceValues<dim>    stokes_fe_face_values (stokes_fe,
                                                    face_quadrature,
                                                    update_values         |
                                                    update_quadrature_points  |
                                                    update_JxW_values  |
                                                    update_normal_vectors |
                                                    update_gradients);
        
        FEFaceValues<dim>    elasticity_fe_face_values (elasticity_fe,
                                                        face_quadrature,
                                                        update_values         |
                                                        update_quadrature_points  |
                                                        update_JxW_values  |
                                                        update_normal_vectors |
                                                        update_gradients);
        
        const unsigned int   stokes_dofs_per_cell   = stokes_fe.dofs_per_cell;
        const unsigned int   elasticity_dofs_per_cell   = elasticity_fe.dofs_per_cell;
        
        
      
        
        const FEValuesExtractors::Vector fluid_velocities (0);
        const FEValuesExtractors::Vector fluid_displacements (dim);
        const FEValuesExtractors::Scalar pressure (dim+dim);
        const FEValuesExtractors::Vector solid_displacements (dim+dim+1);
        
        
        const unsigned int   n_q_points     = q_collection.max_n_quadrature_points();
        unsigned int  n_face_q_points = face_quadrature.size();
        
        
        // We declare Vectors and Tensors for the solutions at the previous Newton iteration:
        std::vector<Vector<double> > old_solution_values (n_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points,std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        std::vector<Vector<double> > old_solution_face_values (n_face_q_points,Vector<double>(dim+dim+1+dim));
        std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points,std::vector<Tensor<1,dim> > (dim+dim+1+dim));
        
        
       const Tensor<2,dim> Identity = Fluid_quantities
       ::get_Identity<dim> ();
        
        Vector<double> localized_solution (solution);  // Check if this causes a bottleneck, alternatively move the get_function_values function outside the If clause!
       
        
        
        typename hp::DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            if(cell->subdomain_id() == this_mpi_process)
            {
                if(cell_is_in_fluid_domain(cell))
                {
                    const unsigned int actual_dofs_per_cell = cell->get_fe().dofs_per_cell;
                    Assert (actual_dofs_per_cell == stokes_fe.dofs_per_cell,
                            ExcInternalError());
                    
                    
                    for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                        {
                            
                            // We look here for solid-fluid interface and build drag_lift_tensor
                            if(!cell->face(f)->at_boundary())
                            {
                                if(cell_is_in_solid_domain(cell->neighbor(f))) // -------> We are selecting now the interface between solid and fluid from the fluid side
                                {
                                    stokes_fe_face_values.reinit (cell, f);
                                    
                                    
                                    stokes_fe_face_values.get_function_values (localized_solution, old_solution_face_values);
                                    stokes_fe_face_values.get_function_gradients (localized_solution, old_solution_face_grads);
                                    
                                    
                                    for (unsigned int q=0; q<stokes_fe_face_values.get_quadrature().size(); ++q)
                                    {
                                        
                                        
                                        const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);
                                        
                                        const Tensor<2,dim> Identity = Fluid_quantities
                                        ::get_Identity<dim> ();
                                        
                                        const Tensor<2,dim> grad_vf = Fluid_quantities
                                        ::get_grad_vf(q,old_solution_face_grads);
                                        
                                        const Tensor<2,dim> grad_vf_T = Fluid_quantities
                                        ::get_grad_vf_T(q, old_solution_face_grads);
                                        
                                        const Tensor<2,dim> F_f = Fluid_quantities
                                        ::get_F_f(q, old_solution_face_grads);
                                        
                                        const double J_f = Fluid_quantities
                                        ::get_J_f<dim> (F_f);
                                        
                                        const Tensor<2,dim> F_f_Inv = Fluid_quantities
                                        ::get_F_f_Inv(F_f);
                                        
                                        const Tensor<2,dim> F_f_Inv_T = Fluid_quantities
                                        ::get_F_f_Inv_T(F_f_Inv);
                                        
                                        const double pf = Fluid_quantities
                                        ::get_pf<dim>(q,old_solution_face_values);
                                        
                                        const Tensor<1,dim> vf = Fluid_quantities
                                        ::get_vf<dim>(q,old_solution_face_values);
                                        
                                        const Tensor<2,dim> cauchy_f_vu = Fluid_Terms
                                        ::get_cauchy_f_vu(viscosity,density_fluid,grad_vf,grad_vf_T,F_f_Inv,F_f_Inv_T);
                                        
                                        const double modulo = std::sqrt(scalar_product(cauchy_f_vu, cauchy_f_vu));
                                        
                                        const Tensor<2,dim> fluid_new_stress_term = 2.0 * density_fluid * viscosity * std::pow((epsilon * epsilon + modulo * modulo),(p-2.0)/2.0) * cauchy_f_vu;
                                        
                                        
                                        drag_lift_on_process   -= (J_f * (-pf*Identity + fluid_new_stress_term) * F_f_Inv_T) * normal_vector * stokes_fe_face_values.JxW(q);
                                        drag_lift_on_processFS -= (J_f * (-pf*Identity + fluid_new_stress_term) * F_f_Inv_T) * normal_vector * stokes_fe_face_values.JxW(q);
                                        

                                        
                                    } // Closes the loop on face quadrature points
                                    
                                }// Closes if the neighbor cell is in solid domain
                                
                            }// cell not  at b
                            
                            // We select the outflow boundary and evaluate the outflow here
                            if(cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
                            {
                                
                                stokes_fe_face_values.reinit (cell, f);
                                stokes_fe_face_values.get_function_values (localized_solution, old_solution_face_values);
                                
                                for (unsigned int q=0; q<stokes_fe_face_values.get_quadrature().size(); ++q)
                                {
                                    
                                    const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);
                                    
                                    const Tensor<1,dim> vf = Fluid_quantities
                                    ::get_vf<dim>(q, old_solution_face_values);
                                    
                                    
                                    outflow_on_process += vf * normal_vector * stokes_fe_face_values.JxW(q);
                                    
                                } // Closes the loop on face quadrature points
                                
                                
                            }// cell at boundary_id == 1

                        }
                    
                    
                        hp_fe_values.reinit (cell);
                        
                        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
                        const unsigned int n_q_points = fe_values.n_quadrature_points;
                        fe_values.get_function_gradients (localized_solution, old_solution_grads);
                        
                        for(unsigned int q=0; q<n_q_points; ++q)
                        {
                            const Tensor<2,dim> F_f = Fluid_quantities
                            ::get_F_f(q, old_solution_grads);
                            
                            const double J_f = Fluid_quantities
                            ::get_J_f<dim> (F_f);
                            
                            const Tensor<2,dim> grad_uf = Fluid_quantities
                            ::get_grad_uf(q, old_solution_grads);
                            
                            const Tensor<2,dim> grad_vf = Fluid_quantities
                            ::get_grad_vf(q, old_solution_grads);
                            
                            Tensor<2,dim> JF;
                            JF[0][0] = 1.0 + grad_uf[1][1];
                            JF[0][1] = - grad_uf[0][1];
                            JF[1][0] = - grad_uf[1][0];
                            JF[1][1] = 1.0 + grad_uf[0][0];
                            
                            double v1y = grad_vf[0][0] * JF[0][1] + grad_vf[0][1] * JF[1][1];
                            double v2x = grad_vf[1][0] * JF[0][0] + grad_vf[1][1] * JF[1][0];
                            
                            double a = grad_vf[0][0] * (1.0 + grad_uf[1][1]) - grad_vf[0][1] * grad_uf[1][0];
                            double b = -grad_vf[0][0] * grad_uf[0][1] + grad_vf[0][1] * (1.0 + grad_uf[0][0]);
                            double c = grad_vf[1][0] * (1.0 + grad_uf[1][1]) - grad_vf[1][1] * grad_uf[1][0];
                            double d = -grad_vf[1][0] * grad_uf[0][1] + grad_vf[1][1] * (1.0 + grad_uf[0][0]);
                            
                            double A = 1.0/J_f * (a*d - b*c);
                            double g = A * A * A / (A * A + 1.0);
                            
                            double g_pos = 0.0;
                            if (A <= 0.0)
                                g_pos = 0.0;
                            else 
                                g_pos = g;
                            
                            // Compute vorticity in terms of okubo weiss criterion
                            vorticity_on_process += J_f * g_pos * fe_values.JxW(q);
                            
                            // Check for lowest value of J_f
                            Jf_on_q_point = J_f;
                            if(Jf_on_q_point < min_Jf_on_process) min_Jf_on_process = Jf_on_q_point;
                            
                            
                        }
                        
                    } // Closes if cell is in fluid domain
                
                if(cell_is_in_solid_domain(cell))
                {
                    for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                    {
                        if(!cell->face(f)->at_boundary())
                        {
                            if(cell_is_in_fluid_domain(cell->neighbor(f))) //------> We are selecting now the interface betwen solid and fluid from the solid side
                            {
                                elasticity_fe_face_values.reinit (cell, f);
                                stokes_fe_face_values.reinit (cell->neighbor(f),cell->neighbor_of_neighbor(f));
                                
                                elasticity_fe_face_values.get_function_values (localized_solution, old_solution_face_values);
                                elasticity_fe_face_values.get_function_gradients (localized_solution, old_solution_face_grads);
                                
                                for (unsigned int q=0; q< elasticity_fe_face_values.get_quadrature().size(); ++q)
                                {
                                    // Again before looping on the dofs we define the quantities that depend only on the quadrature points
                                    
                                   
                                    double g_growth = 0.0;
                                    
                                    if ((std::abs(elasticity_fe_face_values.quadrature_point(q)[1]) > 1.0) && (time <=stop_growth))
                                    {
                                        g_growth = 1.0 + alpha_growth *
                                        std::exp(-elasticity_fe_face_values.quadrature_point(q)[0] * elasticity_fe_face_values.quadrature_point(q)[0])
                                        * (2.0 - std::abs(elasticity_fe_face_values.quadrature_point(q)[1]));
                                        //g_growth = 1.0;
                                    }
                                    
                                    else if ((std::abs(elasticity_fe_face_values.quadrature_point(q)[1]) > 1.0) && (time > stop_growth))
                                    {
                                        abort();
                                        
                                    }
                                    
                                    else
                                    {
                                        g_growth = 1.0;
                                      /*
                                        pcout<<std::endl;
                                        pcout<<"-----------------------"<<std::endl;
                                        pcout<<"SOMEHOW WE ENDED IN THE OPTION g_growth = 1.0"<<std::endl;
                                        pcout<<"-----------------------"<<std::endl;
                                        pcout<<std::endl;
                                       */
                                        
                                        
                                    }
                                    

                                    
                                    
                                    const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);
                                    
                                    const Tensor<2,dim> Identity = Fluid_quantities
                                    ::get_Identity<dim> ();
                                    
                                    const Tensor <2,dim> F_s = Solid_quantities
                                    ::get_F_s(q,old_solution_face_grads,g_growth);
                                    
                                    const Tensor <2,dim> F_s_T = Solid_quantities
                                    ::get_F_s_T(F_s);
                                    
                                    const Tensor<2,dim> E = Solid_Terms
                                    ::get_E(F_s_T,F_s,Identity);
                                    
                                    const Tensor<2,dim> SIG = Solid_Terms
                                    ::get_SIG(lame_coefficient_mu,lame_coefficient_lambda,E,Identity);
                                    
                                    drag_lift_on_processFS -= (F_s * SIG * normal_vector) * elasticity_fe_face_values.JxW(q);
                                     
                                     
                                    
                                    
                                    
                                } // Closes the loop on face quadrature points
                                
                            }// Closes if the cell is in fluid domain
                        }// celll at b
                    }
                }// Closes the if cell is in solid domain

                
            }     //Closes the if on the MPI process involved
        }   // Closes the loop over cells
        
        
       drag_lift_vec_on_process[0] = drag_lift_on_process[0];   // This apparently unusefull step is necessary because the MPI::sum function
       drag_lift_vec_on_process[1] = drag_lift_on_process[1];  // accepts only std::vector and tmp_drag... is a Tensor<1,dim>!!!!!
        
        
       drag_lift_vec_on_processFS[0] = drag_lift_on_processFS[0];   // This apparently unusefull step is necessary because the MPI::sum function
       drag_lift_vec_on_processFS[1] = drag_lift_on_processFS[1];  // accepts only std::vector and tmp_drag... is a Tensor<1,dim>!!!!!


       
        Utilities::MPI::sum(drag_lift_vec_on_process, mpi_communicator, global_drag_lift_tensor);
        Utilities::MPI::sum(drag_lift_vec_on_processFS, mpi_communicator, global_drag_lift_tensorFS);
        global_min_Jf = Utilities::MPI::min(min_Jf_on_process ,mpi_communicator);
        global_outflow = Utilities::MPI::sum(outflow_on_process,mpi_communicator);
        global_vorticity = Utilities::MPI::sum(vorticity_on_process,mpi_communicator);

        
       
        // Multiplication with 0.5 because Stefan and Thomas compute
        // on the half channel
        drag_summed += 0.5 * std::abs(global_drag_lift_tensor[0]);
        drag = 0.5 * global_drag_lift_tensor[0];
        us_x = compute_point_value(growth_point, dim+dim+1);
        us_y = compute_point_value(growth_point, dim+dim+dim);
        width = 2.0 - 2.0 * us_y;

        
        pcout << "------------------" << std::endl;
        pcout << "DisX:        " << time << "   " << us_x << std::endl;
        pcout << "DisY:        " << time << "   " << us_y << std::endl;
        pcout << "Width:       " << time << "   " << width << std::endl;
        pcout << "Drag:        " << time << "   " << global_drag_lift_tensor[0] << std::endl;
        pcout << "Lift:        " << time << "   " << global_drag_lift_tensor[1] << std::endl;
        pcout << "Half_Drag:   " << time << "   " << drag << std::endl;
        pcout << "Hald_Drag_Sum" << time << "   " << drag_summed << std::endl;
        pcout << "Outflow:     " << time << "   " << global_outflow << std::endl;
        pcout << "Vorticity:   " << time << "   " << global_vorticity << std::endl;
        pcout << "LScgm:       " << time << "   " << alpha_growth << std::endl;
        pcout << "Min_J:       " << time << "   " << global_min_Jf << std::endl;
        pcout << "------------------" << std::endl;
        
        
        if(this_mpi_process == 0) data_file    << time <<" "
                                               << us_x  <<" "
                                               << us_y  <<" "
                                               << width <<" "
                                               << global_drag_lift_tensor[0] <<" "
                                               << global_drag_lift_tensor[1] <<" "
                                               << global_outflow <<" "
                                               << global_vorticity <<" "
                                               << alpha_growth <<" "
                                               << global_min_Jf <<std::endl;
        
    }
    
    
    template <int dim>
    void FsiProblem<dim>::run ()
    {
        
        std::ostringstream data_file_name;
        
        data_file_name <<refinement_level
                       <<"_"
                       <<MMPDE
                       <<".dat";
        
        
        if(this_mpi_process == 0) data_file.open(data_file_name.str().c_str());
        import_grid();
        set_active_fe_indices();
        set_runtime_parameters();
        setup_system();               // ----> In this function the Newton boundary conditions are set
      
        
        pcout << "\n=============================="
        << "====================================="  << std::endl;
        pcout << "Parameters\n"
        << "==========\n"
        << "Density fluid:     "   <<  density_fluid << "\n"
        << "Density structure: "   <<  density_structure << "\n"
        << "Viscosity fluid:   "   <<  viscosity << "\n"
        << "Lame coeff. mu:    "   <<  lame_coefficient_mu << "\n"
        << std::endl;

     
        
      
        const unsigned int output_skip = 2;
        
        
        drag_summed = 0.0;
        drag = growth_initial;
        u_y = 0.0;
        
        do
        {
            
            
            max_no_timesteps = 251; //251;
            
            if (timestep_number < 1 && p_switch == 0) timestep = 1.0;
            
            else timestep = 86400.0;
            
            
            if(p_switch == 0) p = 2.0;
            /*
             else if(p_switch == 1) p = 1.8;
             else if(p_switch == 2) p = 1.7;
             else if(p_switch == 3) p = 1.6;
             else if(p_switch == 4) p = 1.5;
             else if(p_switch == 5) p = 1.4;
             else if(p_switch == 6) p = 1.3;
             else if(p_switch == 7) p = 1.2;
             else if(p_switch == 8) p = 1.1;
        */
            
            
            else p = 1.3;

            
             if (p_switch != 0) alpha_growth = alpha_growth + gamma_zero * timestep * 1.0/(1.0 + drag/50.0);
            
            if (p_switch !=0)
            {
                pcout << "Timestep " << timestep_number
                << " (" << time_stepping_scheme
                << ")" <<    ": " << time
                << " (" << timestep << ")"
                << " (p = "<<p <<")"
                << "\n=============================="
                << "====================================="
                << std::endl;
                pcout << std::endl;
            }

            
            // Compute next time step
          
            Vector<double> tmp_old_timestep_solution(solution);
            old_timestep_solution = tmp_old_timestep_solution;
            
            // old_timestep_solution=(solution);
            
            newton_iteration (time);
           
            if (breaker > 0.9 && p_switch != 0)
            {
                timestep_number = max_no_timesteps;
                break;
            }

            if(p_switch !=0)
            {
                update_u_y();
                compute_functional_values(time);
                
                
                
                // Write solutions
                  if ((timestep_number % output_skip == 0))
                     output_results (timestep_number);
                
                time += timestep;
                ++timestep_number;
            }
            
            
            drag_summed = 0.0;
            final_drag_summed = 0.0;
            
            p_switch +=1;
            
            
        }

            
        
        
        while (timestep_number <= max_no_timesteps);
        
        
        
        
        data_file.close();
       
        pcout<<" The program has been executed correctly"<<std::endl;
        pcout<<" The max number of Newton iterations is "<<max_newton_iterations<<std::endl;
        
       }
        
  
 } // Closes namespace ALE_Problem



int main (int argc, char **argv)
{
    try
    {
        using namespace dealii;
        using namespace ALE_Problem;
        
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        
        FsiProblem<2> clogging_vessel(2,2,"linear_elastic",3);
       
       
        
        clogging_vessel.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    
    return 0;
}

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty::{self};

use std::collections::HashSet;
use std::rc::Rc;

pub fn identify_constrained_type_params<'tcx>(_tcx: &ty::ctxt<'tcx>,
                                              predicates: &[ty::Predicate<'tcx>],
                                              impl_trait_ref: Option<Rc<ty::TraitRef<'tcx>>>,
                                              input_parameters: &mut HashSet<ty::ParamTy>)
{
    loop {
        let num_inputs = input_parameters.len();

        let projection_predicates =
            predicates.iter()
                      .filter_map(|predicate| {
                          match *predicate {
                              // Ignore higher-ranked binders. For the purposes
                              // of this check, they don't matter because they
                              // only affect named regions, and we're just
                              // concerned about type parameters here.
                              ty::Predicate::Projection(ref data) => Some(data.0.clone()),
                              _ => None,
                          }
                      });

        for projection in projection_predicates {
            // Special case: watch out for some kind of sneaky attempt
            // to project out an associated type defined by this very trait.
            if Some(projection.projection_ty.trait_ref.clone()) == impl_trait_ref {
                continue;
            }

            let relies_only_on_inputs =
                projection.projection_ty.trait_ref.input_types()
                                                  .iter()
                                                  .flat_map(|t| t.walk())
                                                  .filter_map(|t| t.as_opt_param_ty())
                                                  .all(|t| input_parameters.contains(&t));

            if relies_only_on_inputs {
                input_parameters.extend(
                    projection.ty.walk().filter_map(|t| t.as_opt_param_ty()));
            }
        }

        if input_parameters.len() == num_inputs {
            break;
        }
    }
}

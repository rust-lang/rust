// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst;
use middle::ty::{self, Ty};

use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Parameter {
    Type(ty::ParamTy),
    Region(ty::EarlyBoundRegion),
}

pub fn parameters_for_type<'tcx>(ty: Ty<'tcx>) -> Vec<Parameter> {
    ty.walk()
      .flat_map(|ty| parameters_for_type_shallow(ty).into_iter())
      .collect()
}

pub fn parameters_for_trait_ref<'tcx>(trait_ref: &Rc<ty::TraitRef<'tcx>>) -> Vec<Parameter> {
    let mut region_parameters =
        parameters_for_regions_in_substs(&trait_ref.substs);

    let type_parameters =
        trait_ref.substs.types.iter()
                              .flat_map(|ty| parameters_for_type(ty).into_iter());

    region_parameters.extend(type_parameters);

    region_parameters
}

fn parameters_for_type_shallow<'tcx>(ty: Ty<'tcx>) -> Vec<Parameter> {
    match ty.sty {
        ty::ty_param(ref d) =>
            vec![Parameter::Type(d.clone())],
        ty::ty_rptr(region, _) =>
            parameters_for_region(region).into_iter().collect(),
        ty::ty_struct(_, substs) |
        ty::ty_enum(_, substs) =>
            parameters_for_regions_in_substs(substs),
        ty::ty_trait(ref data) =>
            parameters_for_regions_in_substs(&data.principal.skip_binder().substs),
        _ =>
            vec![],
    }
}

fn parameters_for_regions_in_substs(substs: &subst::Substs) -> Vec<Parameter> {
    substs.regions()
          .iter()
          .filter_map(|r| parameters_for_region(r))
          .collect()
}

fn parameters_for_region(region: &ty::Region) -> Option<Parameter> {
    match *region {
        ty::ReEarlyBound(data) => Some(Parameter::Region(data)),
        _ => None,
    }
}

pub fn identify_constrained_type_params<'tcx>(_tcx: &ty::ctxt<'tcx>,
                                              predicates: &[ty::Predicate<'tcx>],
                                              impl_trait_ref: Option<Rc<ty::TraitRef<'tcx>>>,
                                              input_parameters: &mut HashSet<Parameter>)
{
    loop {
        let num_inputs = input_parameters.len();

        let poly_projection_predicates = // : iterator over PolyProjectionPredicate
            predicates.iter()
                      .filter_map(|predicate| {
                          match *predicate {
                              ty::Predicate::Projection(ref data) => Some(data.clone()),
                              _ => None,
                          }
                      });

        for poly_projection in poly_projection_predicates {
            // Note that we can skip binder here because the impl
            // trait ref never contains any late-bound regions.
            let projection = poly_projection.skip_binder();

            // Special case: watch out for some kind of sneaky attempt
            // to project out an associated type defined by this very
            // trait.
            let unbound_trait_ref = &projection.projection_ty.trait_ref;
            if Some(unbound_trait_ref.clone()) == impl_trait_ref {
                continue;
            }

            let inputs = parameters_for_trait_ref(&projection.projection_ty.trait_ref);
            let relies_only_on_inputs = inputs.iter().all(|p| input_parameters.contains(&p));
            if relies_only_on_inputs {
                input_parameters.extend(parameters_for_type(projection.ty));
            }
        }

        if input_parameters.len() == num_inputs {
            break;
        }
    }
}

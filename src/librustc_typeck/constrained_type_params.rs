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

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Parameter {
    Type(ty::ParamTy),
    Region(ty::EarlyBoundRegion),
}

/// Returns the list of parameters that are constrained by the type `ty`
/// - i.e. the value of each parameter in the list is uniquely determined
/// by `ty` (see RFC 447).
pub fn parameters_for_type<'tcx>(ty: Ty<'tcx>) -> Vec<Parameter> {
    let mut result = vec![];
    ty.maybe_walk(|t| {
        if let ty::TyProjection(..) = t.sty {
            false // projections are not injective.
        } else {
            result.append(&mut parameters_for_type_shallow(t));
            // non-projection type constructors are injective.
            true
        }
    });
    result
}

pub fn parameters_for_trait_ref<'tcx>(trait_ref: &ty::TraitRef<'tcx>) -> Vec<Parameter> {
    let mut region_parameters =
        parameters_for_regions_in_substs(&trait_ref.substs);

    let type_parameters =
        trait_ref.substs.types.iter()
                              .flat_map(|ty| parameters_for_type(ty));

    region_parameters.extend(type_parameters);

    region_parameters
}

fn parameters_for_type_shallow<'tcx>(ty: Ty<'tcx>) -> Vec<Parameter> {
    match ty.sty {
        ty::TyParam(ref d) =>
            vec![Parameter::Type(d.clone())],
        ty::TyRef(region, _) =>
            parameters_for_region(region).into_iter().collect(),
        ty::TyStruct(_, substs) |
        ty::TyEnum(_, substs) =>
            parameters_for_regions_in_substs(substs),
        ty::TyTrait(ref data) =>
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
                                              impl_trait_ref: Option<ty::TraitRef<'tcx>>,
                                              input_parameters: &mut HashSet<Parameter>)
{
    let mut predicates = predicates.to_owned();
    setup_constraining_predicates(_tcx, &mut predicates, impl_trait_ref, input_parameters);
}


/// Order the predicates in `predicates` such that each parameter is
/// constrained before it is used, if that is possible, and add the
/// paramaters so constrained to `input_parameters`. For example,
/// imagine the following impl:
///
///     impl<T: Debug, U: Iterator<Item=T>> Trait for U
///
/// The impl's predicates are collected from left to right. Ignoring
/// the implicit `Sized` bounds, these are
///   * T: Debug
///   * U: Iterator
///   * <U as Iterator>::Item = T -- a desugared ProjectionPredicate
///
/// When we, for example, try to go over the trait-reference
/// `IntoIter<u32> as Trait`, we substitute the impl parameters with fresh
/// variables and match them with the impl trait-ref, so we know that
/// `$U = IntoIter<u32>`.
///
/// However, in order to process the `$T: Debug` predicate, we must first
/// know the value of `$T` - which is only given by processing the
/// projection. As we occasionally want to process predicates in a single
/// pass, we want the projection to come first. In fact, as projections
/// can (acyclically) depend on one another - see RFC447 for details - we
/// need to topologically sort them.
pub fn setup_constraining_predicates<'tcx>(_tcx: &ty::ctxt<'tcx>,
                                           predicates: &mut [ty::Predicate<'tcx>],
                                           impl_trait_ref: Option<ty::TraitRef<'tcx>>,
                                           input_parameters: &mut HashSet<Parameter>)
{
    // The canonical way of doing the needed topological sort
    // would be a DFS, but getting the graph and its ownership
    // right is annoying, so I am using an in-place fixed-point iteration,
    // which is `O(nt)` where `t` is the depth of type-parameter constraints,
    // remembering that `t` should be less than 7 in practice.
    //
    // Basically, I iterate over all projections and swap every
    // "ready" projection to the start of the list, such that
    // all of the projections before `i` are topologically sorted
    // and constrain all the parameters in `input_parameters`.
    //
    // In the example, `input_parameters` starts by containing `U` - which
    // is constrained by the trait-ref - and so on the first pass we
    // observe that `<U as Iterator>::Item = T` is a "ready" projection that
    // constrains `T` and swap it to front. As it is the sole projection,
    // no more swaps can take place afterwards, with the result being
    //   * <U as Iterator>::Item = T
    //   * T: Debug
    //   * U: Iterator
    let mut i = 0;
    let mut changed = true;
    while changed {
        changed = false;

        for j in i..predicates.len() {

            if let ty::Predicate::Projection(ref poly_projection) = predicates[j] {
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
                if !relies_only_on_inputs {
                    continue;
                }
                input_parameters.extend(parameters_for_type(projection.ty));
            } else {
                continue;
            }
            // fancy control flow to bypass borrow checker
            predicates.swap(i, j);
            i += 1;
            changed = true;
        }
    }
}

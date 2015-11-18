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

/// If `include_projections` is false, returns the list of parameters that are
/// constrained by the type `ty` - i.e. the value of each parameter in the list is
/// uniquely determined by `ty` (see RFC 447). If it is true, return the list
/// of parameters whose values are needed in order to constrain `ty` - these
/// differ, with the latter being a superset, in the presence of projections.
pub fn parameters_for_type<'tcx>(ty: Ty<'tcx>,
                                 include_projections: bool) -> Vec<Parameter> {
    let mut result = vec![];
    ty.maybe_walk(|t| match t.sty {
        ty::TyProjection(..) if !include_projections => {

            false // projections are not injective.
        }
        _ => {
            result.append(&mut parameters_for_type_shallow(t));
            // non-projection type constructors are injective.
            true
        }
    });
    result
}

pub fn parameters_for_trait_ref<'tcx>(trait_ref: &ty::TraitRef<'tcx>,
                                      include_projections: bool) -> Vec<Parameter> {
    let mut region_parameters =
        parameters_for_regions_in_substs(&trait_ref.substs);

    let type_parameters =
        trait_ref.substs
                 .types
                 .iter()
                 .flat_map(|ty| parameters_for_type(ty, include_projections));

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
        ty::TyProjection(ref pi) =>
            parameters_for_regions_in_substs(&pi.trait_ref.substs),
        ty::TyBool | ty::TyChar | ty::TyInt(..) | ty::TyUint(..) |
        ty::TyFloat(..) | ty::TyBox(..) | ty::TyStr |
        ty::TyArray(..) | ty::TySlice(..) | ty::TyBareFn(..) |
        ty::TyTuple(..) | ty::TyRawPtr(..) |
        ty::TyInfer(..) | ty::TyClosure(..) | ty::TyError =>
            vec![]
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
///
/// We *do* have to be somewhat careful when projection targets contain
/// projections themselves, for example in
///     impl<S,U,V,W> Trait for U where
/// /* 0 */   S: Iterator<Item=U>,
/// /* - */   U: Iterator,
/// /* 1 */   <U as Iterator>::Item: ToOwned<Owned=(W,<V as Iterator>::Item)>
/// /* 2 */   W: Iterator<Item=V>
/// /* 3 */   V: Debug
/// we have to evaluate the projections in the order I wrote them:
/// `V: Debug` requires `V` to be evaluated. The only projection that
/// *determines* `V` is 2 (1 contains it, but *does not determine it*,
/// as it is only contained within a projection), but that requires `W`
/// which is determined by 1, which requires `U`, that is determined
/// by 0. I should probably pick a less tangled example, but I can't
/// think of any.
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

                // A projection depends on its input types and determines its output
                // type. For example, if we have
                //     `<<T as Bar>::Baz as Iterator>::Output = <U as Iterator>::Output`
                // Then the projection only applies if `T` is known, but it still
                // does not determine `U`.

                let inputs = parameters_for_trait_ref(&projection.projection_ty.trait_ref, true);
                let relies_only_on_inputs = inputs.iter().all(|p| input_parameters.contains(&p));
                if !relies_only_on_inputs {
                    continue;
                }
                input_parameters.extend(parameters_for_type(projection.ty, false));
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

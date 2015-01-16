// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use check::regionck::{Rcx};

use middle::infer;
use middle::region;
use middle::subst;
use middle::ty::{self, Ty};
use util::ppaux::{Repr};

use syntax::codemap::Span;

pub fn check_safety_of_destructor_if_necessary<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                                                     typ: ty::Ty<'tcx>,
                                                     span: Span,
                                                     scope: region::CodeExtent) {
    debug!("check_safety_of_destructor_if_necessary typ: {} scope: {:?}",
           typ.repr(rcx.tcx()), scope);

    // types that have been traversed so far by `traverse_type_if_unseen`
    let mut breadcrumbs: Vec<Ty<'tcx>> = Vec::new();

    iterate_over_potentially_unsafe_regions_in_type(
        rcx,
        &mut breadcrumbs,
        typ,
        span,
        scope,
        false,
        0);
}

fn constrain_region_for_destructor_safety(rcx: &mut Rcx,
                                          region: ty::Region,
                                          inner_scope: region::CodeExtent,
                                          span: Span) {
    debug!("constrain_region_for_destructor_safety region: {:?} inner_scope: {:?}",
           region, inner_scope);

    // Ignore bound regions.
    match region {
        ty::ReEarlyBound(..) | ty::ReLateBound(..) => return,
        ty::ReFree(_) | ty::ReScope(_) | ty::ReStatic |
        ty::ReInfer(_) | ty::ReEmpty => {}
    }

    // Get the parent scope.
    let parent_inner_region =
        match rcx.tcx().region_maps.opt_encl_scope(inner_scope) {
            Some(parent_inner_scope) => ty::ReScope(parent_inner_scope),
            None =>
                rcx.tcx().sess.span_bug(
                    span, format!("no enclosing scope found for inner_scope: {:?}",
                                  inner_scope).as_slice()),
        };

    rcx.mk_subr(infer::SafeDestructor(span),
                parent_inner_region,
                region);
}

fn traverse_type_if_unseen<'a, 'tcx, P>(rcx: &mut Rcx<'a, 'tcx>,
                                        breadcrumbs: &mut Vec<Ty<'tcx>>,
                                        typ: ty::Ty<'tcx>,
                                        keep_going: P) -> bool where
    P: Fn(&mut Rcx<'a, 'tcx>, &mut Vec<Ty<'tcx>>) -> bool,
{
    // Avoid recursing forever.
    if !breadcrumbs.contains(&typ) {
        breadcrumbs.push(typ);
        let keep_going = keep_going(rcx, breadcrumbs);

        // You might be tempted to pop breadcrumbs here after the
        // `keep_going` call, but then you hit exponential time
        // blowup e.g. on compile-fail/huge-struct.rs. Instead, we
        // do not remove anything from the breadcrumbs vector
        // during any particular traversal, and instead clear it
        // after the whole traversal is done.

        keep_going
    } else {
        false
    }
}


fn iterate_over_potentially_unsafe_regions_in_type<'a, 'tcx>(
    rcx: &mut Rcx<'a, 'tcx>,
    breadcrumbs: &mut Vec<Ty<'tcx>>,
    typ: ty::Ty<'tcx>,
    span: Span,
    scope: region::CodeExtent,
    reachable_by_destructor: bool,
    depth: uint)
{
    ty::maybe_walk_ty(typ, |typ| {
        // Avoid recursing forever.
        traverse_type_if_unseen(rcx, breadcrumbs, typ, |rcx, breadcrumbs| {
            debug!("iterate_over_potentially_unsafe_regions_in_type \
                    {}typ: {} scope: {:?} reachable_by_destructor: {}",
                   (0..depth).map(|_| ' ').collect::<String>(),
                   typ.repr(rcx.tcx()), scope, reachable_by_destructor);

            let keep_going = match typ.sty {
                ty::ty_struct(structure_id, substitutions) => {
                    let reachable_by_destructor =
                        reachable_by_destructor ||
                        ty::has_dtor(rcx.tcx(), structure_id);

                    let fields =
                        ty::lookup_struct_fields(rcx.tcx(),
                                                 structure_id);
                    for field in fields.iter() {
                        let field_type =
                            ty::lookup_field_type(rcx.tcx(),
                                                  structure_id,
                                                  field.id,
                                                  substitutions);
                        iterate_over_potentially_unsafe_regions_in_type(
                            rcx,
                            breadcrumbs,
                            field_type,
                            span,
                            scope,
                            reachable_by_destructor, depth+1)
                    }

                    false
                }
                ty::ty_enum(enumeration_id, substitutions) => {
                    let reachable_by_destructor = reachable_by_destructor ||
                        ty::has_dtor(rcx.tcx(), enumeration_id);

                    let all_variant_info =
                        ty::substd_enum_variants(rcx.tcx(),
                                                 enumeration_id,
                                                 substitutions);
                    for variant_info in all_variant_info.iter() {
                        for argument_type in variant_info.args.iter() {
                            iterate_over_potentially_unsafe_regions_in_type(
                                rcx,
                                breadcrumbs,
                                *argument_type,
                                span,
                                scope,
                                reachable_by_destructor, depth+1)
                        }
                    }

                    false
                }
                ty::ty_rptr(region, _) => {
                    if reachable_by_destructor {
                        constrain_region_for_destructor_safety(rcx,
                                                               *region,
                                                               scope,
                                                               span)
                    }
                    // Don't recurse, since references do not own their
                    // contents.
                    false
                }
                ty::ty_unboxed_closure(..) => {
                    true
                }
                ty::ty_trait(ref trait_type) => {
                    if reachable_by_destructor {
                        match trait_type.principal.substs().regions {
                            subst::NonerasedRegions(ref regions) => {
                                for region in regions.iter() {
                                    constrain_region_for_destructor_safety(
                                        rcx,
                                        *region,
                                        scope,
                                        span)
                                }
                            }
                            subst::ErasedRegions => {}
                        }

                        // FIXME (pnkfelix): Added by pnkfelix, but
                        // need to double-check that this additional
                        // constraint is necessary.
                        constrain_region_for_destructor_safety(
                            rcx,
                            trait_type.bounds.region_bound,
                            scope,
                            span)
                    }
                    true
                }
                ty::ty_ptr(_) | ty::ty_bare_fn(..) => {
                    // Don't recurse, since pointers, boxes, and bare
                    // functions don't own instances of the types appearing
                    // within them.
                    false
                }
                ty::ty_bool | ty::ty_char | ty::ty_int(_) | ty::ty_uint(_) |
                ty::ty_float(_) | ty::ty_uniq(_) | ty::ty_str |
                ty::ty_vec(..) | ty::ty_tup(_) | ty::ty_param(_) |
                ty::ty_infer(_) | ty::ty_open(_) | ty::ty_err => true,

                ty::ty_projection(_) => {
                    // We keep going, since we want to descend into
                    // the substructure `Trait<..>` within the
                    // projection `<T as Trait<..>>::N`.
                    //
                    // Furthermore, in the future, we are likely to
                    // support higher-kinded projections (i.e. an
                    // associated item that is parameterized over a
                    // lifetime). When that is supported, we will need
                    // to ensure that we constrain the input regions
                    // accordingly (which might go here, or might end
                    // up in some recursive part of the traversal).
                    true
                }
            };

            keep_going
        })
    });
}

use middle::region;
use middle::subst;
use middle::ty;
use middle::typeck::check::regionck::{Rcx};
use middle::typeck::infer;
use util::ppaux::{Repr};

use syntax::codemap::Span;

pub fn check_safety_of_destructor_if_necessary<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                                                     typ: ty::Ty<'tcx>,
                                                     span: Span,
                                                     scope: region::CodeExtent) {
    debug!("check_safety_of_destructor_if_necessary typ: {} scope: {}",
           typ.repr(rcx.tcx()), scope);
    iterate_over_potentially_unsafe_regions_in_type(
        rcx,
        typ,
        span,
        scope,
        false,
        0)
}

fn constrain_region_for_destructor_safety(rcx: &mut Rcx,
                                          region: ty::Region,
                                          inner_scope: region::CodeExtent,
                                          span: Span) {
    debug!("constrain_region_for_destructor_safety region: {} inner_scope: {}",
           region, inner_scope);

    // Ignore bound regions.
    match region {
        ty::ReEarlyBound(..) | ty::ReLateBound(..) => return,
        ty::ReFunction | ty::ReFree(_) | ty::ReScope(_) | ty::ReStatic |
        ty::ReInfer(_) | ty::ReEmpty => {}
    }

    // Get the parent scope.
    let parent_inner_region =
        match rcx.tcx().region_maps.opt_encl_scope(inner_scope) {
            None | Some(region::CodeExtent::Closure(_)) => ty::ReFunction,
            Some(parent_inner_scope) => ty::ReScope(parent_inner_scope),
        };

    rcx.tcx().sess.span_note(
        span,
        format!("constrain_region_for_destructor_safety \
                 region: {} sub/inner_scope: {} sup/parent_inner_region: {}",
                region, inner_scope, parent_inner_region).as_slice());

    rcx.mk_subr(infer::SafeDestructor(span),
                parent_inner_region,
                region);
}

fn iterate_over_potentially_unsafe_regions_in_type<'a, 'tcx>(
        rcx: &mut Rcx<'a, 'tcx>,
        typ: ty::Ty<'tcx>,
        span: Span,
        scope: region::CodeExtent,
        reachable_by_destructor: bool,
        depth: uint) {
    ty::maybe_walk_ty(typ, |typ| {
        // Avoid recursing forever.
        rcx.traverse_type_if_unseen(typ, |rcx| {
            debug!("iterate_over_potentially_unsafe_regions_in_type \
                    {}typ: {} scope: {} reachable_by_destructor: {}",
                   String::from_char(depth, ' '),
                   typ.repr(rcx.tcx()), scope, reachable_by_destructor);

            let keep_going = match typ.sty {
                ty::ty_struct(structure_id, ref substitutions) => {
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
                            field_type,
                            span,
                            scope,
                            reachable_by_destructor, depth+1)
                    }

                    false
                }
                ty::ty_enum(enumeration_id, ref substitutions) => {
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
                                                               region,
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
                ty::ty_closure(ref closure_type) => {
                    match closure_type.store {
                        ty::RegionTraitStore(region, _) => {
                            if reachable_by_destructor {
                                constrain_region_for_destructor_safety(rcx,
                                                                       region,
                                                                       scope,
                                                                       span)
                            }
                        }
                        ty::UniqTraitStore => {}
                    }
                    // Don't recurse, since closures don't own the types
                    // appearing in their signature.
                    false
                }
                ty::ty_trait(ref trait_type) => {
                    if reachable_by_destructor {
                        match trait_type.principal.substs.regions {
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
                ty::ty_ptr(_) | ty::ty_bare_fn(_) => {
                    // Don't recurse, since pointers, boxes, and bare
                    // functions don't own instances of the types appearing
                    // within them.
                    false
                }
                ty::ty_bool | ty::ty_char | ty::ty_int(_) | ty::ty_uint(_) |
                ty::ty_float(_) | ty::ty_uniq(_) | ty::ty_str |
                ty::ty_vec(..) | ty::ty_tup(_) | ty::ty_param(_) |
                ty::ty_infer(_) | ty::ty_open(_) | ty::ty_err => true,
            };

            keep_going
        })
    });
}

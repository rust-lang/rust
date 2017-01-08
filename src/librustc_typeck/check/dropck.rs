// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use CrateCtxt;
use check::regionck::RegionCtxt;

use hir::def_id::DefId;
use middle::free_region::FreeRegionMap;
use rustc::infer::{self, InferOk};
use middle::region;
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, AdtKind, Ty, TyCtxt};
use rustc::traits::{self, ObligationCause, Reveal};
use util::nodemap::FxHashSet;

use syntax::ast;
use syntax_pos::Span;

/// check_drop_impl confirms that the Drop implementation identfied by
/// `drop_impl_did` is not any more specialized than the type it is
/// attached to (Issue #8142).
///
/// This means:
///
/// 1. The self type must be nominal (this is already checked during
///    coherence),
///
/// 2. The generic region/type parameters of the impl's self-type must
///    all be parameters of the Drop impl itself (i.e. no
///    specialization like `impl Drop for Foo<i32>`), and,
///
/// 3. Any bounds on the generic parameters must be reflected in the
///    struct/enum definition for the nominal type itself (i.e.
///    cannot do `struct S<T>; impl<T:Clone> Drop for S<T> { ... }`).
///
pub fn check_drop_impl(ccx: &CrateCtxt, drop_impl_did: DefId) -> Result<(), ()> {
    let dtor_self_type = ccx.tcx.item_type(drop_impl_did);
    let dtor_predicates = ccx.tcx.item_predicates(drop_impl_did);
    match dtor_self_type.sty {
        ty::TyAdt(adt_def, self_to_impl_substs) => {
            ensure_drop_params_and_item_params_correspond(ccx,
                                                          drop_impl_did,
                                                          dtor_self_type,
                                                          adt_def.did)?;

            ensure_drop_predicates_are_implied_by_item_defn(ccx,
                                                            drop_impl_did,
                                                            &dtor_predicates,
                                                            adt_def.did,
                                                            self_to_impl_substs)
        }
        _ => {
            // Destructors only work on nominal types.  This was
            // already checked by coherence, so we can panic here.
            let span = ccx.tcx.def_span(drop_impl_did);
            span_bug!(span,
                      "should have been rejected by coherence check: {}",
                      dtor_self_type);
        }
    }
}

fn ensure_drop_params_and_item_params_correspond<'a, 'tcx>(
    ccx: &CrateCtxt<'a, 'tcx>,
    drop_impl_did: DefId,
    drop_impl_ty: Ty<'tcx>,
    self_type_did: DefId)
    -> Result<(), ()>
{
    let tcx = ccx.tcx;
    let drop_impl_node_id = tcx.map.as_local_node_id(drop_impl_did).unwrap();
    let self_type_node_id = tcx.map.as_local_node_id(self_type_did).unwrap();

    // check that the impl type can be made to match the trait type.

    let impl_param_env = ty::ParameterEnvironment::for_item(tcx, self_type_node_id);
    tcx.infer_ctxt(impl_param_env, Reveal::NotSpecializable).enter(|infcx| {
        let tcx = infcx.tcx;
        let mut fulfillment_cx = traits::FulfillmentContext::new();

        let named_type = tcx.item_type(self_type_did);
        let named_type = named_type.subst(tcx, &infcx.parameter_environment.free_substs);

        let drop_impl_span = tcx.def_span(drop_impl_did);
        let fresh_impl_substs =
            infcx.fresh_substs_for_item(drop_impl_span, drop_impl_did);
        let fresh_impl_self_ty = drop_impl_ty.subst(tcx, fresh_impl_substs);

        let cause = &ObligationCause::misc(drop_impl_span, drop_impl_node_id);
        match infcx.eq_types(true, cause, named_type, fresh_impl_self_ty) {
            Ok(InferOk { obligations, .. }) => {
                // FIXME(#32730) propagate obligations
                assert!(obligations.is_empty());
            }
            Err(_) => {
                let item_span = tcx.map.span(self_type_node_id);
                struct_span_err!(tcx.sess, drop_impl_span, E0366,
                                 "Implementations of Drop cannot be specialized")
                    .span_note(item_span,
                               "Use same sequence of generic type and region \
                                parameters that is on the struct/enum definition")
                    .emit();
                return Err(());
            }
        }

        if let Err(ref errors) = fulfillment_cx.select_all_or_error(&infcx) {
            // this could be reached when we get lazy normalization
            infcx.report_fulfillment_errors(errors);
            return Err(());
        }

        let free_regions = FreeRegionMap::new();
        infcx.resolve_regions_and_report_errors(&free_regions, drop_impl_node_id);
        Ok(())
    })
}

/// Confirms that every predicate imposed by dtor_predicates is
/// implied by assuming the predicates attached to self_type_did.
fn ensure_drop_predicates_are_implied_by_item_defn<'a, 'tcx>(
    ccx: &CrateCtxt<'a, 'tcx>,
    drop_impl_did: DefId,
    dtor_predicates: &ty::GenericPredicates<'tcx>,
    self_type_did: DefId,
    self_to_impl_substs: &Substs<'tcx>)
    -> Result<(), ()>
{

    // Here is an example, analogous to that from
    // `compare_impl_method`.
    //
    // Consider a struct type:
    //
    //     struct Type<'c, 'b:'c, 'a> {
    //         x: &'a Contents            // (contents are irrelevant;
    //         y: &'c Cell<&'b Contents>, //  only the bounds matter for our purposes.)
    //     }
    //
    // and a Drop impl:
    //
    //     impl<'z, 'y:'z, 'x:'y> Drop for P<'z, 'y, 'x> {
    //         fn drop(&mut self) { self.y.set(self.x); } // (only legal if 'x: 'y)
    //     }
    //
    // We start out with self_to_impl_substs, that maps the generic
    // parameters of Type to that of the Drop impl.
    //
    //     self_to_impl_substs = {'c => 'z, 'b => 'y, 'a => 'x}
    //
    // Applying this to the predicates (i.e. assumptions) provided by the item
    // definition yields the instantiated assumptions:
    //
    //     ['y : 'z]
    //
    // We then check all of the predicates of the Drop impl:
    //
    //     ['y:'z, 'x:'y]
    //
    // and ensure each is in the list of instantiated
    // assumptions. Here, `'y:'z` is present, but `'x:'y` is
    // absent. So we report an error that the Drop impl injected a
    // predicate that is not present on the struct definition.

    let tcx = ccx.tcx;

    let self_type_node_id = tcx.map.as_local_node_id(self_type_did).unwrap();

    let drop_impl_span = tcx.def_span(drop_impl_did);

    // We can assume the predicates attached to struct/enum definition
    // hold.
    let generic_assumptions = tcx.item_predicates(self_type_did);

    let assumptions_in_impl_context = generic_assumptions.instantiate(tcx, &self_to_impl_substs);
    let assumptions_in_impl_context = assumptions_in_impl_context.predicates;

    // An earlier version of this code attempted to do this checking
    // via the traits::fulfill machinery. However, it ran into trouble
    // since the fulfill machinery merely turns outlives-predicates
    // 'a:'b and T:'b into region inference constraints. It is simpler
    // just to look for all the predicates directly.

    assert_eq!(dtor_predicates.parent, None);
    for predicate in &dtor_predicates.predicates {
        // (We do not need to worry about deep analysis of type
        // expressions etc because the Drop impls are already forced
        // to take on a structure that is roughly an alpha-renaming of
        // the generic parameters of the item definition.)

        // This path now just checks *all* predicates via the direct
        // lookup, rather than using fulfill machinery.
        //
        // However, it may be more efficient in the future to batch
        // the analysis together via the fulfill , rather than the
        // repeated `contains` calls.

        if !assumptions_in_impl_context.contains(&predicate) {
            let item_span = tcx.map.span(self_type_node_id);
            struct_span_err!(tcx.sess, drop_impl_span, E0367,
                             "The requirement `{}` is added only by the Drop impl.", predicate)
                .span_note(item_span,
                           "The same requirement must be part of \
                            the struct/enum definition")
                .emit();
        }
    }

    if tcx.sess.has_errors() {
        return Err(());
    }
    Ok(())
}

/// check_safety_of_destructor_if_necessary confirms that the type
/// expression `typ` conforms to the "Drop Check Rule" from the Sound
/// Generic Drop (RFC 769).
///
/// ----
///
/// The simplified (*) Drop Check Rule is the following:
///
/// Let `v` be some value (either temporary or named) and 'a be some
/// lifetime (scope). If the type of `v` owns data of type `D`, where
///
/// * (1.) `D` has a lifetime- or type-parametric Drop implementation,
///        (where that `Drop` implementation does not opt-out of
///         this check via the `unsafe_destructor_blind_to_params`
///         attribute), and
/// * (2.) the structure of `D` can reach a reference of type `&'a _`,
///
/// then 'a must strictly outlive the scope of v.
///
/// ----
///
/// This function is meant to by applied to the type for every
/// expression in the program.
///
/// ----
///
/// (*) The qualifier "simplified" is attached to the above
/// definition of the Drop Check Rule, because it is a simplification
/// of the original Drop Check rule, which attempted to prove that
/// some `Drop` implementations could not possibly access data even if
/// it was technically reachable, due to parametricity.
///
/// However, (1.) parametricity on its own turned out to be a
/// necessary but insufficient condition, and (2.)  future changes to
/// the language are expected to make it impossible to ensure that a
/// `Drop` implementation is actually parametric with respect to any
/// particular type parameter. (In particular, impl specialization is
/// expected to break the needed parametricity property beyond
/// repair.)
///
/// Therefore we have scaled back Drop-Check to a more conservative
/// rule that does not attempt to deduce whether a `Drop`
/// implementation could not possible access data of a given lifetime;
/// instead Drop-Check now simply assumes that if a destructor has
/// access (direct or indirect) to a lifetime parameter, then that
/// lifetime must be forced to outlive that destructor's dynamic
/// extent. We then provide the `unsafe_destructor_blind_to_params`
/// attribute as a way for destructor implementations to opt-out of
/// this conservative assumption (and thus assume the obligation of
/// ensuring that they do not access data nor invoke methods of
/// values that have been previously dropped).
///
pub fn check_safety_of_destructor_if_necessary<'a, 'gcx, 'tcx>(
    rcx: &mut RegionCtxt<'a, 'gcx, 'tcx>,
    typ: ty::Ty<'tcx>,
    span: Span,
    scope: region::CodeExtent)
{
    debug!("check_safety_of_destructor_if_necessary typ: {:?} scope: {:?}",
           typ, scope);

    let parent_scope = rcx.tcx.region_maps.opt_encl_scope(scope).unwrap_or_else(|| {
        span_bug!(span, "no enclosing scope found for scope: {:?}", scope)
    });

    let result = iterate_over_potentially_unsafe_regions_in_type(
        &mut DropckContext {
            rcx: rcx,
            span: span,
            parent_scope: parent_scope,
            breadcrumbs: FxHashSet()
        },
        TypeContext::Root,
        typ,
        0);
    match result {
        Ok(()) => {}
        Err(Error::Overflow(ref ctxt, ref detected_on_typ)) => {
            let tcx = rcx.tcx;
            let mut err = struct_span_err!(tcx.sess, span, E0320,
                                           "overflow while adding drop-check rules for {}", typ);
            match *ctxt {
                TypeContext::Root => {
                    // no need for an additional note if the overflow
                    // was somehow on the root.
                }
                TypeContext::ADT { def_id, variant, field } => {
                    let adt = tcx.lookup_adt_def(def_id);
                    let variant_name = match adt.adt_kind() {
                        AdtKind::Enum => format!("enum {} variant {}",
                                                 tcx.item_path_str(def_id),
                                                 variant),
                        AdtKind::Struct => format!("struct {}",
                                                   tcx.item_path_str(def_id)),
                        AdtKind::Union => format!("union {}",
                                                  tcx.item_path_str(def_id)),
                    };
                    span_note!(
                        &mut err,
                        span,
                        "overflowed on {} field {} type: {}",
                        variant_name,
                        field,
                        detected_on_typ);
                }
            }
            err.emit();
        }
    }
}

enum Error<'tcx> {
    Overflow(TypeContext, ty::Ty<'tcx>),
}

#[derive(Copy, Clone)]
enum TypeContext {
    Root,
    ADT {
        def_id: DefId,
        variant: ast::Name,
        field: ast::Name,
    }
}

struct DropckContext<'a, 'b: 'a, 'gcx: 'b+'tcx, 'tcx: 'b> {
    rcx: &'a mut RegionCtxt<'b, 'gcx, 'tcx>,
    /// types that have already been traversed
    breadcrumbs: FxHashSet<Ty<'tcx>>,
    /// span for error reporting
    span: Span,
    /// the scope reachable dtorck types must outlive
    parent_scope: region::CodeExtent
}

// `context` is used for reporting overflow errors
fn iterate_over_potentially_unsafe_regions_in_type<'a, 'b, 'gcx, 'tcx>(
    cx: &mut DropckContext<'a, 'b, 'gcx, 'tcx>,
    context: TypeContext,
    ty: Ty<'tcx>,
    depth: usize)
    -> Result<(), Error<'tcx>>
{
    let tcx = cx.rcx.tcx;
    // Issue #22443: Watch out for overflow. While we are careful to
    // handle regular types properly, non-regular ones cause problems.
    let recursion_limit = tcx.sess.recursion_limit.get();
    if depth / 4 >= recursion_limit {
        // This can get into rather deep recursion, especially in the
        // presence of things like Vec<T> -> Unique<T> -> PhantomData<T> -> T.
        // use a higher recursion limit to avoid errors.
        return Err(Error::Overflow(context, ty))
    }

    // canoncialize the regions in `ty` before inserting - infinitely many
    // region variables can refer to the same region.
    let ty = cx.rcx.resolve_type_and_region_vars_if_possible(&ty);

    if !cx.breadcrumbs.insert(ty) {
        debug!("iterate_over_potentially_unsafe_regions_in_type \
               {}ty: {} scope: {:?} - cached",
               (0..depth).map(|_| ' ').collect::<String>(),
               ty, cx.parent_scope);
        return Ok(()); // we already visited this type
    }
    debug!("iterate_over_potentially_unsafe_regions_in_type \
           {}ty: {} scope: {:?}",
           (0..depth).map(|_| ' ').collect::<String>(),
           ty, cx.parent_scope);

    // If `typ` has a destructor, then we must ensure that all
    // borrowed data reachable via `typ` must outlive the parent
    // of `scope`. This is handled below.
    //
    // However, there is an important special case: for any Drop
    // impl that is tagged as "blind" to their parameters,
    // we assume that data borrowed via such type parameters
    // remains unreachable via that Drop impl.
    //
    // For example, consider:
    //
    // ```rust
    // #[unsafe_destructor_blind_to_params]
    // impl<T> Drop for Vec<T> { ... }
    // ```
    //
    // which does have to be able to drop instances of `T`, but
    // otherwise cannot read data from `T`.
    //
    // Of course, for the type expression passed in for any such
    // unbounded type parameter `T`, we must resume the recursive
    // analysis on `T` (since it would be ignored by
    // type_must_outlive).
    let dropck_kind = has_dtor_of_interest(tcx, ty);
    debug!("iterate_over_potentially_unsafe_regions_in_type \
            ty: {:?} dropck_kind: {:?}", ty, dropck_kind);
    match dropck_kind {
        DropckKind::NoBorrowedDataAccessedInMyDtor => {
            // The maximally blind attribute.
        }
        DropckKind::BorrowedDataMustStrictlyOutliveSelf => {
            cx.rcx.type_must_outlive(infer::SubregionOrigin::SafeDestructor(cx.span),
                                     ty, tcx.mk_region(ty::ReScope(cx.parent_scope)));
            return Ok(());
        }
        DropckKind::RevisedSelf(revised_ty) => {
            cx.rcx.type_must_outlive(infer::SubregionOrigin::SafeDestructor(cx.span),
                                     revised_ty, tcx.mk_region(ty::ReScope(cx.parent_scope)));
            // Do not return early from this case; we want
            // to recursively process the internal structure of Self
            // (because even though the Drop for Self has been asserted
            //  safe, the types instantiated for the generics of Self
            //  may themselves carry dropck constraints.)
        }
    }

    debug!("iterate_over_potentially_unsafe_regions_in_type \
           {}ty: {} scope: {:?} - checking interior",
           (0..depth).map(|_| ' ').collect::<String>(),
           ty, cx.parent_scope);

    // We still need to ensure all referenced data is safe.
    match ty.sty {
        ty::TyBool | ty::TyChar | ty::TyInt(_) | ty::TyUint(_) |
        ty::TyFloat(_) | ty::TyStr | ty::TyNever => {
            // primitive - definitely safe
            Ok(())
        }

        ty::TyBox(ity) | ty::TyArray(ity, _) | ty::TySlice(ity) => {
            // single-element containers, behave like their element
            iterate_over_potentially_unsafe_regions_in_type(
                cx, context, ity, depth+1)
        }

        ty::TyAdt(def, substs) if def.is_phantom_data() => {
            // PhantomData<T> - behaves identically to T
            let ity = substs.type_at(0);
            iterate_over_potentially_unsafe_regions_in_type(
                cx, context, ity, depth+1)
        }

        ty::TyAdt(def, substs) => {
            let did = def.did;
            for variant in &def.variants {
                for field in variant.fields.iter() {
                    let fty = field.ty(tcx, substs);
                    let fty = cx.rcx.fcx.resolve_type_vars_with_obligations(
                        cx.rcx.fcx.normalize_associated_types_in(cx.span, &fty));
                    iterate_over_potentially_unsafe_regions_in_type(
                        cx,
                        TypeContext::ADT {
                            def_id: did,
                            field: field.name,
                            variant: variant.name,
                        },
                        fty,
                        depth+1)?
                }
            }
            Ok(())
        }

        ty::TyClosure(def_id, substs) => {
            for ty in substs.upvar_tys(def_id, tcx) {
                iterate_over_potentially_unsafe_regions_in_type(cx, context, ty, depth+1)?
            }
            Ok(())
        }

        ty::TyTuple(tys) => {
            for ty in tys {
                iterate_over_potentially_unsafe_regions_in_type(cx, context, ty, depth+1)?
            }
            Ok(())
        }

        ty::TyRawPtr(..) | ty::TyRef(..) | ty::TyParam(..) => {
            // these always come with a witness of liveness (references
            // explicitly, pointers implicitly, parameters by the
            // caller).
            Ok(())
        }

        ty::TyFnDef(..) | ty::TyFnPtr(_) => {
            // FIXME(#26656): this type is always destruction-safe, but
            // it implicitly witnesses Self: Fn, which can be false.
            Ok(())
        }

        ty::TyInfer(..) | ty::TyError => {
            tcx.sess.delay_span_bug(cx.span, "unresolved type in regionck");
            Ok(())
        }

        // these are always dtorck
        ty::TyDynamic(..) | ty::TyProjection(_) | ty::TyAnon(..) => bug!(),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum DropckKind<'tcx> {
    /// The "safe" kind; i.e. conservatively assume any borrow
    /// accessed by dtor, and therefore such data must strictly
    /// outlive self.
    ///
    /// Equivalent to RevisedTy with no change to the self type.
    BorrowedDataMustStrictlyOutliveSelf,

    /// The nearly completely-unsafe kind.
    ///
    /// Equivalent to RevisedSelf with *all* parameters remapped to ()
    /// (maybe...?)
    NoBorrowedDataAccessedInMyDtor,

    /// Assume all borrowed data access by dtor occurs as if Self has the
    /// type carried by this variant. In practice this means that some
    /// of the type parameters are remapped to `()` (and some lifetime
    /// parameters remapped to `'static`), because the developer has asserted
    /// that the destructor will not access their contents.
    RevisedSelf(Ty<'tcx>),
}

/// Returns the classification of what kind of check should be applied
/// to `ty`, which may include a revised type where some of the type
/// parameters are re-mapped to `()` to reflect the destructor's
/// "purity" with respect to their actual contents.
fn has_dtor_of_interest<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                        ty: Ty<'tcx>)
                                        -> DropckKind<'tcx> {
    match ty.sty {
        ty::TyAdt(adt_def, substs) => {
            if !adt_def.is_dtorck(tcx) {
                return DropckKind::NoBorrowedDataAccessedInMyDtor;
            }

            // Find the `impl<..> Drop for _` to inspect any
            // attributes attached to the impl's generics.
            let dtor_method = adt_def.destructor()
                .expect("dtorck type without destructor impossible");
            let method = tcx.associated_item(dtor_method);
            let impl_def_id = method.container.id();
            let revised_ty = revise_self_ty(tcx, adt_def, impl_def_id, substs);
            return DropckKind::RevisedSelf(revised_ty);
        }
        ty::TyDynamic(..) | ty::TyProjection(..) | ty::TyAnon(..) => {
            debug!("ty: {:?} isn't known, and therefore is a dropck type", ty);
            return DropckKind::BorrowedDataMustStrictlyOutliveSelf;
        },
        _ => {
            return DropckKind::NoBorrowedDataAccessedInMyDtor;
        }
    }
}

// Constructs new Ty just like the type defined by `adt_def` coupled
// with `substs`, except each type and lifetime parameter marked as
// `#[may_dangle]` in the Drop impl (identified by `impl_def_id`) is
// respectively mapped to `()` or `'static`.
//
// For example: If the `adt_def` maps to:
//
//   enum Foo<'a, X, Y> { ... }
//
// and the `impl_def_id` maps to:
//
//   impl<#[may_dangle] 'a, X, #[may_dangle] Y> Drop for Foo<'a, X, Y> { ... }
//
// then revises input: `Foo<'r,i64,&'r i64>` to: `Foo<'static,i64,()>`
fn revise_self_ty<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                  adt_def: &'tcx ty::AdtDef,
                                  impl_def_id: DefId,
                                  substs: &Substs<'tcx>)
                                  -> Ty<'tcx> {
    // Get generics for `impl Drop` to query for `#[may_dangle]` attr.
    let impl_bindings = tcx.item_generics(impl_def_id);

    // Get Substs attached to Self on `impl Drop`; process in parallel
    // with `substs`, replacing dangling entries as appropriate.
    let self_substs = {
        let impl_self_ty: Ty<'tcx> = tcx.item_type(impl_def_id);
        if let ty::TyAdt(self_adt_def, self_substs) = impl_self_ty.sty {
            assert_eq!(adt_def, self_adt_def);
            self_substs
        } else {
            bug!("Self in `impl Drop for _` must be an Adt.");
        }
    };

    // Walk `substs` + `self_substs`, build new substs appropriate for
    // `adt_def`; each non-dangling param reuses entry from `substs`.
    //
    // Note: The manner we map from a right-hand side (i.e. Region or
    // Ty) for a given `def` to generic parameter associated with that
    // right-hand side is tightly coupled to `Drop` impl constraints.
    //
    // E.g. we know such a Ty must be `TyParam`, because a destructor
    // for `struct Foo<X>` is defined via `impl<Y> Drop for Foo<Y>`,
    // and never by (for example) `impl<Z> Drop for Foo<Vec<Z>>`.
    let substs = Substs::for_item(
        tcx,
        adt_def.did,
        |def, _| {
            let r_orig = substs.region_for_def(def);
            let impl_self_orig = self_substs.region_for_def(def);
            let r = if let ty::Region::ReEarlyBound(ref ebr) = *impl_self_orig {
                if impl_bindings.region_param(ebr).pure_wrt_drop {
                    tcx.mk_region(ty::ReStatic)
                } else {
                    r_orig
                }
            } else {
                bug!("substs for an impl must map regions to ReEarlyBound");
            };
            debug!("has_dtor_of_interest mapping def {:?} orig {:?} to {:?}",
                   def, r_orig, r);
            r
        },
        |def, _| {
            let t_orig = substs.type_for_def(def);
            let impl_self_orig = self_substs.type_for_def(def);
            let t = if let ty::TypeVariants::TyParam(ref pt) = impl_self_orig.sty {
                if impl_bindings.type_param(pt).pure_wrt_drop {
                    tcx.mk_nil()
                } else {
                    t_orig
                }
            } else {
                bug!("substs for an impl must map types to TyParam");
            };
            debug!("has_dtor_of_interest mapping def {:?} orig {:?} {:?} to {:?} {:?}",
                   def, t_orig, t_orig.sty, t, t.sty);
            t
        });

    tcx.mk_adt(adt_def, &substs)
}

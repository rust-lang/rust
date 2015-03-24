// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use check::regionck::{self, Rcx};

use middle::infer;
use middle::region;
use middle::subst::{self, Subst};
use middle::ty::{self, Ty};
use util::ppaux::{Repr, UserString};

use syntax::ast;
use syntax::codemap::{self, Span};

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
pub fn check_drop_impl(tcx: &ty::ctxt, drop_impl_did: ast::DefId) -> Result<(), ()> {
    let ty::TypeScheme { generics: ref dtor_generics,
                         ty: ref dtor_self_type } = ty::lookup_item_type(tcx, drop_impl_did);
    let dtor_predicates = ty::lookup_predicates(tcx, drop_impl_did);
    match dtor_self_type.sty {
        ty::ty_enum(self_type_did, self_to_impl_substs) |
        ty::ty_struct(self_type_did, self_to_impl_substs) |
        ty::ty_closure(self_type_did, self_to_impl_substs) => {
            try!(ensure_drop_params_and_item_params_correspond(tcx,
                                                               drop_impl_did,
                                                               dtor_generics,
                                                               dtor_self_type,
                                                               self_type_did));

            ensure_drop_predicates_are_implied_by_item_defn(tcx,
                                                            drop_impl_did,
                                                            &dtor_predicates,
                                                            self_type_did,
                                                            self_to_impl_substs)
        }
        _ => {
            // Destructors only work on nominal types.  This was
            // already checked by coherence, so we can panic here.
            let span = tcx.map.def_id_span(drop_impl_did, codemap::DUMMY_SP);
            tcx.sess.span_bug(
                span, &format!("should have been rejected by coherence check: {}",
                               dtor_self_type.repr(tcx)));
        }
    }
}

fn ensure_drop_params_and_item_params_correspond<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    drop_impl_did: ast::DefId,
    drop_impl_generics: &ty::Generics<'tcx>,
    drop_impl_ty: &ty::Ty<'tcx>,
    self_type_did: ast::DefId) -> Result<(), ()>
{
    // New strategy based on review suggestion from nikomatsakis.
    //
    // (In the text and code below, "named" denotes "struct/enum", and
    // "generic params" denotes "type and region params")
    //
    // 1. Create fresh skolemized type/region "constants" for each of
    //    the named type's generic params.  Instantiate the named type
    //    with the fresh constants, yielding `named_skolem`.
    //
    // 2. Create unification variables for each of the Drop impl's
    //    generic params.  Instantiate the impl's Self's type with the
    //    unification-vars, yielding `drop_unifier`.
    //
    // 3. Attempt to unify Self_unif with Type_skolem.  If unification
    //    succeeds, continue (i.e. with the predicate checks).

    let ty::TypeScheme { generics: ref named_type_generics,
                         ty: named_type } =
        ty::lookup_item_type(tcx, self_type_did);

    let infcx = infer::new_infer_ctxt(tcx);
    infcx.try(|snapshot| {
        let (named_type_to_skolem, skol_map) =
            infcx.construct_skolemized_subst(named_type_generics, snapshot);
        let named_type_skolem = named_type.subst(tcx, &named_type_to_skolem);

        let drop_impl_span = tcx.map.def_id_span(drop_impl_did, codemap::DUMMY_SP);
        let drop_to_unifier =
            infcx.fresh_substs_for_generics(drop_impl_span, drop_impl_generics);
        let drop_unifier = drop_impl_ty.subst(tcx, &drop_to_unifier);

        if let Ok(()) = infer::mk_eqty(&infcx, true, infer::TypeOrigin::Misc(drop_impl_span),
                                       named_type_skolem, drop_unifier) {
            // Even if we did manage to equate the types, the process
            // may have just gathered unsolvable region constraints
            // like `R == 'static` (represented as a pair of subregion
            // constraints) for some skolemization constant R.
            //
            // However, the leak_check method allows us to confirm
            // that no skolemized regions escaped (i.e. were related
            // to other regions in the constraint graph).
            if let Ok(()) = infcx.leak_check(&skol_map, snapshot) {
                return Ok(())
            }
        }

        span_err!(tcx.sess, drop_impl_span, E0366,
                  "Implementations of Drop cannot be specialized");
        let item_span = tcx.map.span(self_type_did.node);
        tcx.sess.span_note(item_span,
                           "Use same sequence of generic type and region \
                            parameters that is on the struct/enum definition");
        return Err(());
    })
}

/// Confirms that every predicate imposed by dtor_predicates is
/// implied by assuming the predicates attached to self_type_did.
fn ensure_drop_predicates_are_implied_by_item_defn<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    drop_impl_did: ast::DefId,
    dtor_predicates: &ty::GenericPredicates<'tcx>,
    self_type_did: ast::DefId,
    self_to_impl_substs: &subst::Substs<'tcx>) -> Result<(), ()> {

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

    assert_eq!(self_type_did.krate, ast::LOCAL_CRATE);

    let drop_impl_span = tcx.map.def_id_span(drop_impl_did, codemap::DUMMY_SP);

    // We can assume the predicates attached to struct/enum definition
    // hold.
    let generic_assumptions = ty::lookup_predicates(tcx, self_type_did);

    let assumptions_in_impl_context = generic_assumptions.instantiate(tcx, &self_to_impl_substs);
    assert!(assumptions_in_impl_context.predicates.is_empty_in(subst::SelfSpace));
    assert!(assumptions_in_impl_context.predicates.is_empty_in(subst::FnSpace));
    let assumptions_in_impl_context =
        assumptions_in_impl_context.predicates.get_slice(subst::TypeSpace);

    // An earlier version of this code attempted to do this checking
    // via the traits::fulfill machinery. However, it ran into trouble
    // since the fulfill machinery merely turns outlives-predicates
    // 'a:'b and T:'b into region inference constraints. It is simpler
    // just to look for all the predicates directly.

    assert!(dtor_predicates.predicates.is_empty_in(subst::SelfSpace));
    assert!(dtor_predicates.predicates.is_empty_in(subst::FnSpace));
    let predicates = dtor_predicates.predicates.get_slice(subst::TypeSpace);
    for predicate in predicates {
        // (We do not need to worry about deep analysis of type
        // expressions etc because the Drop impls are already forced
        // to take on a structure that is roughly a alpha-renaming of
        // the generic parameters of the item definition.)

        // This path now just checks *all* predicates via the direct
        // lookup, rather than using fulfill machinery.
        //
        // However, it may be more efficient in the future to batch
        // the analysis together via the fulfill , rather than the
        // repeated `contains` calls.

        if !assumptions_in_impl_context.contains(&predicate) {
            let item_span = tcx.map.span(self_type_did.node);
            let req = predicate.user_string(tcx);
            span_err!(tcx.sess, drop_impl_span, E0367,
                      "The requirement `{}` is added only by the Drop impl.", req);
            tcx.sess.span_note(item_span,
                               "The same requirement must be part of \
                                the struct/enum definition");
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
/// The Drop Check Rule is the following:
///
/// Let `v` be some value (either temporary or named) and 'a be some
/// lifetime (scope). If the type of `v` owns data of type `D`, where
///
///   (1.) `D` has a lifetime- or type-parametric Drop implementation, and
///   (2.) the structure of `D` can reach a reference of type `&'a _`, and
///   (3.) either:
///
///     (A.) the Drop impl for `D` instantiates `D` at 'a directly,
///          i.e. `D<'a>`, or,
///
///     (B.) the Drop impl for `D` has some type parameter with a
///          trait bound `T` where `T` is a trait that has at least
///          one method,
///
/// then 'a must strictly outlive the scope of v.
///
/// ----
///
/// This function is meant to by applied to the type for every
/// expression in the program.
pub fn check_safety_of_destructor_if_necessary<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                                                     typ: ty::Ty<'tcx>,
                                                     span: Span,
                                                     scope: region::CodeExtent) {
    debug!("check_safety_of_destructor_if_necessary typ: {} scope: {:?}",
           typ.repr(rcx.tcx()), scope);

    // types that have been traversed so far by `traverse_type_if_unseen`
    let mut breadcrumbs: Vec<Ty<'tcx>> = Vec::new();

    let result = iterate_over_potentially_unsafe_regions_in_type(
        rcx,
        &mut breadcrumbs,
        TypeContext::Root,
        typ,
        span,
        scope,
        0,
        0);
    match result {
        Ok(()) => {}
        Err(Error::Overflow(ref ctxt, ref detected_on_typ)) => {
            let tcx = rcx.tcx();
            span_err!(tcx.sess, span, E0320,
                      "overflow while adding drop-check rules for {}",
                      typ.user_string(rcx.tcx()));
            match *ctxt {
                TypeContext::Root => {
                    // no need for an additional note if the overflow
                    // was somehow on the root.
                }
                TypeContext::EnumVariant { def_id, variant, arg_index } => {
                    // FIXME (pnkfelix): eventually lookup arg_name
                    // for the given index on struct variants.
                    span_note!(
                        rcx.tcx().sess,
                        span,
                        "overflowed on enum {} variant {} argument {} type: {}",
                        ty::item_path_str(tcx, def_id),
                        variant,
                        arg_index,
                        detected_on_typ.user_string(rcx.tcx()));
                }
                TypeContext::Struct { def_id, field } => {
                    span_note!(
                        rcx.tcx().sess,
                        span,
                        "overflowed on struct {} field {} type: {}",
                        ty::item_path_str(tcx, def_id),
                        field,
                        detected_on_typ.user_string(rcx.tcx()));
                }
            }
        }
    }
}

enum Error<'tcx> {
    Overflow(TypeContext, ty::Ty<'tcx>),
}

enum TypeContext {
    Root,
    EnumVariant {
        def_id: ast::DefId,
        variant: ast::Name,
        arg_index: usize,
    },
    Struct {
        def_id: ast::DefId,
        field: ast::Name,
    }
}

// The `depth` counts the number of calls to this function;
// the `xref_depth` counts the subset of such calls that go
// across a `Box<T>` or `PhantomData<T>`.
fn iterate_over_potentially_unsafe_regions_in_type<'a, 'tcx>(
    rcx: &mut Rcx<'a, 'tcx>,
    breadcrumbs: &mut Vec<Ty<'tcx>>,
    context: TypeContext,
    ty_root: ty::Ty<'tcx>,
    span: Span,
    scope: region::CodeExtent,
    depth: uint,
    xref_depth: uint) -> Result<(), Error<'tcx>>
{
    // Issue #22443: Watch out for overflow. While we are careful to
    // handle regular types properly, non-regular ones cause problems.
    let recursion_limit = rcx.tcx().sess.recursion_limit.get();
    if xref_depth >= recursion_limit {
        return Err(Error::Overflow(context, ty_root))
    }

    let origin = || infer::SubregionOrigin::SafeDestructor(span);
    let mut walker = ty_root.walk();
    let opt_phantom_data_def_id = rcx.tcx().lang_items.phantom_data();

    let destructor_for_type = rcx.tcx().destructor_for_type.borrow();

    let xref_depth_orig = xref_depth;

    while let Some(typ) = walker.next() {
        // Avoid recursing forever.
        if breadcrumbs.contains(&typ) {
            continue;
        }
        breadcrumbs.push(typ);

        // If we encounter `PhantomData<T>`, then we should replace it
        // with `T`, the type it represents as owned by the
        // surrounding context, before doing further analysis.
        let (typ, xref_depth) = match typ.sty {
            ty::ty_struct(struct_did, substs) => {
                if opt_phantom_data_def_id == Some(struct_did) {
                    let item_type = ty::lookup_item_type(rcx.tcx(), struct_did);
                    let tp_def = item_type.generics.types
                        .opt_get(subst::TypeSpace, 0).unwrap();
                    let new_typ = substs.type_for_def(tp_def);
                    debug!("replacing phantom {} with {}",
                           typ.repr(rcx.tcx()), new_typ.repr(rcx.tcx()));
                    (new_typ, xref_depth_orig + 1)
                } else {
                    (typ, xref_depth_orig)
                }
            }

            // Note: When ty_uniq is removed from compiler, the
            // definition of `Box<T>` must carry a PhantomData that
            // puts us into the previous case.
            ty::ty_uniq(new_typ) => {
                debug!("replacing ty_uniq {} with {}",
                       typ.repr(rcx.tcx()), new_typ.repr(rcx.tcx()));
                (new_typ, xref_depth_orig + 1)
            }

            _ => {
                (typ, xref_depth_orig)
            }
        };

        let opt_type_did = match typ.sty {
            ty::ty_struct(struct_did, _) => Some(struct_did),
            ty::ty_enum(enum_did, _) => Some(enum_did),
            _ => None,
        };

        let opt_dtor =
            opt_type_did.and_then(|did| destructor_for_type.get(&did));

        debug!("iterate_over_potentially_unsafe_regions_in_type \
                {}typ: {} scope: {:?} opt_dtor: {:?} xref: {}",
               (0..depth).map(|_| ' ').collect::<String>(),
               typ.repr(rcx.tcx()), scope, opt_dtor, xref_depth);

        // If `typ` has a destructor, then we must ensure that all
        // borrowed data reachable via `typ` must outlive the parent
        // of `scope`. This is handled below.
        //
        // However, there is an important special case: by
        // parametricity, any generic type parameters have *no* trait
        // bounds in the Drop impl can not be used in any way (apart
        // from being dropped), and thus we can treat data borrowed
        // via such type parameters remains unreachable.
        //
        // For example, consider `impl<T> Drop for Vec<T> { ... }`,
        // which does have to be able to drop instances of `T`, but
        // otherwise cannot read data from `T`.
        //
        // Of course, for the type expression passed in for any such
        // unbounded type parameter `T`, we must resume the recursive
        // analysis on `T` (since it would be ignored by
        // type_must_outlive).
        //
        // FIXME (pnkfelix): Long term, we could be smart and actually
        // feed which generic parameters can be ignored *into* `fn
        // type_must_outlive` (or some generalization thereof). But
        // for the short term, it probably covers most cases of
        // interest to just special case Drop impls where: (1.) there
        // are no generic lifetime parameters and (2.)  *all* generic
        // type parameters are unbounded.  If both conditions hold, we
        // simply skip the `type_must_outlive` call entirely (but
        // resume the recursive checking of the type-substructure).

        let has_dtor_of_interest;

        if let Some(&dtor_method_did) = opt_dtor {
            let impl_did = ty::impl_of_method(rcx.tcx(), dtor_method_did)
                .unwrap_or_else(|| {
                    rcx.tcx().sess.span_bug(
                        span, "no Drop impl found for drop method")
                });

            let dtor_typescheme = ty::lookup_item_type(rcx.tcx(), impl_did);
            let dtor_generics = dtor_typescheme.generics;
            let dtor_predicates = ty::lookup_predicates(rcx.tcx(), impl_did);

            let has_pred_of_interest = dtor_predicates.predicates.iter().any(|pred| {
                // In `impl<T> Drop where ...`, we automatically
                // assume some predicate will be meaningful and thus
                // represents a type through which we could reach
                // borrowed data. However, there can be implicit
                // predicates (namely for Sized), and so we still need
                // to walk through and filter out those cases.

                let result = match *pred {
                    ty::Predicate::Trait(ty::Binder(ref t_pred)) => {
                        let def_id = t_pred.trait_ref.def_id;
                        match rcx.tcx().lang_items.to_builtin_kind(def_id) {
                            Some(ty::BoundSend) |
                            Some(ty::BoundSized) |
                            Some(ty::BoundCopy) |
                            Some(ty::BoundSync) => false,
                            _ => true,
                        }
                    }
                    ty::Predicate::Equate(..) |
                    ty::Predicate::RegionOutlives(..) |
                    ty::Predicate::TypeOutlives(..) |
                    ty::Predicate::Projection(..) => {
                        // we assume all of these where-clauses may
                        // give the drop implementation the capabilty
                        // to access borrowed data.
                        true
                    }
                };

                if result {
                    debug!("typ: {} has interesting dtor due to generic preds, e.g. {}",
                           typ.repr(rcx.tcx()), pred.repr(rcx.tcx()));
                }

                result
            });

            // In `impl<'a> Drop ...`, we automatically assume
            // `'a` is meaningful and thus represents a bound
            // through which we could reach borrowed data.
            //
            // FIXME (pnkfelix): In the future it would be good to
            // extend the language to allow the user to express,
            // in the impl signature, that a lifetime is not
            // actually used (something like `where 'a: ?Live`).
            let has_region_param_of_interest =
                dtor_generics.has_region_params(subst::TypeSpace);

            has_dtor_of_interest =
                has_region_param_of_interest ||
                has_pred_of_interest;

            if has_dtor_of_interest {
                debug!("typ: {} has interesting dtor, due to \
                        region params: {} or pred: {}",
                       typ.repr(rcx.tcx()),
                       has_region_param_of_interest,
                       has_pred_of_interest);
            } else {
                debug!("typ: {} has dtor, but it is uninteresting",
                       typ.repr(rcx.tcx()));
            }

        } else {
            debug!("typ: {} has no dtor, and thus is uninteresting",
                   typ.repr(rcx.tcx()));
            has_dtor_of_interest = false;
        }

        if has_dtor_of_interest {
            // If `typ` has a destructor, then we must ensure that all
            // borrowed data reachable via `typ` must outlive the
            // parent of `scope`. (It does not suffice for it to
            // outlive `scope` because that could imply that the
            // borrowed data is torn down in between the end of
            // `scope` and when the destructor itself actually runs.)

            let parent_region =
                match rcx.tcx().region_maps.opt_encl_scope(scope) {
                    Some(parent_scope) => ty::ReScope(parent_scope),
                    None => rcx.tcx().sess.span_bug(
                        span, &format!("no enclosing scope found for scope: {:?}",
                                       scope)),
                };

            regionck::type_must_outlive(rcx, origin(), typ, parent_region);

        } else {
            // Okay, `typ` itself is itself not reachable by a
            // destructor; but it may contain substructure that has a
            // destructor.

            match typ.sty {
                ty::ty_struct(struct_did, substs) => {
                    debug!("typ: {} is struct; traverse structure and not type-expression",
                           typ.repr(rcx.tcx()));
                    // Don't recurse; we extract type's substructure,
                    // so do not process subparts of type expression.
                    walker.skip_current_subtree();

                    let fields =
                        ty::lookup_struct_fields(rcx.tcx(), struct_did);
                    for field in fields.iter() {
                        let field_type =
                            ty::lookup_field_type(rcx.tcx(),
                                                  struct_did,
                                                  field.id,
                                                  substs);
                        try!(iterate_over_potentially_unsafe_regions_in_type(
                            rcx,
                            breadcrumbs,
                            TypeContext::Struct {
                                def_id: struct_did,
                                field: field.name,
                            },
                            field_type,
                            span,
                            scope,
                            depth+1,
                            xref_depth))
                    }
                }

                ty::ty_enum(enum_did, substs) => {
                    debug!("typ: {} is enum; traverse structure and not type-expression",
                           typ.repr(rcx.tcx()));
                    // Don't recurse; we extract type's substructure,
                    // so do not process subparts of type expression.
                    walker.skip_current_subtree();

                    let all_variant_info =
                        ty::substd_enum_variants(rcx.tcx(),
                                                 enum_did,
                                                 substs);
                    for variant_info in all_variant_info.iter() {
                        for (i, arg_type) in variant_info.args.iter().enumerate() {
                            try!(iterate_over_potentially_unsafe_regions_in_type(
                                rcx,
                                breadcrumbs,
                                TypeContext::EnumVariant {
                                    def_id: enum_did,
                                    variant: variant_info.name,
                                    arg_index: i,
                                },
                                *arg_type,
                                span,
                                scope,
                                depth+1,
                                xref_depth));
                        }
                    }
                }

                ty::ty_rptr(..) | ty::ty_ptr(_) | ty::ty_bare_fn(..) => {
                    // Don't recurse, since references, pointers,
                    // boxes, and bare functions don't own instances
                    // of the types appearing within them.
                    walker.skip_current_subtree();
                }
                _ => {}
            };

            // You might be tempted to pop breadcrumbs here after
            // processing type's internals above, but then you hit
            // exponential time blowup e.g. on
            // compile-fail/huge-struct.rs. Instead, we do not remove
            // anything from the breadcrumbs vector during any particular
            // traversal, and instead clear it after the whole traversal
            // is done.
        }
    }

    return Ok(());
}

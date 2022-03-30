//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_index::bit_set::GrowableBitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::subst::{GenericArg, InternalSubsts};
use rustc_middle::ty::{self, ImplPolarity, Ty, TyCtxt, TypeFoldable, TypeVisitor};
use rustc_session::lint;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::Span;
use rustc_trait_selection::traits;
use std::ops::ControlFlow;

pub(super) fn orphan_check_crate(tcx: TyCtxt<'_>, (): ()) -> &[LocalDefId] {
    let mut errors = Vec::new();
    for (&trait_def_id, impls_of_trait) in tcx.all_local_trait_impls(()) {
        for &impl_of_trait in impls_of_trait {
            match orphan_check_impl(tcx, impl_of_trait) {
                Ok(()) => {}
                Err(ErrorReported) => errors.push(impl_of_trait),
            }
        }

        if tcx.trait_is_auto(trait_def_id) {
            lint_auto_trait_impls(tcx, trait_def_id, impls_of_trait);
        }
    }
    tcx.arena.alloc_slice(&errors)
}

#[instrument(skip(tcx), level = "debug")]
fn orphan_check_impl(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Result<(), ErrorReported> {
    let trait_ref = tcx.impl_trait_ref(def_id).unwrap();
    let trait_def_id = trait_ref.def_id;

    let item = tcx.hir().item(hir::ItemId { def_id });
    let impl_ = match item.kind {
        hir::ItemKind::Impl(ref impl_) => impl_,
        _ => bug!("{:?} is not an impl: {:?}", def_id, item),
    };
    let sp = tcx.sess.source_map().guess_head_span(item.span);
    let tr = impl_.of_trait.as_ref().unwrap();
    match traits::orphan_check(tcx, item.def_id.to_def_id()) {
        Ok(()) => {}
        Err(err) => emit_orphan_check_error(
            tcx,
            sp,
            tr.path.span,
            impl_.self_ty.span,
            &impl_.generics,
            err,
        )?,
    }

    // In addition to the above rules, we restrict impls of auto traits
    // so that they can only be implemented on nominal types, such as structs,
    // enums or foreign types. To see why this restriction exists, consider the
    // following example (#22978). Imagine that crate A defines an auto trait
    // `Foo` and a fn that operates on pairs of types:
    //
    // ```
    // // Crate A
    // auto trait Foo { }
    // fn two_foos<A:Foo,B:Foo>(..) {
    //     one_foo::<(A,B)>(..)
    // }
    // fn one_foo<T:Foo>(..) { .. }
    // ```
    //
    // This type-checks fine; in particular the fn
    // `two_foos` is able to conclude that `(A,B):Foo`
    // because `A:Foo` and `B:Foo`.
    //
    // Now imagine that crate B comes along and does the following:
    //
    // ```
    // struct A { }
    // struct B { }
    // impl Foo for A { }
    // impl Foo for B { }
    // impl !Send for (A, B) { }
    // ```
    //
    // This final impl is legal according to the orphan
    // rules, but it invalidates the reasoning from
    // `two_foos` above.
    debug!(
        "trait_ref={:?} trait_def_id={:?} trait_is_auto={}",
        trait_ref,
        trait_def_id,
        tcx.trait_is_auto(trait_def_id)
    );

    if tcx.trait_is_auto(trait_def_id) && !trait_def_id.is_local() {
        let self_ty = trait_ref.self_ty();
        let opt_self_def_id = match *self_ty.kind() {
            ty::Adt(self_def, _) => Some(self_def.did),
            ty::Foreign(did) => Some(did),
            _ => None,
        };

        let msg = match opt_self_def_id {
            // We only want to permit nominal types, but not *all* nominal types.
            // They must be local to the current crate, so that people
            // can't do `unsafe impl Send for Rc<SomethingLocal>` or
            // `impl !Send for Box<SomethingLocalAndSend>`.
            Some(self_def_id) => {
                if self_def_id.is_local() {
                    None
                } else {
                    Some((
                        format!(
                            "cross-crate traits with a default impl, like `{}`, \
                                    can only be implemented for a struct/enum type \
                                    defined in the current crate",
                            tcx.def_path_str(trait_def_id)
                        ),
                        "can't implement cross-crate trait for type in another crate",
                    ))
                }
            }
            _ => Some((
                format!(
                    "cross-crate traits with a default impl, like `{}`, can \
                                only be implemented for a struct/enum type, not `{}`",
                    tcx.def_path_str(trait_def_id),
                    self_ty
                ),
                "can't implement cross-crate trait with a default impl for \
                        non-struct/enum type",
            )),
        };

        if let Some((msg, label)) = msg {
            struct_span_err!(tcx.sess, sp, E0321, "{}", msg).span_label(sp, label).emit();
            return Err(ErrorReported);
        }
    }

    if let ty::Opaque(def_id, _) = *trait_ref.self_ty().kind() {
        tcx.sess
            .struct_span_err(sp, "cannot implement trait on type alias impl trait")
            .span_note(tcx.def_span(def_id), "type alias impl trait defined here")
            .emit();
        return Err(ErrorReported);
    }

    Ok(())
}

fn emit_orphan_check_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    trait_span: Span,
    self_ty_span: Span,
    generics: &hir::Generics<'tcx>,
    err: traits::OrphanCheckErr<'tcx>,
) -> Result<!, ErrorReported> {
    match err {
        traits::OrphanCheckErr::NonLocalInputType(tys) => {
            let mut err = struct_span_err!(
                tcx.sess,
                sp,
                E0117,
                "only traits defined in the current crate can be implemented for \
                        arbitrary types"
            );
            err.span_label(sp, "impl doesn't use only types from inside the current crate");
            for (ty, is_target_ty) in &tys {
                let mut ty = *ty;
                tcx.infer_ctxt().enter(|infcx| {
                    // Remove the lifetimes unnecessary for this error.
                    ty = infcx.freshen(ty);
                });
                ty = match ty.kind() {
                    // Remove the type arguments from the output, as they are not relevant.
                    // You can think of this as the reverse of `resolve_vars_if_possible`.
                    // That way if we had `Vec<MyType>`, we will properly attribute the
                    // problem to `Vec<T>` and avoid confusing the user if they were to see
                    // `MyType` in the error.
                    ty::Adt(def, _) => tcx.mk_adt(def, ty::List::empty()),
                    _ => ty,
                };
                let this = "this".to_string();
                let (ty, postfix) = match &ty.kind() {
                    ty::Slice(_) => (this, " because slices are always foreign"),
                    ty::Array(..) => (this, " because arrays are always foreign"),
                    ty::Tuple(..) => (this, " because tuples are always foreign"),
                    _ => (format!("`{}`", ty), ""),
                };
                let msg = format!("{} is not defined in the current crate{}", ty, postfix);
                if *is_target_ty {
                    // Point at `D<A>` in `impl<A, B> for C<B> in D<A>`
                    err.span_label(self_ty_span, &msg);
                } else {
                    // Point at `C<B>` in `impl<A, B> for C<B> in D<A>`
                    err.span_label(trait_span, &msg);
                }
            }
            err.note("define and implement a trait or new type instead");
            err.emit()
        }
        traits::OrphanCheckErr::UncoveredTy(param_ty, local_type) => {
            let mut sp = sp;
            for param in generics.params {
                if param.name.ident().to_string() == param_ty.to_string() {
                    sp = param.span;
                }
            }

            match local_type {
                Some(local_type) => struct_span_err!(
                    tcx.sess,
                    sp,
                    E0210,
                    "type parameter `{}` must be covered by another type \
                    when it appears before the first local type (`{}`)",
                    param_ty,
                    local_type
                )
                .span_label(
                    sp,
                    format!(
                        "type parameter `{}` must be covered by another type \
                    when it appears before the first local type (`{}`)",
                        param_ty, local_type
                    ),
                )
                .note(
                    "implementing a foreign trait is only possible if at \
                        least one of the types for which it is implemented is local, \
                        and no uncovered type parameters appear before that first \
                        local type",
                )
                .note(
                    "in this case, 'before' refers to the following order: \
                        `impl<..> ForeignTrait<T1, ..., Tn> for T0`, \
                        where `T0` is the first and `Tn` is the last",
                )
                .emit(),
                None => struct_span_err!(
                    tcx.sess,
                    sp,
                    E0210,
                    "type parameter `{}` must be used as the type parameter for some \
                    local type (e.g., `MyStruct<{}>`)",
                    param_ty,
                    param_ty
                )
                .span_label(
                    sp,
                    format!(
                        "type parameter `{}` must be used as the type parameter for some \
                    local type",
                        param_ty,
                    ),
                )
                .note(
                    "implementing a foreign trait is only possible if at \
                        least one of the types for which it is implemented is local",
                )
                .note(
                    "only traits defined in the current crate can be \
                        implemented for a type parameter",
                )
                .emit(),
            }
        }
    }

    Err(ErrorReported)
}

#[derive(Default)]
struct AreUniqueParamsVisitor {
    seen: GrowableBitSet<u32>,
}

#[derive(Copy, Clone)]
enum NotUniqueParam<'tcx> {
    DuplicateParam(GenericArg<'tcx>),
    NotParam(GenericArg<'tcx>),
}

impl<'tcx> TypeVisitor<'tcx> for AreUniqueParamsVisitor {
    type BreakTy = NotUniqueParam<'tcx>;
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match t.kind() {
            ty::Param(p) => {
                if self.seen.insert(p.index) {
                    ControlFlow::CONTINUE
                } else {
                    ControlFlow::Break(NotUniqueParam::DuplicateParam(t.into()))
                }
            }
            _ => ControlFlow::Break(NotUniqueParam::NotParam(t.into())),
        }
    }
    fn visit_region(&mut self, _: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        // We don't drop candidates during candidate assembly because of region
        // constraints, so the behavior for impls only constrained by regions
        // will not change.
        ControlFlow::CONTINUE
    }
    fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        match c.val() {
            ty::ConstKind::Param(p) => {
                if self.seen.insert(p.index) {
                    ControlFlow::CONTINUE
                } else {
                    ControlFlow::Break(NotUniqueParam::DuplicateParam(c.into()))
                }
            }
            _ => ControlFlow::Break(NotUniqueParam::NotParam(c.into())),
        }
    }
}

/// Lint impls of auto traits if they are likely to have
/// unsound or surprising effects on auto impls.
fn lint_auto_trait_impls(tcx: TyCtxt<'_>, trait_def_id: DefId, impls: &[LocalDefId]) {
    let mut non_covering_impls = Vec::new();
    for &impl_def_id in impls {
        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        if trait_ref.references_error() {
            return;
        }

        if tcx.impl_polarity(impl_def_id) != ImplPolarity::Positive {
            return;
        }

        assert_eq!(trait_ref.substs.len(), 1);
        let self_ty = trait_ref.self_ty();
        let (self_type_did, substs) = match self_ty.kind() {
            ty::Adt(def, substs) => (def.did, substs),
            _ => {
                // FIXME: should also lint for stuff like `&i32` but
                // considering that auto traits are unstable, that
                // isn't too important for now as this only affects
                // crates using `nightly`, and std.
                continue;
            }
        };

        // Impls which completely cover a given root type are fine as they
        // disable auto impls entirely. So only lint if the substs
        // are not a permutation of the identity substs.
        match substs.visit_with(&mut AreUniqueParamsVisitor::default()) {
            ControlFlow::Continue(()) => {} // ok
            ControlFlow::Break(arg) => {
                // Ideally:
                //
                // - compute the requirements for the auto impl candidate
                // - check whether these are implied by the non covering impls
                // - if not, emit the lint
                //
                // What we do here is a bit simpler:
                //
                // - badly check if an auto impl candidate definitely does not apply
                //   for the given simplified type
                // - if so, do not lint
                if fast_reject_auto_impl(tcx, trait_def_id, self_ty) {
                    // ok
                } else {
                    non_covering_impls.push((impl_def_id, self_type_did, arg));
                }
            }
        }
    }

    for &(impl_def_id, self_type_did, arg) in &non_covering_impls {
        tcx.struct_span_lint_hir(
            lint::builtin::SUSPICIOUS_AUTO_TRAIT_IMPLS,
            tcx.hir().local_def_id_to_hir_id(impl_def_id),
            tcx.def_span(impl_def_id),
            |err| {
                let mut err = err.build(&format!(
                    "cross-crate traits with a default impl, like `{}`, \
                         should not be specialized",
                    tcx.def_path_str(trait_def_id),
                ));
                let item_span = tcx.def_span(self_type_did);
                let self_descr = tcx.def_kind(self_type_did).descr(self_type_did);
                err.span_note(
                    item_span,
                    &format!(
                        "try using the same sequence of generic parameters as the {} definition",
                        self_descr,
                    ),
                );
                match arg {
                    NotUniqueParam::DuplicateParam(arg) => {
                        err.note(&format!("`{}` is mentioned multiple times", arg));
                    }
                    NotUniqueParam::NotParam(arg) => {
                        err.note(&format!("`{}` is not a generic parameter", arg));
                    }
                }
                err.emit();
            },
        );
    }
}

fn fast_reject_auto_impl<'tcx>(tcx: TyCtxt<'tcx>, trait_def_id: DefId, self_ty: Ty<'tcx>) -> bool {
    struct DisableAutoTraitVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        self_ty_root: Ty<'tcx>,
        seen: FxHashSet<DefId>,
    }

    impl<'tcx> TypeVisitor<'tcx> for DisableAutoTraitVisitor<'tcx> {
        type BreakTy = ();
        fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            let tcx = self.tcx;
            if t != self.self_ty_root {
                for impl_def_id in tcx.non_blanket_impls_for_ty(self.trait_def_id, t) {
                    match tcx.impl_polarity(impl_def_id) {
                        ImplPolarity::Negative => return ControlFlow::BREAK,
                        ImplPolarity::Reservation => {}
                        // FIXME(@lcnr): That's probably not good enough, idk
                        //
                        // We might just want to take the rustdoc code and somehow avoid
                        // explicit impls for `Self`.
                        ImplPolarity::Positive => return ControlFlow::CONTINUE,
                    }
                }
            }

            match t.kind() {
                ty::Adt(def, substs) if def.is_phantom_data() => substs.super_visit_with(self),
                ty::Adt(def, substs) => {
                    // @lcnr: This is the only place where cycles can happen. We avoid this
                    // by only visiting each `DefId` once.
                    //
                    // This will be is incorrect in subtle cases, but I don't care :)
                    if self.seen.insert(def.did) {
                        for ty in def.all_fields().map(|field| field.ty(tcx, substs)) {
                            ty.visit_with(self)?;
                        }
                    }

                    ControlFlow::CONTINUE
                }
                _ => t.super_visit_with(self),
            }
        }
    }

    let self_ty_root = match self_ty.kind() {
        ty::Adt(def, _) => tcx.mk_adt(def, InternalSubsts::identity_for_item(tcx, def.did)),
        _ => unimplemented!("unexpected self ty {:?}", self_ty),
    };

    self_ty_root
        .visit_with(&mut DisableAutoTraitVisitor {
            tcx,
            self_ty_root,
            trait_def_id,
            seen: FxHashSet::default(),
        })
        .is_break()
}

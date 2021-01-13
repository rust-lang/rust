//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::{self, TyCtxt};
use rustc_trait_selection::traits;

pub fn check(tcx: TyCtxt<'_>) {
    let mut orphan = OrphanChecker { tcx };
    tcx.hir().krate().visit_all_item_likes(&mut orphan);
}

struct OrphanChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'v> for OrphanChecker<'tcx> {
    /// Checks exactly one impl for orphan rules and other such
    /// restrictions. In this fn, it can happen that multiple errors
    /// apply to a specific impl, so just return after reporting one
    /// to prevent inundating the user with a bunch of similar error
    /// reports.
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        let def_id = self.tcx.hir().local_def_id(item.hir_id);
        // "Trait" impl
        if let hir::ItemKind::Impl(hir::Impl {
            generics, of_trait: Some(ref tr), self_ty, ..
        }) = &item.kind
        {
            debug!(
                "coherence2::orphan check: trait impl {}",
                self.tcx.hir().node_to_string(item.hir_id)
            );
            let trait_ref = self.tcx.impl_trait_ref(def_id).unwrap();
            let trait_def_id = trait_ref.def_id;
            let sm = self.tcx.sess.source_map();
            let sp = sm.guess_head_span(item.span);
            match traits::orphan_check(self.tcx, def_id.to_def_id()) {
                Ok(()) => {}
                Err(traits::OrphanCheckErr::NonLocalInputType(tys)) => {
                    let mut err = struct_span_err!(
                        self.tcx.sess,
                        sp,
                        E0117,
                        "only traits defined in the current crate can be implemented for \
                         arbitrary types"
                    );
                    err.span_label(sp, "impl doesn't use only types from inside the current crate");
                    for (ty, is_target_ty) in &tys {
                        let mut ty = *ty;
                        self.tcx.infer_ctxt().enter(|infcx| {
                            // Remove the lifetimes unnecessary for this error.
                            ty = infcx.freshen(ty);
                        });
                        ty = match ty.kind() {
                            // Remove the type arguments from the output, as they are not relevant.
                            // You can think of this as the reverse of `resolve_vars_if_possible`.
                            // That way if we had `Vec<MyType>`, we will properly attribute the
                            // problem to `Vec<T>` and avoid confusing the user if they were to see
                            // `MyType` in the error.
                            ty::Adt(def, _) => self.tcx.mk_adt(def, ty::List::empty()),
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
                            err.span_label(self_ty.span, &msg);
                        } else {
                            // Point at `C<B>` in `impl<A, B> for C<B> in D<A>`
                            err.span_label(tr.path.span, &msg);
                        }
                    }
                    err.note("define and implement a trait or new type instead");
                    err.emit();
                    return;
                }
                Err(traits::OrphanCheckErr::UncoveredTy(param_ty, local_type)) => {
                    let mut sp = sp;
                    for param in generics.params {
                        if param.name.ident().to_string() == param_ty.to_string() {
                            sp = param.span;
                        }
                    }

                    match local_type {
                        Some(local_type) => {
                            struct_span_err!(
                                self.tcx.sess,
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
                            .emit();
                        }
                        None => {
                            struct_span_err!(
                                self.tcx.sess,
                                sp,
                                E0210,
                                "type parameter `{}` must be used as the type parameter for some \
                                local type (e.g., `MyStruct<{}>`)",
                                param_ty,
                                param_ty
                            ).span_label(sp, format!(
                                "type parameter `{}` must be used as the type parameter for some \
                                local type",
                                param_ty,
                            )).note("implementing a foreign trait is only possible if at \
                                    least one of the types for which it is implemented is local"
                            ).note("only traits defined in the current crate can be \
                                    implemented for a type parameter"
                            ).emit();
                        }
                    };
                    return;
                }
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
                self.tcx.trait_is_auto(trait_def_id)
            );
            if self.tcx.trait_is_auto(trait_def_id) && !trait_def_id.is_local() {
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
                                    self.tcx.def_path_str(trait_def_id)
                                ),
                                "can't implement cross-crate trait for type in another crate",
                            ))
                        }
                    }
                    _ => Some((
                        format!(
                            "cross-crate traits with a default impl, like `{}`, can \
                                       only be implemented for a struct/enum type, not `{}`",
                            self.tcx.def_path_str(trait_def_id),
                            self_ty
                        ),
                        "can't implement cross-crate trait with a default impl for \
                               non-struct/enum type",
                    )),
                };

                if let Some((msg, label)) = msg {
                    struct_span_err!(self.tcx.sess, sp, E0321, "{}", msg)
                        .span_label(sp, label)
                        .emit();
                    return;
                }
            }

            if let ty::Opaque(def_id, _) = *trait_ref.self_ty().kind() {
                self.tcx
                    .sess
                    .struct_span_err(sp, "cannot implement trait on type alias impl trait")
                    .span_note(self.tcx.def_span(def_id), "type alias impl trait defined here")
                    .emit();
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
}

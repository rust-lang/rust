//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc::traits;
use rustc::ty::{self, TyCtxt};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;

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
    fn visit_item(&mut self, item: &hir::Item) {
        let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
        // "Trait" impl
        if let hir::ItemKind::Impl(.., Some(_), _, _) = item.node {
            debug!("coherence2::orphan check: trait impl {}",
                   self.tcx.hir().node_to_string(item.hir_id));
            let trait_ref = self.tcx.impl_trait_ref(def_id).unwrap();
            let trait_def_id = trait_ref.def_id;
            let cm = self.tcx.sess.source_map();
            let sp = cm.def_span(item.span);
            match traits::orphan_check(self.tcx, def_id) {
                Ok(()) => {}
                Err(traits::OrphanCheckErr::NoLocalInputType) => {
                    struct_span_err!(self.tcx.sess,
                                     sp,
                                     E0117,
                                     "only traits defined in the current crate can be \
                                      implemented for arbitrary types")
                        .span_label(sp, "impl doesn't use types inside crate")
                        .note("the impl does not reference only types defined in this crate")
                        .note("define and implement a trait or new type instead")
                        .emit();
                    return;
                }
                Err(traits::OrphanCheckErr::UncoveredTy(param_ty)) => {
                    struct_span_err!(self.tcx.sess,
                                     sp,
                                     E0210,
                                     "type parameter `{}` must be used as the type parameter \
                                      for some local type (e.g., `MyStruct<{}>`)",
                                     param_ty,
                                     param_ty)
                        .span_label(sp,
                                    format!("type parameter `{}` must be used as the type \
                                             parameter for some local type", param_ty))
                        .note("only traits defined in the current crate can be implemented \
                               for a type parameter")
                        .emit();
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
            // This final impl is legal according to the orpan
            // rules, but it invalidates the reasoning from
            // `two_foos` above.
            debug!("trait_ref={:?} trait_def_id={:?} trait_is_auto={}",
                   trait_ref,
                   trait_def_id,
                   self.tcx.trait_is_auto(trait_def_id));
            if self.tcx.trait_is_auto(trait_def_id) &&
               !trait_def_id.is_local() {
                let self_ty = trait_ref.self_ty();
                let opt_self_def_id = match self_ty.sty {
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
                                format!("cross-crate traits with a default impl, like `{}`, \
                                         can only be implemented for a struct/enum type \
                                         defined in the current crate",
                                        self.tcx.def_path_str(trait_def_id)),
                                "can't implement cross-crate trait for type in another crate"
                            ))
                        }
                    }
                    _ => {
                        Some((format!("cross-crate traits with a default impl, like `{}`, can \
                                       only be implemented for a struct/enum type, not `{}`",
                                      self.tcx.def_path_str(trait_def_id),
                                      self_ty),
                              "can't implement cross-crate trait with a default impl for \
                               non-struct/enum type"))
                    }
                };

                if let Some((msg, label)) = msg {
                    struct_span_err!(self.tcx.sess, sp, E0321, "{}", msg)
                        .span_label(sp, label)
                        .emit();
                    return;
                }
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

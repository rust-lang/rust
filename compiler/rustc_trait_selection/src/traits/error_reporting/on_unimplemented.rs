use super::{
    ObligationCauseCode, OnUnimplementedDirective, OnUnimplementedNote, PredicateObligation,
};
use crate::infer::InferCtxt;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, GenericParamDefKind};
use rustc_span::symbol::sym;
use std::iter;

use super::InferCtxtPrivExt;

crate trait InferCtxtExt<'tcx> {
    /*private*/
    fn impl_similar_to(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> Option<DefId>;

    /*private*/
    fn describe_enclosure(&self, hir_id: hir::HirId) -> Option<&'static str>;

    fn on_unimplemented_note(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> OnUnimplementedNote;
}

impl<'a, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'a, 'tcx> {
    fn impl_similar_to(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> Option<DefId> {
        let tcx = self.tcx;
        let param_env = obligation.param_env;
        let trait_ref = tcx.erase_late_bound_regions(trait_ref);
        let trait_self_ty = trait_ref.self_ty();

        let mut self_match_impls = vec![];
        let mut fuzzy_match_impls = vec![];

        self.tcx.for_each_relevant_impl(trait_ref.def_id, trait_self_ty, |def_id| {
            let impl_substs = self.fresh_substs_for_item(obligation.cause.span, def_id);
            let impl_trait_ref = tcx.impl_trait_ref(def_id).unwrap().subst(tcx, impl_substs);

            let impl_self_ty = impl_trait_ref.self_ty();

            if let Ok(..) = self.can_eq(param_env, trait_self_ty, impl_self_ty) {
                self_match_impls.push(def_id);

                if iter::zip(
                    trait_ref.substs.types().skip(1),
                    impl_trait_ref.substs.types().skip(1),
                )
                .all(|(u, v)| self.fuzzy_match_tys(u, v))
                {
                    fuzzy_match_impls.push(def_id);
                }
            }
        });

        let impl_def_id = if self_match_impls.len() == 1 {
            self_match_impls[0]
        } else if fuzzy_match_impls.len() == 1 {
            fuzzy_match_impls[0]
        } else {
            return None;
        };

        tcx.has_attr(impl_def_id, sym::rustc_on_unimplemented).then_some(impl_def_id)
    }

    /// Used to set on_unimplemented's `ItemContext`
    /// to be the enclosing (async) block/function/closure
    fn describe_enclosure(&self, hir_id: hir::HirId) -> Option<&'static str> {
        let hir = &self.tcx.hir();
        let node = hir.find(hir_id)?;
        match &node {
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, _, body_id), .. }) => {
                self.describe_generator(*body_id).or_else(|| {
                    Some(match sig.header {
                        hir::FnHeader { asyncness: hir::IsAsync::Async, .. } => "an async function",
                        _ => "a function",
                    })
                })
            }
            hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(body_id)),
                ..
            }) => self.describe_generator(*body_id).or_else(|| Some("a trait method")),
            hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(sig, body_id),
                ..
            }) => self.describe_generator(*body_id).or_else(|| {
                Some(match sig.header {
                    hir::FnHeader { asyncness: hir::IsAsync::Async, .. } => "an async method",
                    _ => "a method",
                })
            }),
            hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(_is_move, _, body_id, _, gen_movability),
                ..
            }) => self.describe_generator(*body_id).or_else(|| {
                Some(if gen_movability.is_some() { "an async closure" } else { "a closure" })
            }),
            hir::Node::Expr(hir::Expr { .. }) => {
                let parent_hid = hir.get_parent_node(hir_id);
                if parent_hid != hir_id { self.describe_enclosure(parent_hid) } else { None }
            }
            _ => None,
        }
    }

    fn on_unimplemented_note(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> OnUnimplementedNote {
        let def_id =
            self.impl_similar_to(trait_ref, obligation).unwrap_or_else(|| trait_ref.def_id());
        let trait_ref = trait_ref.skip_binder();

        let mut flags = vec![(
            sym::ItemContext,
            self.describe_enclosure(obligation.cause.body_id).map(|s| s.to_owned()),
        )];

        match obligation.cause.code {
            ObligationCauseCode::BuiltinDerivedObligation(..)
            | ObligationCauseCode::ImplDerivedObligation(..)
            | ObligationCauseCode::DerivedObligation(..) => {}
            _ => {
                // this is a "direct", user-specified, rather than derived,
                // obligation.
                flags.push((sym::direct, None));
            }
        }

        if let ObligationCauseCode::ItemObligation(item)
        | ObligationCauseCode::BindingObligation(item, _) = obligation.cause.code
        {
            // FIXME: maybe also have some way of handling methods
            // from other traits? That would require name resolution,
            // which we might want to be some sort of hygienic.
            //
            // Currently I'm leaving it for what I need for `try`.
            if self.tcx.trait_of_item(item) == Some(trait_ref.def_id) {
                let method = self.tcx.item_name(item);
                flags.push((sym::from_method, None));
                flags.push((sym::from_method, Some(method.to_string())));
            }
        }

        if let Some(k) = obligation.cause.span.desugaring_kind() {
            flags.push((sym::from_desugaring, None));
            flags.push((sym::from_desugaring, Some(format!("{:?}", k))));
        }

        // Add all types without trimmed paths.
        ty::print::with_no_trimmed_paths(|| {
            let generics = self.tcx.generics_of(def_id);
            let self_ty = trait_ref.self_ty();
            // This is also included through the generics list as `Self`,
            // but the parser won't allow you to use it
            flags.push((sym::_Self, Some(self_ty.to_string())));
            if let Some(def) = self_ty.ty_adt_def() {
                // We also want to be able to select self's original
                // signature with no type arguments resolved
                flags.push((sym::_Self, Some(self.tcx.type_of(def.did).to_string())));
            }

            for param in generics.params.iter() {
                let value = match param.kind {
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                        trait_ref.substs[param.index as usize].to_string()
                    }
                    GenericParamDefKind::Lifetime => continue,
                };
                let name = param.name;
                flags.push((name, Some(value)));

                if let GenericParamDefKind::Type { .. } = param.kind {
                    let param_ty = trait_ref.substs[param.index as usize].expect_ty();
                    if let Some(def) = param_ty.ty_adt_def() {
                        // We also want to be able to select the parameter's
                        // original signature with no type arguments resolved
                        flags.push((name, Some(self.tcx.type_of(def.did).to_string())));
                    }
                }
            }

            if let Some(true) = self_ty.ty_adt_def().map(|def| def.did.is_local()) {
                flags.push((sym::crate_local, None));
            }

            // Allow targeting all integers using `{integral}`, even if the exact type was resolved
            if self_ty.is_integral() {
                flags.push((sym::_Self, Some("{integral}".to_owned())));
            }

            if let ty::Array(aty, len) = self_ty.kind() {
                flags.push((sym::_Self, Some("[]".to_owned())));
                flags.push((sym::_Self, Some(format!("[{}]", aty))));
                if let Some(def) = aty.ty_adt_def() {
                    // We also want to be able to select the array's type's original
                    // signature with no type arguments resolved
                    let type_string = self.tcx.type_of(def.did).to_string();
                    flags.push((sym::_Self, Some(format!("[{}]", type_string))));

                    let len = len.val.try_to_value().and_then(|v| v.try_to_machine_usize(self.tcx));
                    let string = match len {
                        Some(n) => format!("[{}; {}]", type_string, n),
                        None => format!("[{}; _]", type_string),
                    };
                    flags.push((sym::_Self, Some(string)));
                }
            }
            if let ty::Dynamic(traits, _) = self_ty.kind() {
                for t in traits.iter() {
                    if let ty::ExistentialPredicate::Trait(trait_ref) = t.skip_binder() {
                        flags.push((sym::_Self, Some(self.tcx.def_path_str(trait_ref.def_id))))
                    }
                }
            }
        });

        if let Ok(Some(command)) =
            OnUnimplementedDirective::of_item(self.tcx, trait_ref.def_id, def_id)
        {
            command.evaluate(self.tcx, trait_ref, &flags[..])
        } else {
            OnUnimplementedNote::default()
        }
    }
}

use std::path::PathBuf;

use rustc_hir as hir;
use rustc_hir::attrs::diagnostic::{ConditionOptions, FormatArgs, OnUnimplementedNote};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::find_attr;
pub use rustc_hir::lints::FormatWarning;
use rustc_middle::ty::print::PrintTraitRefExt;
use rustc_middle::ty::{self, GenericParamDef, GenericParamDefKind};
use rustc_span::Symbol;

use super::{ObligationCauseCode, PredicateObligation};
use crate::error_reporting::TypeErrCtxt;

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    /// Used to set on_unimplemented's `ItemContext`
    /// to be the enclosing (async) block/function/closure
    fn describe_enclosure(&self, def_id: LocalDefId) -> Option<&'static str> {
        match self.tcx.hir_node_by_def_id(def_id) {
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { .. }, .. }) => Some("a function"),
            hir::Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Fn(..), .. }) => {
                Some("a trait method")
            }
            hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Fn(..), .. }) => {
                Some("a method")
            }
            hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(hir::Closure { kind, .. }),
                ..
            }) => Some(self.describe_closure(*kind)),
            _ => None,
        }
    }

    pub fn on_unimplemented_note(
        &self,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        long_ty_path: &mut Option<PathBuf>,
    ) -> OnUnimplementedNote {
        if trait_pred.polarity() != ty::PredicatePolarity::Positive {
            return OnUnimplementedNote::default();
        }
        let (condition_options, format_args) =
            self.on_unimplemented_components(trait_pred, obligation, long_ty_path);
        if let Some(command) = find_attr!(self.tcx, trait_pred.def_id(), OnUnimplemented {directive, ..} => directive.as_deref()).flatten() {
            command.evaluate_directive(
                trait_pred.skip_binder().trait_ref,
                &condition_options,
                &format_args,
            )
        } else {
            OnUnimplementedNote::default()
        }
    }

    pub(crate) fn on_unimplemented_components(
        &self,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        long_ty_path: &mut Option<PathBuf>,
    ) -> (ConditionOptions, FormatArgs) {
        let (def_id, args) = (trait_pred.def_id(), trait_pred.skip_binder().trait_ref.args);
        let trait_pred = trait_pred.skip_binder();

        let mut self_types = vec![];
        let mut generic_args: Vec<(Symbol, String)> = vec![];
        let mut crate_local = false;
        // FIXME(-Zlower-impl-trait-in-trait-to-assoc-ty): HIR is not present for RPITITs,
        // but I guess we could synthesize one here. We don't see any errors that rely on
        // that yet, though.
        let item_context = self.describe_enclosure(obligation.cause.body_id).unwrap_or("");

        let direct = match obligation.cause.code() {
            ObligationCauseCode::BuiltinDerived(..)
            | ObligationCauseCode::ImplDerived(..)
            | ObligationCauseCode::WellFormedDerived(..) => false,
            _ => {
                // this is a "direct", user-specified, rather than derived,
                // obligation.
                true
            }
        };

        let from_desugaring = obligation.cause.span.desugaring_kind();

        let cause = if let ObligationCauseCode::MainFunctionType = obligation.cause.code() {
            Some("MainFunctionType".to_string())
        } else {
            None
        };

        // Add all types without trimmed paths or visible paths, ensuring they end up with
        // their "canonical" def path.
        ty::print::with_no_trimmed_paths!(ty::print::with_no_visible_paths!({
            let generics = self.tcx.generics_of(def_id);
            let self_ty = trait_pred.self_ty();
            self_types.push(self_ty.to_string());
            if let Some(def) = self_ty.ty_adt_def() {
                // We also want to be able to select self's original
                // signature with no type arguments resolved
                self_types.push(self.tcx.type_of(def.did()).instantiate_identity().to_string());
            }

            for GenericParamDef { name, kind, index, .. } in generics.own_params.iter() {
                let value = match kind {
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                        args[*index as usize].to_string()
                    }
                    GenericParamDefKind::Lifetime => continue,
                };
                generic_args.push((*name, value));

                if let GenericParamDefKind::Type { .. } = kind {
                    let param_ty = args[*index as usize].expect_ty();
                    if let Some(def) = param_ty.ty_adt_def() {
                        // We also want to be able to select the parameter's
                        // original signature with no type arguments resolved
                        generic_args.push((
                            *name,
                            self.tcx.type_of(def.did()).instantiate_identity().to_string(),
                        ));
                    }
                }
            }

            if let Some(true) = self_ty.ty_adt_def().map(|def| def.did().is_local()) {
                crate_local = true;
            }

            // Allow targeting all integers using `{integral}`, even if the exact type was resolved
            if self_ty.is_integral() {
                self_types.push("{integral}".to_owned());
            }

            if self_ty.is_array_slice() {
                self_types.push("&[]".to_owned());
            }

            if self_ty.is_fn() {
                let fn_sig = self_ty.fn_sig(self.tcx);
                let shortname = if let ty::FnDef(def_id, _) = *self_ty.kind()
                    && self.tcx.codegen_fn_attrs(def_id).safe_target_features
                {
                    "#[target_feature] fn"
                } else {
                    match fn_sig.safety() {
                        hir::Safety::Safe => "fn",
                        hir::Safety::Unsafe => "unsafe fn",
                    }
                };
                self_types.push(shortname.to_owned());
            }

            // Slices give us `[]`, `[{ty}]`
            if let ty::Slice(aty) = self_ty.kind() {
                self_types.push("[]".to_owned());
                if let Some(def) = aty.ty_adt_def() {
                    // We also want to be able to select the slice's type's original
                    // signature with no type arguments resolved
                    self_types
                        .push(format!("[{}]", self.tcx.type_of(def.did()).instantiate_identity()));
                }
                if aty.is_integral() {
                    self_types.push("[{integral}]".to_string());
                }
            }

            // Arrays give us `[]`, `[{ty}; _]` and `[{ty}; N]`
            if let ty::Array(aty, len) = self_ty.kind() {
                self_types.push("[]".to_string());
                let len = len.try_to_target_usize(self.tcx);
                self_types.push(format!("[{aty}; _]"));
                if let Some(n) = len {
                    self_types.push(format!("[{aty}; {n}]"));
                }
                if let Some(def) = aty.ty_adt_def() {
                    // We also want to be able to select the array's type's original
                    // signature with no type arguments resolved
                    let def_ty = self.tcx.type_of(def.did()).instantiate_identity();
                    self_types.push(format!("[{def_ty}; _]"));
                    if let Some(n) = len {
                        self_types.push(format!("[{def_ty}; {n}]"));
                    }
                }
                if aty.is_integral() {
                    self_types.push("[{integral}; _]".to_string());
                    if let Some(n) = len {
                        self_types.push(format!("[{{integral}}; {n}]"));
                    }
                }
            }
            if let ty::Dynamic(traits, _) = self_ty.kind() {
                for t in traits.iter() {
                    if let ty::ExistentialPredicate::Trait(trait_ref) = t.skip_binder() {
                        self_types.push(self.tcx.def_path_str(trait_ref.def_id));
                    }
                }
            }

            // `&[{integral}]` - `FromIterator` needs that.
            if let ty::Ref(_, ref_ty, rustc_ast::Mutability::Not) = self_ty.kind()
                && let ty::Slice(sty) = ref_ty.kind()
                && sty.is_integral()
            {
                self_types.push("&[{integral}]".to_owned());
            }
        }));

        let this = self.tcx.def_path_str(trait_pred.trait_ref.def_id);
        let trait_sugared = trait_pred.trait_ref.print_trait_sugared().to_string();

        let condition_options = ConditionOptions {
            self_types,
            from_desugaring,
            cause,
            crate_local,
            direct,
            generic_args,
        };

        // Unlike the generic_args earlier,
        // this one is *not* collected under `with_no_trimmed_paths!`
        // for printing the type to the user
        //
        // This includes `Self`, as it is the first parameter in `own_params`.
        let generic_args = self
            .tcx
            .generics_of(trait_pred.trait_ref.def_id)
            .own_params
            .iter()
            .filter_map(|param| {
                let value = match param.kind {
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                        if let Some(ty) = trait_pred.trait_ref.args[param.index as usize].as_type()
                        {
                            self.tcx.short_string(ty, long_ty_path)
                        } else {
                            trait_pred.trait_ref.args[param.index as usize].to_string()
                        }
                    }
                    GenericParamDefKind::Lifetime => return None,
                };
                let name = param.name;
                Some((name, value))
            })
            .collect();

        let format_args = FormatArgs { this, trait_sugared, generic_args, item_context };
        (condition_options, format_args)
    }
}

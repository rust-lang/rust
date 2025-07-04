use std::iter;
use std::path::PathBuf;

use rustc_ast::{LitKind, MetaItem, MetaItemInner, MetaItemKind, MetaItemLit};
use rustc_errors::codes::*;
use rustc_errors::{ErrorGuaranteed, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{AttrArgs, Attribute};
use rustc_macros::LintDiagnostic;
use rustc_middle::bug;
use rustc_middle::ty::print::PrintTraitRefExt;
use rustc_middle::ty::{self, GenericArgsRef, GenericParamDef, GenericParamDefKind, TyCtxt};
use rustc_session::lint::builtin::UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::{Span, Symbol, sym};
use tracing::{debug, info};

use super::{ObligationCauseCode, PredicateObligation};
use crate::error_reporting::TypeErrCtxt;
use crate::error_reporting::traits::on_unimplemented_condition::{
    ConditionOptions, OnUnimplementedCondition,
};
use crate::error_reporting::traits::on_unimplemented_format::{
    Ctx, FormatArgs, FormatString, FormatWarning,
};
use crate::errors::{InvalidOnClause, NoValueInOnUnimplemented};
use crate::infer::InferCtxtExt;

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    fn impl_similar_to(
        &self,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> Option<(DefId, GenericArgsRef<'tcx>)> {
        let tcx = self.tcx;
        let param_env = obligation.param_env;
        self.enter_forall(trait_pred, |trait_pred| {
            let trait_self_ty = trait_pred.self_ty();

            let mut self_match_impls = vec![];
            let mut fuzzy_match_impls = vec![];

            self.tcx.for_each_relevant_impl(trait_pred.def_id(), trait_self_ty, |def_id| {
                let impl_args = self.fresh_args_for_item(obligation.cause.span, def_id);
                let impl_trait_ref =
                    tcx.impl_trait_ref(def_id).unwrap().instantiate(tcx, impl_args);

                let impl_self_ty = impl_trait_ref.self_ty();

                if self.can_eq(param_env, trait_self_ty, impl_self_ty) {
                    self_match_impls.push((def_id, impl_args));

                    if iter::zip(
                        trait_pred.trait_ref.args.types().skip(1),
                        impl_trait_ref.args.types().skip(1),
                    )
                    .all(|(u, v)| self.fuzzy_match_tys(u, v, false).is_some())
                    {
                        fuzzy_match_impls.push((def_id, impl_args));
                    }
                }
            });

            let impl_def_id_and_args = if let [impl_] = self_match_impls[..] {
                impl_
            } else if let [impl_] = fuzzy_match_impls[..] {
                impl_
            } else {
                return None;
            };

            tcx.has_attr(impl_def_id_and_args.0, sym::rustc_on_unimplemented)
                .then_some(impl_def_id_and_args)
        })
    }

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
        long_ty_file: &mut Option<PathBuf>,
    ) -> OnUnimplementedNote {
        if trait_pred.polarity() != ty::PredicatePolarity::Positive {
            return OnUnimplementedNote::default();
        }

        let (def_id, args) = self
            .impl_similar_to(trait_pred, obligation)
            .unwrap_or_else(|| (trait_pred.def_id(), trait_pred.skip_binder().trait_ref.args));
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
                let shortname = if let ty::FnDef(def_id, _) = self_ty.kind()
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
            if let ty::Dynamic(traits, _, _) = self_ty.kind() {
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
        let trait_sugared = trait_pred.trait_ref.print_trait_sugared();

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
                            self.tcx.short_string(ty, long_ty_file)
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

        if let Ok(Some(command)) = OnUnimplementedDirective::of_item(self.tcx, def_id) {
            command.evaluate(self.tcx, trait_pred.trait_ref, &condition_options, &format_args)
        } else {
            OnUnimplementedNote::default()
        }
    }
}

/// Represents a format string in a on_unimplemented attribute,
/// like the "content" in `#[diagnostic::on_unimplemented(message = "content")]`
#[derive(Clone, Debug)]
pub struct OnUnimplementedFormatString {
    /// Symbol of the format string, i.e. `"content"`
    symbol: Symbol,
    ///The span of the format string, i.e. `"content"`
    span: Span,
    is_diagnostic_namespace_variant: bool,
}

#[derive(Debug)]
pub struct OnUnimplementedDirective {
    condition: Option<OnUnimplementedCondition>,
    subcommands: Vec<OnUnimplementedDirective>,
    message: Option<(Span, OnUnimplementedFormatString)>,
    label: Option<(Span, OnUnimplementedFormatString)>,
    notes: Vec<OnUnimplementedFormatString>,
    parent_label: Option<OnUnimplementedFormatString>,
    append_const_msg: Option<AppendConstMessage>,
}

/// For the `#[rustc_on_unimplemented]` attribute
#[derive(Default)]
pub struct OnUnimplementedNote {
    pub message: Option<String>,
    pub label: Option<String>,
    pub notes: Vec<String>,
    pub parent_label: Option<String>,
    // If none, should fall back to a generic message
    pub append_const_msg: Option<AppendConstMessage>,
}

/// Append a message for `[const] Trait` errors.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AppendConstMessage {
    #[default]
    Default,
    Custom(Symbol, Span),
}

#[derive(LintDiagnostic)]
#[diag(trait_selection_malformed_on_unimplemented_attr)]
#[help]
pub struct MalformedOnUnimplementedAttrLint {
    #[label]
    pub span: Span,
}

impl MalformedOnUnimplementedAttrLint {
    pub fn new(span: Span) -> Self {
        Self { span }
    }
}

#[derive(LintDiagnostic)]
#[diag(trait_selection_missing_options_for_on_unimplemented_attr)]
#[help]
pub struct MissingOptionsForOnUnimplementedAttr;

#[derive(LintDiagnostic)]
#[diag(trait_selection_ignored_diagnostic_option)]
pub struct IgnoredDiagnosticOption {
    pub option_name: &'static str,
    #[label]
    pub span: Span,
    #[label(trait_selection_other_label)]
    pub prev_span: Span,
}

impl IgnoredDiagnosticOption {
    pub fn maybe_emit_warning<'tcx>(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        new: Option<Span>,
        old: Option<Span>,
        option_name: &'static str,
    ) {
        if let (Some(new_item), Some(old_item)) = (new, old) {
            if let Some(item_def_id) = item_def_id.as_local() {
                tcx.emit_node_span_lint(
                    UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                    tcx.local_def_id_to_hir_id(item_def_id),
                    new_item,
                    IgnoredDiagnosticOption { span: new_item, prev_span: old_item, option_name },
                );
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(trait_selection_wrapped_parser_error)]
pub struct WrappedParserError {
    pub description: String,
    pub label: String,
}

impl<'tcx> OnUnimplementedDirective {
    fn parse(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        items: &[MetaItemInner],
        span: Span,
        is_root: bool,
        is_diagnostic_namespace_variant: bool,
    ) -> Result<Option<Self>, ErrorGuaranteed> {
        let mut errored = None;
        let mut item_iter = items.iter();

        let parse_value = |value_str, span| {
            OnUnimplementedFormatString::try_parse(
                tcx,
                item_def_id,
                value_str,
                span,
                is_diagnostic_namespace_variant,
            )
            .map(Some)
        };

        let condition = if is_root {
            None
        } else {
            let cond = item_iter
                .next()
                .ok_or_else(|| tcx.dcx().emit_err(InvalidOnClause::Empty { span }))?;

            let generics: Vec<Symbol> = tcx
                .generics_of(item_def_id)
                .own_params
                .iter()
                .filter_map(|param| {
                    if matches!(param.kind, GenericParamDefKind::Lifetime) {
                        None
                    } else {
                        Some(param.name)
                    }
                })
                .collect();
            match OnUnimplementedCondition::parse(cond, &generics) {
                Ok(condition) => Some(condition),
                Err(e) => return Err(tcx.dcx().emit_err(e)),
            }
        };

        let mut message = None;
        let mut label = None;
        let mut notes = Vec::new();
        let mut parent_label = None;
        let mut subcommands = vec![];
        let mut append_const_msg = None;

        let get_value_and_span = |item: &_, key| {
            if let MetaItemInner::MetaItem(MetaItem {
                path,
                kind: MetaItemKind::NameValue(MetaItemLit { span, kind: LitKind::Str(s, _), .. }),
                ..
            }) = item
                && *path == key
            {
                Some((*s, *span))
            } else {
                None
            }
        };

        for item in item_iter {
            if let Some((message_, span)) = get_value_and_span(item, sym::message)
                && message.is_none()
            {
                message = parse_value(message_, span)?.map(|l| (item.span(), l));
                continue;
            } else if let Some((label_, span)) = get_value_and_span(item, sym::label)
                && label.is_none()
            {
                label = parse_value(label_, span)?.map(|l| (item.span(), l));
                continue;
            } else if let Some((note_, span)) = get_value_and_span(item, sym::note) {
                if let Some(note) = parse_value(note_, span)? {
                    notes.push(note);
                    continue;
                }
            } else if item.has_name(sym::parent_label)
                && parent_label.is_none()
                && !is_diagnostic_namespace_variant
            {
                if let Some(parent_label_) = item.value_str() {
                    parent_label = parse_value(parent_label_, item.span())?;
                    continue;
                }
            } else if item.has_name(sym::on)
                && is_root
                && message.is_none()
                && label.is_none()
                && notes.is_empty()
                && !is_diagnostic_namespace_variant
            // FIXME(diagnostic_namespace): disallow filters for now
            {
                if let Some(items) = item.meta_item_list() {
                    match Self::parse(
                        tcx,
                        item_def_id,
                        items,
                        item.span(),
                        false,
                        is_diagnostic_namespace_variant,
                    ) {
                        Ok(Some(subcommand)) => subcommands.push(subcommand),
                        Ok(None) => bug!(
                            "This cannot happen for now as we only reach that if `is_diagnostic_namespace_variant` is false"
                        ),
                        Err(reported) => errored = Some(reported),
                    };
                    continue;
                }
            } else if item.has_name(sym::append_const_msg)
                && append_const_msg.is_none()
                && !is_diagnostic_namespace_variant
            {
                if let Some(msg) = item.value_str() {
                    append_const_msg = Some(AppendConstMessage::Custom(msg, item.span()));
                    continue;
                } else if item.is_word() {
                    append_const_msg = Some(AppendConstMessage::Default);
                    continue;
                }
            }

            if is_diagnostic_namespace_variant {
                if let Some(def_id) = item_def_id.as_local() {
                    tcx.emit_node_span_lint(
                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        tcx.local_def_id_to_hir_id(def_id),
                        vec![item.span()],
                        MalformedOnUnimplementedAttrLint::new(item.span()),
                    );
                }
            } else {
                // nothing found
                tcx.dcx().emit_err(NoValueInOnUnimplemented { span: item.span() });
            }
        }

        if let Some(reported) = errored {
            if is_diagnostic_namespace_variant { Ok(None) } else { Err(reported) }
        } else {
            Ok(Some(OnUnimplementedDirective {
                condition,
                subcommands,
                message,
                label,
                notes,
                parent_label,
                append_const_msg,
            }))
        }
    }

    pub fn of_item(tcx: TyCtxt<'tcx>, item_def_id: DefId) -> Result<Option<Self>, ErrorGuaranteed> {
        if !tcx.is_trait(item_def_id) {
            // It could be a trait_alias (`trait MyTrait = SomeOtherTrait`)
            // or an implementation (`impl MyTrait for Foo {}`)
            //
            // We don't support those.
            return Ok(None);
        }
        if let Some(attr) = tcx.get_attr(item_def_id, sym::rustc_on_unimplemented) {
            return Self::parse_attribute(attr, false, tcx, item_def_id);
        } else {
            tcx.get_attrs_by_path(item_def_id, &[sym::diagnostic, sym::on_unimplemented])
                .filter_map(|attr| Self::parse_attribute(attr, true, tcx, item_def_id).transpose())
                .try_fold(None, |aggr: Option<Self>, directive| {
                    let directive = directive?;
                    if let Some(aggr) = aggr {
                        let mut subcommands = aggr.subcommands;
                        subcommands.extend(directive.subcommands);
                        let mut notes = aggr.notes;
                        notes.extend(directive.notes);
                        IgnoredDiagnosticOption::maybe_emit_warning(
                            tcx,
                            item_def_id,
                            directive.message.as_ref().map(|f| f.0),
                            aggr.message.as_ref().map(|f| f.0),
                            "message",
                        );
                        IgnoredDiagnosticOption::maybe_emit_warning(
                            tcx,
                            item_def_id,
                            directive.label.as_ref().map(|f| f.0),
                            aggr.label.as_ref().map(|f| f.0),
                            "label",
                        );
                        IgnoredDiagnosticOption::maybe_emit_warning(
                            tcx,
                            item_def_id,
                            directive.condition.as_ref().map(|i| i.span()),
                            aggr.condition.as_ref().map(|i| i.span()),
                            "condition",
                        );
                        IgnoredDiagnosticOption::maybe_emit_warning(
                            tcx,
                            item_def_id,
                            directive.parent_label.as_ref().map(|f| f.span),
                            aggr.parent_label.as_ref().map(|f| f.span),
                            "parent_label",
                        );
                        IgnoredDiagnosticOption::maybe_emit_warning(
                            tcx,
                            item_def_id,
                            directive.append_const_msg.as_ref().and_then(|c| {
                                if let AppendConstMessage::Custom(_, s) = c {
                                    Some(*s)
                                } else {
                                    None
                                }
                            }),
                            aggr.append_const_msg.as_ref().and_then(|c| {
                                if let AppendConstMessage::Custom(_, s) = c {
                                    Some(*s)
                                } else {
                                    None
                                }
                            }),
                            "append_const_msg",
                        );

                        Ok(Some(Self {
                            condition: aggr.condition.or(directive.condition),
                            subcommands,
                            message: aggr.message.or(directive.message),
                            label: aggr.label.or(directive.label),
                            notes,
                            parent_label: aggr.parent_label.or(directive.parent_label),
                            append_const_msg: aggr.append_const_msg.or(directive.append_const_msg),
                        }))
                    } else {
                        Ok(Some(directive))
                    }
                })
        }
    }

    fn parse_attribute(
        attr: &Attribute,
        is_diagnostic_namespace_variant: bool,
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
    ) -> Result<Option<Self>, ErrorGuaranteed> {
        let result = if let Some(items) = attr.meta_item_list() {
            Self::parse(
                tcx,
                item_def_id,
                &items,
                attr.span(),
                true,
                is_diagnostic_namespace_variant,
            )
        } else if let Some(value) = attr.value_str() {
            if !is_diagnostic_namespace_variant {
                Ok(Some(OnUnimplementedDirective {
                    condition: None,
                    message: None,
                    subcommands: vec![],
                    label: Some((
                        attr.span(),
                        OnUnimplementedFormatString::try_parse(
                            tcx,
                            item_def_id,
                            value,
                            attr.value_span().unwrap_or(attr.span()),
                            is_diagnostic_namespace_variant,
                        )?,
                    )),
                    notes: Vec::new(),
                    parent_label: None,
                    append_const_msg: None,
                }))
            } else {
                let item = attr.get_normal_item();
                let report_span = match &item.args {
                    AttrArgs::Empty => item.path.span,
                    AttrArgs::Delimited(args) => args.dspan.entire(),
                    AttrArgs::Eq { eq_span, expr } => eq_span.to(expr.span),
                };

                if let Some(item_def_id) = item_def_id.as_local() {
                    tcx.emit_node_span_lint(
                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        tcx.local_def_id_to_hir_id(item_def_id),
                        report_span,
                        MalformedOnUnimplementedAttrLint::new(report_span),
                    );
                }
                Ok(None)
            }
        } else if is_diagnostic_namespace_variant {
            match attr {
                Attribute::Unparsed(p) if !matches!(p.args, AttrArgs::Empty) => {
                    if let Some(item_def_id) = item_def_id.as_local() {
                        tcx.emit_node_span_lint(
                            UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                            tcx.local_def_id_to_hir_id(item_def_id),
                            attr.span(),
                            MalformedOnUnimplementedAttrLint::new(attr.span()),
                        );
                    }
                }
                _ => {
                    if let Some(item_def_id) = item_def_id.as_local() {
                        tcx.emit_node_span_lint(
                            UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                            tcx.local_def_id_to_hir_id(item_def_id),
                            attr.span(),
                            MissingOptionsForOnUnimplementedAttr,
                        )
                    }
                }
            };

            Ok(None)
        } else {
            let reported = tcx.dcx().delayed_bug("of_item: neither meta_item_list nor value_str");
            return Err(reported);
        };
        debug!("of_item({:?}) = {:?}", item_def_id, result);
        result
    }

    pub(crate) fn evaluate(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
        condition_options: &ConditionOptions,
        args: &FormatArgs<'tcx>,
    ) -> OnUnimplementedNote {
        let mut message = None;
        let mut label = None;
        let mut notes = Vec::new();
        let mut parent_label = None;
        let mut append_const_msg = None;
        info!(
            "evaluate({:?}, trait_ref={:?}, options={:?}, args ={:?})",
            self, trait_ref, condition_options, args
        );

        for command in self.subcommands.iter().chain(Some(self)).rev() {
            debug!(?command);
            if let Some(ref condition) = command.condition
                && !condition.matches_predicate(condition_options)
            {
                debug!("evaluate: skipping {:?} due to condition", command);
                continue;
            }
            debug!("evaluate: {:?} succeeded", command);
            if let Some(ref message_) = command.message {
                message = Some(message_.clone());
            }

            if let Some(ref label_) = command.label {
                label = Some(label_.clone());
            }

            notes.extend(command.notes.clone());

            if let Some(ref parent_label_) = command.parent_label {
                parent_label = Some(parent_label_.clone());
            }

            append_const_msg = command.append_const_msg;
        }

        OnUnimplementedNote {
            label: label.map(|l| l.1.format(tcx, trait_ref, args)),
            message: message.map(|m| m.1.format(tcx, trait_ref, args)),
            notes: notes.into_iter().map(|n| n.format(tcx, trait_ref, args)).collect(),
            parent_label: parent_label.map(|e_s| e_s.format(tcx, trait_ref, args)),
            append_const_msg,
        }
    }
}

impl<'tcx> OnUnimplementedFormatString {
    fn try_parse(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        from: Symbol,
        span: Span,
        is_diagnostic_namespace_variant: bool,
    ) -> Result<Self, ErrorGuaranteed> {
        let result =
            OnUnimplementedFormatString { symbol: from, span, is_diagnostic_namespace_variant };
        result.verify(tcx, item_def_id)?;
        Ok(result)
    }

    fn verify(&self, tcx: TyCtxt<'tcx>, trait_def_id: DefId) -> Result<(), ErrorGuaranteed> {
        if !tcx.is_trait(trait_def_id) {
            return Ok(());
        };

        let ctx = if self.is_diagnostic_namespace_variant {
            Ctx::DiagnosticOnUnimplemented { tcx, trait_def_id }
        } else {
            Ctx::RustcOnUnimplemented { tcx, trait_def_id }
        };

        let mut result = Ok(());

        let snippet = tcx.sess.source_map().span_to_snippet(self.span).ok();
        match FormatString::parse(self.symbol, snippet, self.span, &ctx) {
            // Warnings about format specifiers, deprecated parameters, wrong parameters etc.
            // In other words we'd like to let the author know, but we can still try to format the string later
            Ok(FormatString { warnings, .. }) => {
                if self.is_diagnostic_namespace_variant {
                    for w in warnings {
                        w.emit_warning(tcx, trait_def_id)
                    }
                } else {
                    for w in warnings {
                        match w {
                            FormatWarning::UnknownParam { argument_name, span } => {
                                let reported = struct_span_code_err!(
                                    tcx.dcx(),
                                    span,
                                    E0230,
                                    "cannot find parameter {} on this trait",
                                    argument_name,
                                )
                                .emit();
                                result = Err(reported);
                            }
                            FormatWarning::PositionalArgument { span, .. } => {
                                let reported = struct_span_code_err!(
                                    tcx.dcx(),
                                    span,
                                    E0231,
                                    "positional format arguments are not allowed here"
                                )
                                .emit();
                                result = Err(reported);
                            }
                            FormatWarning::InvalidSpecifier { .. }
                            | FormatWarning::FutureIncompat { .. } => {}
                        }
                    }
                }
            }
            // Error from the underlying `rustc_parse_format::Parser`
            Err(e) => {
                // we cannot return errors from processing the format string as hard error here
                // as the diagnostic namespace guarantees that malformed input cannot cause an error
                //
                // if we encounter any error while processing we nevertheless want to show it as warning
                // so that users are aware that something is not correct
                if self.is_diagnostic_namespace_variant {
                    if let Some(trait_def_id) = trait_def_id.as_local() {
                        tcx.emit_node_span_lint(
                            UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                            tcx.local_def_id_to_hir_id(trait_def_id),
                            self.span,
                            WrappedParserError { description: e.description, label: e.label },
                        );
                    }
                } else {
                    let reported =
                        struct_span_code_err!(tcx.dcx(), self.span, E0231, "{}", e.description,)
                            .emit();
                    result = Err(reported);
                }
            }
        }

        result
    }

    pub fn format(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
        args: &FormatArgs<'tcx>,
    ) -> String {
        let trait_def_id = trait_ref.def_id;
        let ctx = if self.is_diagnostic_namespace_variant {
            Ctx::DiagnosticOnUnimplemented { tcx, trait_def_id }
        } else {
            Ctx::RustcOnUnimplemented { tcx, trait_def_id }
        };

        // No point passing a snippet here, we already did that in `verify`
        if let Ok(s) = FormatString::parse(self.symbol, None, self.span, &ctx) {
            s.format(args)
        } else {
            // we cannot return errors from processing the format string as hard error here
            // as the diagnostic namespace guarantees that malformed input cannot cause an error
            //
            // if we encounter any error while processing the format string
            // we don't want to show the potentially half assembled formatted string,
            // therefore we fall back to just showing the input string in this case
            //
            // The actual parser errors are emitted earlier
            // as lint warnings in OnUnimplementedFormatString::verify
            self.symbol.as_str().into()
        }
    }
}

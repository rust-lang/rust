use std::iter;
use std::path::PathBuf;

use rustc_ast::MetaItemInner;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::codes::*;
use rustc_errors::{ErrorGuaranteed, struct_span_code_err};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{AttrArgs, AttrKind, Attribute};
use rustc_macros::LintDiagnostic;
use rustc_middle::bug;
use rustc_middle::ty::print::PrintTraitRefExt as _;
use rustc_middle::ty::{self, GenericArgsRef, GenericParamDefKind, ToPolyTraitRef, TyCtxt};
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_session::lint::builtin::UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::Span;
use rustc_span::symbol::{Symbol, kw, sym};
use tracing::{debug, info};
use {rustc_attr_parsing as attr, rustc_hir as hir};

use super::{ObligationCauseCode, PredicateObligation};
use crate::error_reporting::TypeErrCtxt;
use crate::errors::{
    EmptyOnClauseInOnUnimplemented, InvalidOnClauseInOnUnimplemented, NoValueInOnUnimplemented,
};
use crate::infer::InferCtxtExt;

/// The symbols which are always allowed in a format string
static ALLOWED_FORMAT_SYMBOLS: &[Symbol] = &[
    kw::SelfUpper,
    sym::ItemContext,
    sym::from_desugaring,
    sym::direct,
    sym::cause,
    sym::integral,
    sym::integer_,
    sym::float,
    sym::_Self,
    sym::crate_local,
    sym::Trait,
];

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    fn impl_similar_to(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> Option<(DefId, GenericArgsRef<'tcx>)> {
        let tcx = self.tcx;
        let param_env = obligation.param_env;
        self.enter_forall(trait_ref, |trait_ref| {
            let trait_self_ty = trait_ref.self_ty();

            let mut self_match_impls = vec![];
            let mut fuzzy_match_impls = vec![];

            self.tcx.for_each_relevant_impl(trait_ref.def_id, trait_self_ty, |def_id| {
                let impl_args = self.fresh_args_for_item(obligation.cause.span, def_id);
                let impl_trait_ref =
                    tcx.impl_trait_ref(def_id).unwrap().instantiate(tcx, impl_args);

                let impl_self_ty = impl_trait_ref.self_ty();

                if self.can_eq(param_env, trait_self_ty, impl_self_ty) {
                    self_match_impls.push((def_id, impl_args));

                    if iter::zip(
                        trait_ref.args.types().skip(1),
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
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(..), .. }) => Some("a function"),
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
            .impl_similar_to(trait_pred.to_poly_trait_ref(), obligation)
            .unwrap_or_else(|| (trait_pred.def_id(), trait_pred.skip_binder().trait_ref.args));
        let trait_pred = trait_pred.skip_binder();

        let mut flags = vec![];
        // FIXME(-Zlower-impl-trait-in-trait-to-assoc-ty): HIR is not present for RPITITs,
        // but I guess we could synthesize one here. We don't see any errors that rely on
        // that yet, though.
        let enclosure = self.describe_enclosure(obligation.cause.body_id).map(|t| t.to_owned());
        flags.push((sym::ItemContext, enclosure));

        match obligation.cause.code() {
            ObligationCauseCode::BuiltinDerived(..)
            | ObligationCauseCode::ImplDerived(..)
            | ObligationCauseCode::WellFormedDerived(..) => {}
            _ => {
                // this is a "direct", user-specified, rather than derived,
                // obligation.
                flags.push((sym::direct, None));
            }
        }

        if let Some(k) = obligation.cause.span.desugaring_kind() {
            flags.push((sym::from_desugaring, None));
            flags.push((sym::from_desugaring, Some(format!("{k:?}"))));
        }

        if let ObligationCauseCode::MainFunctionType = obligation.cause.code() {
            flags.push((sym::cause, Some("MainFunctionType".to_string())));
        }

        flags.push((sym::Trait, Some(trait_pred.trait_ref.print_trait_sugared().to_string())));

        // Add all types without trimmed paths or visible paths, ensuring they end up with
        // their "canonical" def path.
        ty::print::with_no_trimmed_paths!(ty::print::with_no_visible_paths!({
            let generics = self.tcx.generics_of(def_id);
            let self_ty = trait_pred.self_ty();
            // This is also included through the generics list as `Self`,
            // but the parser won't allow you to use it
            flags.push((sym::_Self, Some(self_ty.to_string())));
            if let Some(def) = self_ty.ty_adt_def() {
                // We also want to be able to select self's original
                // signature with no type arguments resolved
                flags.push((
                    sym::_Self,
                    Some(self.tcx.type_of(def.did()).instantiate_identity().to_string()),
                ));
            }

            for param in generics.own_params.iter() {
                let value = match param.kind {
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                        args[param.index as usize].to_string()
                    }
                    GenericParamDefKind::Lifetime => continue,
                };
                let name = param.name;
                flags.push((name, Some(value)));

                if let GenericParamDefKind::Type { .. } = param.kind {
                    let param_ty = args[param.index as usize].expect_ty();
                    if let Some(def) = param_ty.ty_adt_def() {
                        // We also want to be able to select the parameter's
                        // original signature with no type arguments resolved
                        flags.push((
                            name,
                            Some(self.tcx.type_of(def.did()).instantiate_identity().to_string()),
                        ));
                    }
                }
            }

            if let Some(true) = self_ty.ty_adt_def().map(|def| def.did().is_local()) {
                flags.push((sym::crate_local, None));
            }

            // Allow targeting all integers using `{integral}`, even if the exact type was resolved
            if self_ty.is_integral() {
                flags.push((sym::_Self, Some("{integral}".to_owned())));
            }

            if self_ty.is_array_slice() {
                flags.push((sym::_Self, Some("&[]".to_owned())));
            }

            if self_ty.is_fn() {
                let fn_sig = self_ty.fn_sig(self.tcx);
                let shortname = match fn_sig.safety() {
                    hir::Safety::Safe => "fn",
                    hir::Safety::Unsafe => "unsafe fn",
                };
                flags.push((sym::_Self, Some(shortname.to_owned())));
            }

            // Slices give us `[]`, `[{ty}]`
            if let ty::Slice(aty) = self_ty.kind() {
                flags.push((sym::_Self, Some("[]".to_string())));
                if let Some(def) = aty.ty_adt_def() {
                    // We also want to be able to select the slice's type's original
                    // signature with no type arguments resolved
                    flags.push((
                        sym::_Self,
                        Some(format!("[{}]", self.tcx.type_of(def.did()).instantiate_identity())),
                    ));
                }
                if aty.is_integral() {
                    flags.push((sym::_Self, Some("[{integral}]".to_string())));
                }
            }

            // Arrays give us `[]`, `[{ty}; _]` and `[{ty}; N]`
            if let ty::Array(aty, len) = self_ty.kind() {
                flags.push((sym::_Self, Some("[]".to_string())));
                let len = len.try_to_target_usize(self.tcx);
                flags.push((sym::_Self, Some(format!("[{aty}; _]"))));
                if let Some(n) = len {
                    flags.push((sym::_Self, Some(format!("[{aty}; {n}]"))));
                }
                if let Some(def) = aty.ty_adt_def() {
                    // We also want to be able to select the array's type's original
                    // signature with no type arguments resolved
                    let def_ty = self.tcx.type_of(def.did()).instantiate_identity();
                    flags.push((sym::_Self, Some(format!("[{def_ty}; _]"))));
                    if let Some(n) = len {
                        flags.push((sym::_Self, Some(format!("[{def_ty}; {n}]"))));
                    }
                }
                if aty.is_integral() {
                    flags.push((sym::_Self, Some("[{integral}; _]".to_string())));
                    if let Some(n) = len {
                        flags.push((sym::_Self, Some(format!("[{{integral}}; {n}]"))));
                    }
                }
            }
            if let ty::Dynamic(traits, _, _) = self_ty.kind() {
                for t in traits.iter() {
                    if let ty::ExistentialPredicate::Trait(trait_ref) = t.skip_binder() {
                        flags.push((sym::_Self, Some(self.tcx.def_path_str(trait_ref.def_id))))
                    }
                }
            }

            // `&[{integral}]` - `FromIterator` needs that.
            if let ty::Ref(_, ref_ty, rustc_ast::Mutability::Not) = self_ty.kind()
                && let ty::Slice(sty) = ref_ty.kind()
                && sty.is_integral()
            {
                flags.push((sym::_Self, Some("&[{integral}]".to_owned())));
            }
        }));

        if let Ok(Some(command)) = OnUnimplementedDirective::of_item(self.tcx, def_id) {
            command.evaluate(self.tcx, trait_pred.trait_ref, &flags, long_ty_file)
        } else {
            OnUnimplementedNote::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnUnimplementedFormatString {
    symbol: Symbol,
    span: Span,
    is_diagnostic_namespace_variant: bool,
}

#[derive(Debug)]
pub struct OnUnimplementedDirective {
    pub condition: Option<MetaItemInner>,
    pub subcommands: Vec<OnUnimplementedDirective>,
    pub message: Option<OnUnimplementedFormatString>,
    pub label: Option<OnUnimplementedFormatString>,
    pub notes: Vec<OnUnimplementedFormatString>,
    pub parent_label: Option<OnUnimplementedFormatString>,
    pub append_const_msg: Option<AppendConstMessage>,
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

/// Append a message for `~const Trait` errors.
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
    fn new(span: Span) -> Self {
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
    fn maybe_emit_warning<'tcx>(
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
#[diag(trait_selection_unknown_format_parameter_for_on_unimplemented_attr)]
#[help]
pub struct UnknownFormatParameterForOnUnimplementedAttr {
    argument_name: Symbol,
    trait_name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(trait_selection_disallowed_positional_argument)]
#[help]
pub struct DisallowedPositionalArgument;

#[derive(LintDiagnostic)]
#[diag(trait_selection_invalid_format_specifier)]
#[help]
pub struct InvalidFormatSpecifier;

#[derive(LintDiagnostic)]
#[diag(trait_selection_wrapped_parser_error)]
pub struct WrappedParserError {
    description: String,
    label: String,
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

        let parse_value = |value_str, value_span| {
            OnUnimplementedFormatString::try_parse(
                tcx,
                item_def_id,
                value_str,
                value_span,
                is_diagnostic_namespace_variant,
            )
            .map(Some)
        };

        let condition = if is_root {
            None
        } else {
            let cond = item_iter
                .next()
                .ok_or_else(|| tcx.dcx().emit_err(EmptyOnClauseInOnUnimplemented { span }))?
                .meta_item_or_bool()
                .ok_or_else(|| tcx.dcx().emit_err(InvalidOnClauseInOnUnimplemented { span }))?;
            attr::eval_condition(cond, &tcx.sess, Some(tcx.features()), &mut |cfg| {
                if let Some(value) = cfg.value
                    && let Err(guar) = parse_value(value, cfg.span)
                {
                    errored = Some(guar);
                }
                true
            });
            Some(cond.clone())
        };

        let mut message = None;
        let mut label = None;
        let mut notes = Vec::new();
        let mut parent_label = None;
        let mut subcommands = vec![];
        let mut append_const_msg = None;

        for item in item_iter {
            if item.has_name(sym::message) && message.is_none() {
                if let Some(message_) = item.value_str() {
                    message = parse_value(message_, item.span())?;
                    continue;
                }
            } else if item.has_name(sym::label) && label.is_none() {
                if let Some(label_) = item.value_str() {
                    label = parse_value(label_, item.span())?;
                    continue;
                }
            } else if item.has_name(sym::note) {
                if let Some(note_) = item.value_str() {
                    if let Some(note) = parse_value(note_, item.span())? {
                        notes.push(note);
                        continue;
                    }
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
                            directive.message.as_ref().map(|f| f.span),
                            aggr.message.as_ref().map(|f| f.span),
                            "message",
                        );
                        IgnoredDiagnosticOption::maybe_emit_warning(
                            tcx,
                            item_def_id,
                            directive.label.as_ref().map(|f| f.span),
                            aggr.label.as_ref().map(|f| f.span),
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
            Self::parse(tcx, item_def_id, &items, attr.span, true, is_diagnostic_namespace_variant)
        } else if let Some(value) = attr.value_str() {
            if !is_diagnostic_namespace_variant {
                Ok(Some(OnUnimplementedDirective {
                    condition: None,
                    message: None,
                    subcommands: vec![],
                    label: Some(OnUnimplementedFormatString::try_parse(
                        tcx,
                        item_def_id,
                        value,
                        attr.span,
                        is_diagnostic_namespace_variant,
                    )?),
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
            match &attr.kind {
                AttrKind::Normal(p) if !matches!(p.args, AttrArgs::Empty) => {
                    if let Some(item_def_id) = item_def_id.as_local() {
                        tcx.emit_node_span_lint(
                            UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                            tcx.local_def_id_to_hir_id(item_def_id),
                            attr.span,
                            MalformedOnUnimplementedAttrLint::new(attr.span),
                        );
                    }
                }
                _ => {
                    if let Some(item_def_id) = item_def_id.as_local() {
                        tcx.emit_node_span_lint(
                            UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                            tcx.local_def_id_to_hir_id(item_def_id),
                            attr.span,
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

    pub fn evaluate(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
        options: &[(Symbol, Option<String>)],
        long_ty_file: &mut Option<PathBuf>,
    ) -> OnUnimplementedNote {
        let mut message = None;
        let mut label = None;
        let mut notes = Vec::new();
        let mut parent_label = None;
        let mut append_const_msg = None;
        info!("evaluate({:?}, trait_ref={:?}, options={:?})", self, trait_ref, options);

        let options_map: FxHashMap<Symbol, String> =
            options.iter().filter_map(|(k, v)| v.clone().map(|v| (*k, v))).collect();

        for command in self.subcommands.iter().chain(Some(self)).rev() {
            debug!(?command);
            if let Some(ref condition) = command.condition
                && !attr::eval_condition(condition, &tcx.sess, Some(tcx.features()), &mut |cfg| {
                    let value = cfg.value.map(|v| {
                        // `with_no_visible_paths` is also used when generating the options,
                        // so we need to match it here.
                        ty::print::with_no_visible_paths!(
                            OnUnimplementedFormatString {
                                symbol: v,
                                span: cfg.span,
                                is_diagnostic_namespace_variant: false
                            }
                            .format(
                                tcx,
                                trait_ref,
                                &options_map,
                                long_ty_file
                            )
                        )
                    });

                    options.contains(&(cfg.name, value))
                })
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
            label: label.map(|l| l.format(tcx, trait_ref, &options_map, long_ty_file)),
            message: message.map(|m| m.format(tcx, trait_ref, &options_map, long_ty_file)),
            notes: notes
                .into_iter()
                .map(|n| n.format(tcx, trait_ref, &options_map, long_ty_file))
                .collect(),
            parent_label: parent_label
                .map(|e_s| e_s.format(tcx, trait_ref, &options_map, long_ty_file)),
            append_const_msg,
        }
    }
}

impl<'tcx> OnUnimplementedFormatString {
    fn try_parse(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        from: Symbol,
        value_span: Span,
        is_diagnostic_namespace_variant: bool,
    ) -> Result<Self, ErrorGuaranteed> {
        let result = OnUnimplementedFormatString {
            symbol: from,
            span: value_span,
            is_diagnostic_namespace_variant,
        };
        result.verify(tcx, item_def_id)?;
        Ok(result)
    }

    fn verify(&self, tcx: TyCtxt<'tcx>, item_def_id: DefId) -> Result<(), ErrorGuaranteed> {
        let trait_def_id = if tcx.is_trait(item_def_id) {
            item_def_id
        } else {
            tcx.trait_id_of_impl(item_def_id)
                .expect("expected `on_unimplemented` to correspond to a trait")
        };
        let trait_name = tcx.item_name(trait_def_id);
        let generics = tcx.generics_of(item_def_id);
        let s = self.symbol.as_str();
        let mut parser = Parser::new(s, None, None, false, ParseMode::Format);
        let mut result = Ok(());
        for token in &mut parser {
            match token {
                Piece::String(_) => (), // Normal string, no need to check it
                Piece::NextArgument(a) => {
                    let format_spec = a.format;
                    if self.is_diagnostic_namespace_variant
                        && (format_spec.ty_span.is_some()
                            || format_spec.width_span.is_some()
                            || format_spec.precision_span.is_some()
                            || format_spec.fill_span.is_some())
                    {
                        if let Some(item_def_id) = item_def_id.as_local() {
                            tcx.emit_node_span_lint(
                                UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                                tcx.local_def_id_to_hir_id(item_def_id),
                                self.span,
                                InvalidFormatSpecifier,
                            );
                        }
                    }
                    match a.position {
                        Position::ArgumentNamed(s) => {
                            match Symbol::intern(s) {
                                // `{ThisTraitsName}` is allowed
                                s if s == trait_name && !self.is_diagnostic_namespace_variant => (),
                                s if ALLOWED_FORMAT_SYMBOLS.contains(&s)
                                    && !self.is_diagnostic_namespace_variant =>
                                {
                                    ()
                                }
                                // So is `{A}` if A is a type parameter
                                s if generics.own_params.iter().any(|param| param.name == s) => (),
                                s => {
                                    if self.is_diagnostic_namespace_variant {
                                        if let Some(item_def_id) = item_def_id.as_local() {
                                            tcx.emit_node_span_lint(
                                                UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                                                tcx.local_def_id_to_hir_id(item_def_id),
                                                self.span,
                                                UnknownFormatParameterForOnUnimplementedAttr {
                                                    argument_name: s,
                                                    trait_name,
                                                },
                                            );
                                        }
                                    } else {
                                        result = Err(struct_span_code_err!(
                                            tcx.dcx(),
                                            self.span,
                                            E0230,
                                            "there is no parameter `{}` on {}",
                                            s,
                                            if trait_def_id == item_def_id {
                                                format!("trait `{trait_name}`")
                                            } else {
                                                "impl".to_string()
                                            }
                                        )
                                        .emit());
                                    }
                                }
                            }
                        }
                        // `{:1}` and `{}` are not to be used
                        Position::ArgumentIs(..) | Position::ArgumentImplicitlyIs(_) => {
                            if self.is_diagnostic_namespace_variant {
                                if let Some(item_def_id) = item_def_id.as_local() {
                                    tcx.emit_node_span_lint(
                                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                                        tcx.local_def_id_to_hir_id(item_def_id),
                                        self.span,
                                        DisallowedPositionalArgument,
                                    );
                                }
                            } else {
                                let reported = struct_span_code_err!(
                                    tcx.dcx(),
                                    self.span,
                                    E0231,
                                    "only named generic parameters are allowed"
                                )
                                .emit();
                                result = Err(reported);
                            }
                        }
                    }
                }
            }
        }
        // we cannot return errors from processing the format string as hard error here
        // as the diagnostic namespace guarantees that malformed input cannot cause an error
        //
        // if we encounter any error while processing we nevertheless want to show it as warning
        // so that users are aware that something is not correct
        for e in parser.errors {
            if self.is_diagnostic_namespace_variant {
                if let Some(item_def_id) = item_def_id.as_local() {
                    tcx.emit_node_span_lint(
                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        tcx.local_def_id_to_hir_id(item_def_id),
                        self.span,
                        WrappedParserError { description: e.description, label: e.label },
                    );
                }
            } else {
                let reported =
                    struct_span_code_err!(tcx.dcx(), self.span, E0231, "{}", e.description,).emit();
                result = Err(reported);
            }
        }

        result
    }

    pub fn format(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
        options: &FxHashMap<Symbol, String>,
        long_ty_file: &mut Option<PathBuf>,
    ) -> String {
        let name = tcx.item_name(trait_ref.def_id);
        let trait_str = tcx.def_path_str(trait_ref.def_id);
        let generics = tcx.generics_of(trait_ref.def_id);
        let generic_map = generics
            .own_params
            .iter()
            .filter_map(|param| {
                let value = match param.kind {
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                        if let Some(ty) = trait_ref.args[param.index as usize].as_type() {
                            tcx.short_ty_string(ty, long_ty_file)
                        } else {
                            trait_ref.args[param.index as usize].to_string()
                        }
                    }
                    GenericParamDefKind::Lifetime => return None,
                };
                let name = param.name;
                Some((name, value))
            })
            .collect::<FxHashMap<Symbol, String>>();
        let empty_string = String::new();

        let s = self.symbol.as_str();
        let mut parser = Parser::new(s, None, None, false, ParseMode::Format);
        let item_context = (options.get(&sym::ItemContext)).unwrap_or(&empty_string);
        let constructed_message = (&mut parser)
            .map(|p| match p {
                Piece::String(s) => s.to_owned(),
                Piece::NextArgument(a) => match a.position {
                    Position::ArgumentNamed(arg) => {
                        let s = Symbol::intern(arg);
                        match generic_map.get(&s) {
                            Some(val) => val.to_string(),
                            None if self.is_diagnostic_namespace_variant => {
                                format!("{{{arg}}}")
                            }
                            None if s == name => trait_str.clone(),
                            None => {
                                if let Some(val) = options.get(&s) {
                                    val.clone()
                                } else if s == sym::from_desugaring {
                                    // don't break messages using these two arguments incorrectly
                                    String::new()
                                } else if s == sym::ItemContext
                                    && !self.is_diagnostic_namespace_variant
                                {
                                    item_context.clone()
                                } else if s == sym::integral {
                                    String::from("{integral}")
                                } else if s == sym::integer_ {
                                    String::from("{integer}")
                                } else if s == sym::float {
                                    String::from("{float}")
                                } else {
                                    bug!(
                                        "broken on_unimplemented {:?} for {:?}: \
                                      no argument matching {:?}",
                                        self.symbol,
                                        trait_ref,
                                        s
                                    )
                                }
                            }
                        }
                    }
                    Position::ArgumentImplicitlyIs(_) if self.is_diagnostic_namespace_variant => {
                        String::from("{}")
                    }
                    Position::ArgumentIs(idx) if self.is_diagnostic_namespace_variant => {
                        format!("{{{idx}}}")
                    }
                    _ => bug!("broken on_unimplemented {:?} - bad format arg", self.symbol),
                },
            })
            .collect();
        // we cannot return errors from processing the format string as hard error here
        // as the diagnostic namespace guarantees that malformed input cannot cause an error
        //
        // if we encounter any error while processing the format string
        // we don't want to show the potentially half assembled formatted string,
        // therefore we fall back to just showing the input string in this case
        //
        // The actual parser errors are emitted earlier
        // as lint warnings in OnUnimplementedFormatString::verify
        if self.is_diagnostic_namespace_variant && !parser.errors.is_empty() {
            String::from(s)
        } else {
            constructed_message
        }
    }
}

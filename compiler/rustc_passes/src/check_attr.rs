//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use std::cell::Cell;
use std::collections::hash_map::Entry;

use rustc_ast::{
    AttrKind, AttrStyle, Attribute, LitKind, MetaItemInner, MetaItemKind, MetaItemLit, ast,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, DiagCtxtHandle, IntoDiagArg, MultiSpan, StashKey};
use rustc_feature::{AttributeDuplicates, AttributeType, BUILTIN_ATTRIBUTE_MAP, BuiltinAttribute};
use rustc_hir::def_id::LocalModDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{
    self as hir, self, AssocItemKind, CRATE_HIR_ID, CRATE_OWNER_ID, FnSig, ForeignItem, HirId,
    Item, ItemKind, MethodKind, Safety, Target, TraitItem,
};
use rustc_macros::LintDiagnostic;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault;
use rustc_middle::query::Providers;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, TyCtxt, TypingMode};
use rustc_middle::{bug, span_bug};
use rustc_session::lint::builtin::{
    CONFLICTING_REPR_HINTS, INVALID_DOC_ATTRIBUTES, INVALID_MACRO_EXPORT_ARGUMENTS,
    UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES, UNUSED_ATTRIBUTES,
};
use rustc_session::parse::feature_err;
use rustc_span::symbol::{Symbol, kw, sym};
use rustc_span::{BytePos, DUMMY_SP, Span};
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::infer::{TyCtxtInferExt, ValuePairs};
use rustc_trait_selection::traits::ObligationCtxt;
use tracing::debug;

use crate::{errors, fluent_generated as fluent};

#[derive(LintDiagnostic)]
#[diag(passes_diagnostic_diagnostic_on_unimplemented_only_for_traits)]
struct DiagnosticOnUnimplementedOnlyForTraits;

fn target_from_impl_item<'tcx>(tcx: TyCtxt<'tcx>, impl_item: &hir::ImplItem<'_>) -> Target {
    match impl_item.kind {
        hir::ImplItemKind::Const(..) => Target::AssocConst,
        hir::ImplItemKind::Fn(..) => {
            let parent_def_id = tcx.hir().get_parent_item(impl_item.hir_id()).def_id;
            let containing_item = tcx.hir().expect_item(parent_def_id);
            let containing_impl_is_for_trait = match &containing_item.kind {
                hir::ItemKind::Impl(impl_) => impl_.of_trait.is_some(),
                _ => bug!("parent of an ImplItem must be an Impl"),
            };
            if containing_impl_is_for_trait {
                Target::Method(MethodKind::Trait { body: true })
            } else {
                Target::Method(MethodKind::Inherent)
            }
        }
        hir::ImplItemKind::Type(..) => Target::AssocTy,
    }
}

#[derive(Clone, Copy)]
enum ItemLike<'tcx> {
    Item(&'tcx Item<'tcx>),
    ForeignItem,
}

#[derive(Copy, Clone)]
pub(crate) enum ProcMacroKind {
    FunctionLike,
    Derive,
    Attribute,
}

impl IntoDiagArg for ProcMacroKind {
    fn into_diag_arg(self) -> rustc_errors::DiagArgValue {
        match self {
            ProcMacroKind::Attribute => "attribute proc macro",
            ProcMacroKind::Derive => "derive proc macro",
            ProcMacroKind::FunctionLike => "function-like proc macro",
        }
        .into_diag_arg()
    }
}

struct CheckAttrVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    // Whether or not this visitor should abort after finding errors
    abort: Cell<bool>,
}

impl<'tcx> CheckAttrVisitor<'tcx> {
    fn dcx(&self) -> DiagCtxtHandle<'tcx> {
        self.tcx.dcx()
    }

    /// Checks any attribute.
    fn check_attributes(
        &self,
        hir_id: HirId,
        span: Span,
        target: Target,
        item: Option<ItemLike<'_>>,
    ) {
        let mut doc_aliases = FxHashMap::default();
        let mut specified_inline = None;
        let mut seen = FxHashMap::default();
        let attrs = self.tcx.hir().attrs(hir_id);
        for attr in attrs {
            match attr.path().as_slice() {
                [sym::diagnostic, sym::do_not_recommend, ..] => {
                    self.check_do_not_recommend(attr.span, hir_id, target)
                }
                [sym::diagnostic, sym::on_unimplemented, ..] => {
                    self.check_diagnostic_on_unimplemented(attr.span, hir_id, target)
                }
                [sym::inline, ..] => self.check_inline(hir_id, attr, span, target),
                [sym::coverage, ..] => self.check_coverage(attr, span, target),
                [sym::optimize, ..] => self.check_optimize(hir_id, attr, span, target),
                [sym::no_sanitize, ..] => self.check_no_sanitize(attr, span, target),
                [sym::non_exhaustive, ..] => self.check_non_exhaustive(hir_id, attr, span, target),
                [sym::marker, ..] => self.check_marker(hir_id, attr, span, target),
                [sym::target_feature, ..] => {
                    self.check_target_feature(hir_id, attr, span, target, attrs)
                }
                [sym::thread_local, ..] => self.check_thread_local(attr, span, target),
                [sym::track_caller, ..] => {
                    self.check_track_caller(hir_id, attr.span, attrs, span, target)
                }
                [sym::doc, ..] => self.check_doc_attrs(
                    attr,
                    hir_id,
                    target,
                    &mut specified_inline,
                    &mut doc_aliases,
                ),
                [sym::no_link, ..] => self.check_no_link(hir_id, attr, span, target),
                [sym::export_name, ..] => self.check_export_name(hir_id, attr, span, target),
                [sym::rustc_layout_scalar_valid_range_start, ..]
                | [sym::rustc_layout_scalar_valid_range_end, ..] => {
                    self.check_rustc_layout_scalar_valid_range(attr, span, target)
                }
                [sym::allow_internal_unstable, ..] => {
                    self.check_allow_internal_unstable(hir_id, attr, span, target, attrs)
                }
                [sym::debugger_visualizer, ..] => self.check_debugger_visualizer(attr, target),
                [sym::rustc_allow_const_fn_unstable, ..] => {
                    self.check_rustc_allow_const_fn_unstable(hir_id, attr, span, target)
                }
                [sym::rustc_std_internal_symbol, ..] => {
                    self.check_rustc_std_internal_symbol(attr, span, target)
                }
                [sym::naked, ..] => self.check_naked(hir_id, attr, span, target, attrs),
                [sym::rustc_as_ptr, ..] => {
                    self.check_applied_to_fn_or_method(hir_id, attr, span, target)
                }
                [sym::rustc_never_returns_null_ptr, ..] => {
                    self.check_applied_to_fn_or_method(hir_id, attr, span, target)
                }
                [sym::rustc_legacy_const_generics, ..] => {
                    self.check_rustc_legacy_const_generics(hir_id, attr, span, target, item)
                }
                [sym::rustc_lint_query_instability, ..] => {
                    self.check_applied_to_fn_or_method(hir_id, attr, span, target)
                }
                [sym::rustc_lint_untracked_query_information, ..] => {
                    self.check_applied_to_fn_or_method(hir_id, attr, span, target)
                }
                [sym::rustc_lint_diagnostics, ..] => {
                    self.check_applied_to_fn_or_method(hir_id, attr, span, target)
                }
                [sym::rustc_lint_opt_ty, ..] => self.check_rustc_lint_opt_ty(attr, span, target),
                [sym::rustc_lint_opt_deny_field_access, ..] => {
                    self.check_rustc_lint_opt_deny_field_access(attr, span, target)
                }
                [sym::rustc_clean, ..]
                | [sym::rustc_dirty, ..]
                | [sym::rustc_if_this_changed, ..]
                | [sym::rustc_then_this_would_need, ..] => self.check_rustc_dirty_clean(attr),
                [sym::rustc_coinductive, ..]
                | [sym::rustc_must_implement_one_of, ..]
                | [sym::rustc_deny_explicit_impl, ..]
                | [sym::const_trait, ..] => self.check_must_be_applied_to_trait(attr, span, target),
                [sym::collapse_debuginfo, ..] => self.check_collapse_debuginfo(attr, span, target),
                [sym::must_not_suspend, ..] => self.check_must_not_suspend(attr, span, target),
                [sym::must_use, ..] => self.check_must_use(hir_id, attr, target),
                [sym::may_dangle, ..] => self.check_may_dangle(hir_id, attr),
                [sym::rustc_pass_by_value, ..] => self.check_pass_by_value(attr, span, target),
                [sym::rustc_allow_incoherent_impl, ..] => {
                    self.check_allow_incoherent_impl(attr, span, target)
                }
                [sym::rustc_has_incoherent_inherent_impls, ..] => {
                    self.check_has_incoherent_inherent_impls(attr, span, target)
                }
                [sym::ffi_pure, ..] => self.check_ffi_pure(attr.span, attrs, target),
                [sym::ffi_const, ..] => self.check_ffi_const(attr.span, target),
                [sym::rustc_const_unstable, ..]
                | [sym::rustc_const_stable, ..]
                | [sym::unstable, ..]
                | [sym::stable, ..]
                | [sym::rustc_allowed_through_unstable_modules, ..]
                | [sym::rustc_promotable, ..] => self.check_stability_promotable(attr, target),
                [sym::link_ordinal, ..] => self.check_link_ordinal(attr, span, target),
                [sym::rustc_confusables, ..] => self.check_confusables(attr, target),
                [sym::cold, ..] => self.check_cold(hir_id, attr, span, target),
                [sym::link, ..] => self.check_link(hir_id, attr, span, target),
                [sym::link_name, ..] => self.check_link_name(hir_id, attr, span, target),
                [sym::link_section, ..] => self.check_link_section(hir_id, attr, span, target),
                [sym::no_mangle, ..] => self.check_no_mangle(hir_id, attr, span, target),
                [sym::deprecated, ..] => self.check_deprecated(hir_id, attr, span, target),
                [sym::macro_use, ..] | [sym::macro_escape, ..] => {
                    self.check_macro_use(hir_id, attr, target)
                }
                [sym::path, ..] => self.check_generic_attr(hir_id, attr, target, Target::Mod),
                [sym::macro_export, ..] => self.check_macro_export(hir_id, attr, target),
                [sym::ignore, ..] | [sym::should_panic, ..] => {
                    self.check_generic_attr(hir_id, attr, target, Target::Fn)
                }
                [sym::automatically_derived, ..] => {
                    self.check_generic_attr(hir_id, attr, target, Target::Impl)
                }
                [sym::no_implicit_prelude, ..] => {
                    self.check_generic_attr(hir_id, attr, target, Target::Mod)
                }
                [sym::rustc_object_lifetime_default, ..] => self.check_object_lifetime_default(hir_id),
                [sym::proc_macro, ..] => {
                    self.check_proc_macro(hir_id, target, ProcMacroKind::FunctionLike)
                }
                [sym::proc_macro_attribute, ..] => {
                    self.check_proc_macro(hir_id, target, ProcMacroKind::Attribute);
                }
                [sym::proc_macro_derive, ..] => {
                    self.check_generic_attr(hir_id, attr, target, Target::Fn);
                    self.check_proc_macro(hir_id, target, ProcMacroKind::Derive)
                }
                [sym::autodiff, ..] => {
                    self.check_autodiff(hir_id, attr, span, target)
                }
                [sym::coroutine, ..] => {
                    self.check_coroutine(attr, target);
                }
                [sym::linkage, ..] => self.check_linkage(attr, span, target),
                [sym::rustc_pub_transparent, ..] => self.check_rustc_pub_transparent( attr.span, span, attrs),
                [
                    // ok
                    sym::allow
                    | sym::expect
                    | sym::warn
                    | sym::deny
                    | sym::forbid
                    | sym::cfg
                    | sym::cfg_attr
                    // need to be fixed
                    | sym::cfi_encoding // FIXME(cfi_encoding)
                    | sym::pointee // FIXME(derive_coerce_pointee)
                    | sym::omit_gdb_pretty_printer_section // FIXME(omit_gdb_pretty_printer_section)
                    | sym::used // handled elsewhere to restrict to static items
                    | sym::repr // handled elsewhere to restrict to type decls items
                    | sym::instruction_set // broken on stable!!!
                    | sym::windows_subsystem // broken on stable!!!
                    | sym::patchable_function_entry // FIXME(patchable_function_entry)
                    | sym::deprecated_safe // FIXME(deprecated_safe)
                    // internal
                    | sym::prelude_import
                    | sym::panic_handler
                    | sym::allow_internal_unsafe
                    | sym::fundamental
                    | sym::lang
                    | sym::needs_allocator
                    | sym::default_lib_allocator
                    | sym::start
                    | sym::custom_mir,
                    ..
                ] => {}
                [name, ..] => {
                    match BUILTIN_ATTRIBUTE_MAP.get(name) {
                        // checked below
                        Some(BuiltinAttribute { type_: AttributeType::CrateLevel, .. }) => {}
                        Some(_) => {
                            // FIXME: differentiate between unstable and internal attributes just
                            // like we do with features instead of just accepting `rustc_`
                            // attributes by name. That should allow trimming the above list, too.
                            if !name.as_str().starts_with("rustc_") {
                                span_bug!(
                                    attr.span,
                                    "builtin attribute {name:?} not handled by `CheckAttrVisitor`"
                                )
                            }
                        }
                        None => (),
                    }
                }
                [] => unreachable!(),
            }

            let builtin = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));

            if hir_id != CRATE_HIR_ID {
                if let Some(BuiltinAttribute { type_: AttributeType::CrateLevel, .. }) =
                    attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name))
                {
                    match attr.style {
                        ast::AttrStyle::Outer => self.tcx.emit_node_span_lint(
                            UNUSED_ATTRIBUTES,
                            hir_id,
                            attr.span,
                            errors::OuterCrateLevelAttr,
                        ),
                        ast::AttrStyle::Inner => self.tcx.emit_node_span_lint(
                            UNUSED_ATTRIBUTES,
                            hir_id,
                            attr.span,
                            errors::InnerCrateLevelAttr,
                        ),
                    }
                }
            }

            if let Some(BuiltinAttribute { duplicates, .. }) = builtin {
                check_duplicates(self.tcx, attr, hir_id, *duplicates, &mut seen);
            }

            self.check_unused_attribute(hir_id, attr)
        }

        self.check_repr(attrs, span, target, item, hir_id);
        self.check_used(attrs, target, span);
    }

    fn inline_attr_str_error_with_macro_def(&self, hir_id: HirId, attr: &Attribute, sym: &str) {
        self.tcx.emit_node_span_lint(
            UNUSED_ATTRIBUTES,
            hir_id,
            attr.span,
            errors::IgnoredAttrWithMacro { sym },
        );
    }

    fn inline_attr_str_error_without_macro_def(&self, hir_id: HirId, attr: &Attribute, sym: &str) {
        self.tcx
            .emit_node_span_lint(UNUSED_ATTRIBUTES, hir_id, attr.span, errors::IgnoredAttr { sym });
    }

    /// Checks if `#[diagnostic::do_not_recommend]` is applied on a trait impl.
    fn check_do_not_recommend(&self, attr_span: Span, hir_id: HirId, target: Target) {
        if !matches!(target, Target::Impl) {
            self.tcx.emit_node_span_lint(
                UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                hir_id,
                attr_span,
                errors::IncorrectDoNotRecommendLocation,
            );
        }
    }

    /// Checks if `#[diagnostic::on_unimplemented]` is applied to a trait definition
    fn check_diagnostic_on_unimplemented(&self, attr_span: Span, hir_id: HirId, target: Target) {
        if !matches!(target, Target::Trait) {
            self.tcx.emit_node_span_lint(
                UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                hir_id,
                attr_span,
                DiagnosticOnUnimplementedOnlyForTraits,
            );
        }
    }

    /// Checks if an `#[inline]` is applied to a function or a closure.
    fn check_inline(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Fn
            | Target::Closure
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => {}
            Target::Method(MethodKind::Trait { body: false }) | Target::ForeignFn => {
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::IgnoredInlineAttrFnProto,
                )
            }
            // FIXME(#65833): We permit associated consts to have an `#[inline]` attribute with
            // just a lint, because we previously erroneously allowed it and some crates used it
            // accidentally, to be compatible with crates depending on them, we can't throw an
            // error here.
            Target::AssocConst => self.tcx.emit_node_span_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr.span,
                errors::IgnoredInlineAttrConstants,
            ),
            // FIXME(#80564): Same for fields, arms, and macro defs
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "inline")
            }
            _ => {
                self.dcx().emit_err(errors::InlineNotFnOrClosure {
                    attr_span: attr.span,
                    defn_span: span,
                });
            }
        }
    }

    /// Checks that `#[coverage(..)]` is applied to a function/closure/method,
    /// or to an impl block or module.
    fn check_coverage(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Fn
            | Target::Closure
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent)
            | Target::Impl
            | Target::Mod => {}

            _ => {
                self.dcx().emit_err(errors::CoverageNotFnOrClosure {
                    attr_span: attr.span,
                    defn_span: span,
                });
            }
        }
    }

    /// Checks that `#[optimize(..)]` is applied to a function/closure/method,
    /// or to an impl block or module.
    fn check_optimize(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        let is_valid = matches!(
            target,
            Target::Fn
                | Target::Closure
                | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent)
        );
        if !is_valid {
            self.dcx().emit_err(errors::OptimizeInvalidTarget {
                attr_span: attr.span,
                defn_span: span,
                on_crate: hir_id == CRATE_HIR_ID,
            });
        }
    }

    fn check_no_sanitize(&self, attr: &Attribute, span: Span, target: Target) {
        if let Some(list) = attr.meta_item_list() {
            for item in list.iter() {
                let sym = item.name_or_empty();
                match sym {
                    sym::address | sym::hwaddress => {
                        let is_valid =
                            matches!(target, Target::Fn | Target::Method(..) | Target::Static);
                        if !is_valid {
                            self.dcx().emit_err(errors::NoSanitize {
                                attr_span: item.span(),
                                defn_span: span,
                                accepted_kind: "a function or static",
                                attr_str: sym.as_str(),
                            });
                        }
                    }
                    _ => {
                        let is_valid = matches!(target, Target::Fn | Target::Method(..));
                        if !is_valid {
                            self.dcx().emit_err(errors::NoSanitize {
                                attr_span: item.span(),
                                defn_span: span,
                                accepted_kind: "a function",
                                attr_str: sym.as_str(),
                            });
                        }
                    }
                }
            }
        }
    }

    fn check_generic_attr(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        target: Target,
        allowed_target: Target,
    ) {
        if target != allowed_target {
            self.tcx.emit_node_span_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr.span,
                errors::OnlyHasEffectOn {
                    attr_name: attr.name_or_empty(),
                    target_name: allowed_target.name().replace(' ', "_"),
                },
            );
        }
    }

    /// Checks if `#[naked]` is applied to a function definition.
    fn check_naked(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
        attrs: &[Attribute],
    ) {
        // many attributes don't make sense in combination with #[naked].
        // Notable attributes that are incompatible with `#[naked]` are:
        //
        // * `#[inline]`
        // * `#[track_caller]`
        // * `#[test]`, `#[ignore]`, `#[should_panic]`
        //
        // NOTE: when making changes to this list, check that `error_codes/E0736.md` remains accurate
        const ALLOW_LIST: &[rustc_span::Symbol] = &[
            // conditional compilation
            sym::cfg,
            sym::cfg_attr,
            // testing (allowed here so better errors can be generated in `rustc_builtin_macros::test`)
            sym::test,
            sym::ignore,
            sym::should_panic,
            sym::bench,
            // diagnostics
            sym::allow,
            sym::warn,
            sym::deny,
            sym::forbid,
            sym::deprecated,
            sym::must_use,
            // abi, linking and FFI
            sym::export_name,
            sym::link_section,
            sym::linkage,
            sym::no_mangle,
            sym::naked,
            sym::instruction_set,
            sym::repr,
            // code generation
            sym::cold,
            sym::target_feature,
            // documentation
            sym::doc,
        ];

        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => {
                for other_attr in attrs {
                    // this covers "sugared doc comments" of the form `/// ...`
                    // it does not cover `#[doc = "..."]`, which is handled below
                    if other_attr.is_doc_comment() {
                        continue;
                    }

                    if !ALLOW_LIST.iter().any(|name| other_attr.has_name(*name)) {
                        self.dcx().emit_err(errors::NakedFunctionIncompatibleAttribute {
                            span: other_attr.span,
                            naked_span: attr.span,
                            attr: other_attr.name_or_empty(),
                        });

                        return;
                    }
                }
            }
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[naked]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "naked")
            }
            _ => {
                self.dcx().emit_err(errors::AttrShouldBeAppliedToFn {
                    attr_span: attr.span,
                    defn_span: span,
                    on_crate: hir_id == CRATE_HIR_ID,
                });
            }
        }
    }

    /// Debugging aid for `object_lifetime_default` query.
    fn check_object_lifetime_default(&self, hir_id: HirId) {
        let tcx = self.tcx;
        if let Some(owner_id) = hir_id.as_owner()
            && let Some(generics) = tcx.hir().get_generics(owner_id.def_id)
        {
            for p in generics.params {
                let hir::GenericParamKind::Type { .. } = p.kind else { continue };
                let default = tcx.object_lifetime_default(p.def_id);
                let repr = match default {
                    ObjectLifetimeDefault::Empty => "BaseDefault".to_owned(),
                    ObjectLifetimeDefault::Static => "'static".to_owned(),
                    ObjectLifetimeDefault::Param(def_id) => tcx.item_name(def_id).to_string(),
                    ObjectLifetimeDefault::Ambiguous => "Ambiguous".to_owned(),
                };
                tcx.dcx().emit_err(errors::ObjectLifetimeErr { span: p.span, repr });
            }
        }
    }

    /// Checks if `#[collapse_debuginfo]` is applied to a macro.
    fn check_collapse_debuginfo(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::MacroDef => {}
            _ => {
                self.tcx
                    .dcx()
                    .emit_err(errors::CollapseDebuginfo { attr_span: attr.span, defn_span: span });
            }
        }
    }

    /// Checks if a `#[track_caller]` is applied to a function.
    fn check_track_caller(
        &self,
        hir_id: HirId,
        attr_span: Span,
        attrs: &[Attribute],
        span: Span,
        target: Target,
    ) {
        match target {
            Target::Fn => {
                // `#[track_caller]` is not valid on weak lang items because they are called via
                // `extern` declarations and `#[track_caller]` would alter their ABI.
                if let Some((lang_item, _)) = hir::lang_items::extract(attrs)
                    && let Some(item) = hir::LangItem::from_name(lang_item)
                    && item.is_weak()
                {
                    let sig = self.tcx.hir_node(hir_id).fn_sig().unwrap();

                    self.dcx().emit_err(errors::LangItemWithTrackCaller {
                        attr_span,
                        name: lang_item,
                        sig_span: sig.span,
                    });
                }
            }
            Target::Method(..) | Target::ForeignFn | Target::Closure => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[track_caller]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                for attr in attrs {
                    self.inline_attr_str_error_with_macro_def(hir_id, attr, "track_caller");
                }
            }
            _ => {
                self.dcx().emit_err(errors::TrackedCallerWrongLocation {
                    attr_span,
                    defn_span: span,
                    on_crate: hir_id == CRATE_HIR_ID,
                });
            }
        }
    }

    /// Checks if the `#[non_exhaustive]` attribute on an `item` is valid.
    fn check_non_exhaustive(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Struct | Target::Enum | Target::Variant => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[non_exhaustive]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "non_exhaustive");
            }
            _ => {
                self.dcx().emit_err(errors::NonExhaustiveWrongLocation {
                    attr_span: attr.span,
                    defn_span: span,
                });
            }
        }
    }

    /// Checks if the `#[marker]` attribute on an `item` is valid.
    fn check_marker(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Trait => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[marker]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "marker");
            }
            _ => {
                self.dcx().emit_err(errors::AttrShouldBeAppliedToTrait {
                    attr_span: attr.span,
                    defn_span: span,
                });
            }
        }
    }

    /// Checks if the `#[target_feature]` attribute on `item` is valid.
    fn check_target_feature(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
        attrs: &[Attribute],
    ) {
        match target {
            Target::Fn => {
                // `#[target_feature]` is not allowed in lang items.
                if let Some((lang_item, _)) = hir::lang_items::extract(attrs)
                    // Calling functions with `#[target_feature]` is
                    // not unsafe on WASM, see #84988
                    && !self.tcx.sess.target.is_like_wasm
                    && !self.tcx.sess.opts.actually_rustdoc
                {
                    let sig = self.tcx.hir_node(hir_id).fn_sig().unwrap();

                    self.dcx().emit_err(errors::LangItemWithTargetFeature {
                        attr_span: attr.span,
                        name: lang_item,
                        sig_span: sig.span,
                    });
                }
            }
            Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => {}
            // FIXME: #[target_feature] was previously erroneously allowed on statements and some
            // crates used this, so only emit a warning.
            Target::Statement => {
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::TargetFeatureOnStatement,
                );
            }
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[target_feature]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "target_feature");
            }
            _ => {
                self.dcx().emit_err(errors::AttrShouldBeAppliedToFn {
                    attr_span: attr.span,
                    defn_span: span,
                    on_crate: hir_id == CRATE_HIR_ID,
                });
            }
        }
    }

    /// Checks if the `#[thread_local]` attribute on `item` is valid.
    fn check_thread_local(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::ForeignStatic | Target::Static => {}
            _ => {
                self.dcx().emit_err(errors::AttrShouldBeAppliedToStatic {
                    attr_span: attr.span,
                    defn_span: span,
                });
            }
        }
    }

    fn doc_attr_str_error(&self, meta: &MetaItemInner, attr_name: &str) {
        self.dcx().emit_err(errors::DocExpectStr { attr_span: meta.span(), attr_name });
    }

    fn check_doc_alias_value(
        &self,
        meta: &MetaItemInner,
        doc_alias: Symbol,
        hir_id: HirId,
        target: Target,
        is_list: bool,
        aliases: &mut FxHashMap<String, Span>,
    ) {
        let tcx = self.tcx;
        let span = meta.name_value_literal_span().unwrap_or_else(|| meta.span());
        let attr_str =
            &format!("`#[doc(alias{})]`", if is_list { "(\"...\")" } else { " = \"...\"" });
        if doc_alias == kw::Empty {
            tcx.dcx().emit_err(errors::DocAliasEmpty { span, attr_str });
            return;
        }

        let doc_alias_str = doc_alias.as_str();
        if let Some(c) = doc_alias_str
            .chars()
            .find(|&c| c == '"' || c == '\'' || (c.is_whitespace() && c != ' '))
        {
            tcx.dcx().emit_err(errors::DocAliasBadChar { span, attr_str, char_: c });
            return;
        }
        if doc_alias_str.starts_with(' ') || doc_alias_str.ends_with(' ') {
            tcx.dcx().emit_err(errors::DocAliasStartEnd { span, attr_str });
            return;
        }

        let span = meta.span();
        if let Some(location) = match target {
            Target::AssocTy => {
                let parent_def_id = self.tcx.hir().get_parent_item(hir_id).def_id;
                let containing_item = self.tcx.hir().expect_item(parent_def_id);
                if Target::from_item(containing_item) == Target::Impl {
                    Some("type alias in implementation block")
                } else {
                    None
                }
            }
            Target::AssocConst => {
                let parent_def_id = self.tcx.hir().get_parent_item(hir_id).def_id;
                let containing_item = self.tcx.hir().expect_item(parent_def_id);
                // We can't link to trait impl's consts.
                let err = "associated constant in trait implementation block";
                match containing_item.kind {
                    ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) => Some(err),
                    _ => None,
                }
            }
            // we check the validity of params elsewhere
            Target::Param => return,
            Target::Expression
            | Target::Statement
            | Target::Arm
            | Target::ForeignMod
            | Target::Closure
            | Target::Impl => Some(target.name()),
            Target::ExternCrate
            | Target::Use
            | Target::Static
            | Target::Const
            | Target::Fn
            | Target::Mod
            | Target::GlobalAsm
            | Target::TyAlias
            | Target::Enum
            | Target::Variant
            | Target::Struct
            | Target::Field
            | Target::Union
            | Target::Trait
            | Target::TraitAlias
            | Target::Method(..)
            | Target::ForeignFn
            | Target::ForeignStatic
            | Target::ForeignTy
            | Target::GenericParam(..)
            | Target::MacroDef
            | Target::PatField
            | Target::ExprField => None,
        } {
            tcx.dcx().emit_err(errors::DocAliasBadLocation { span, attr_str, location });
            return;
        }
        let item_name = self.tcx.hir().name(hir_id);
        if item_name == doc_alias {
            tcx.dcx().emit_err(errors::DocAliasNotAnAlias { span, attr_str });
            return;
        }
        if let Err(entry) = aliases.try_insert(doc_alias_str.to_owned(), span) {
            self.tcx.emit_node_span_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                span,
                errors::DocAliasDuplicated { first_defn: *entry.entry.get() },
            );
        }
    }

    fn check_doc_alias(
        &self,
        meta: &MetaItemInner,
        hir_id: HirId,
        target: Target,
        aliases: &mut FxHashMap<String, Span>,
    ) {
        if let Some(values) = meta.meta_item_list() {
            for v in values {
                match v.lit() {
                    Some(l) => match l.kind {
                        LitKind::Str(s, _) => {
                            self.check_doc_alias_value(v, s, hir_id, target, true, aliases);
                        }
                        _ => {
                            self.tcx
                                .dcx()
                                .emit_err(errors::DocAliasNotStringLiteral { span: v.span() });
                        }
                    },
                    None => {
                        self.tcx
                            .dcx()
                            .emit_err(errors::DocAliasNotStringLiteral { span: v.span() });
                    }
                }
            }
        } else if let Some(doc_alias) = meta.value_str() {
            self.check_doc_alias_value(meta, doc_alias, hir_id, target, false, aliases)
        } else {
            self.dcx().emit_err(errors::DocAliasMalformed { span: meta.span() });
        }
    }

    fn check_doc_keyword(&self, meta: &MetaItemInner, hir_id: HirId) {
        let doc_keyword = meta.value_str().unwrap_or(kw::Empty);
        if doc_keyword == kw::Empty {
            self.doc_attr_str_error(meta, "keyword");
            return;
        }
        let item_kind = match self.tcx.hir_node(hir_id) {
            hir::Node::Item(item) => Some(&item.kind),
            _ => None,
        };
        match item_kind {
            Some(ItemKind::Mod(module)) => {
                if !module.item_ids.is_empty() {
                    self.dcx().emit_err(errors::DocKeywordEmptyMod { span: meta.span() });
                    return;
                }
            }
            _ => {
                self.dcx().emit_err(errors::DocKeywordNotMod { span: meta.span() });
                return;
            }
        }
        if !rustc_lexer::is_ident(doc_keyword.as_str()) {
            self.dcx().emit_err(errors::DocKeywordInvalidIdent {
                span: meta.name_value_literal_span().unwrap_or_else(|| meta.span()),
                doc_keyword,
            });
        }
    }

    fn check_doc_fake_variadic(&self, meta: &MetaItemInner, hir_id: HirId) {
        let item_kind = match self.tcx.hir_node(hir_id) {
            hir::Node::Item(item) => Some(&item.kind),
            _ => None,
        };
        match item_kind {
            Some(ItemKind::Impl(i)) => {
                let is_valid = doc_fake_variadic_is_allowed_self_ty(i.self_ty)
                    || if let Some(&[hir::GenericArg::Type(ty)]) = i
                        .of_trait
                        .as_ref()
                        .and_then(|trait_ref| trait_ref.path.segments.last())
                        .map(|last_segment| last_segment.args().args)
                    {
                        matches!(&ty.kind, hir::TyKind::Tup([_]))
                    } else {
                        false
                    };
                if !is_valid {
                    self.dcx().emit_err(errors::DocFakeVariadicNotValid { span: meta.span() });
                }
            }
            _ => {
                self.dcx().emit_err(errors::DocKeywordOnlyImpl { span: meta.span() });
            }
        }
    }

    fn check_doc_search_unbox(&self, meta: &MetaItemInner, hir_id: HirId) {
        let hir::Node::Item(item) = self.tcx.hir_node(hir_id) else {
            self.dcx().emit_err(errors::DocSearchUnboxInvalid { span: meta.span() });
            return;
        };
        match item.kind {
            ItemKind::Enum(_, generics) | ItemKind::Struct(_, generics)
                if generics.params.len() != 0 => {}
            ItemKind::Trait(_, _, generics, _, items)
                if generics.params.len() != 0
                    || items.iter().any(|item| matches!(item.kind, AssocItemKind::Type)) => {}
            _ => {
                self.dcx().emit_err(errors::DocSearchUnboxInvalid { span: meta.span() });
            }
        }
    }

    /// Checks `#[doc(inline)]`/`#[doc(no_inline)]` attributes.
    ///
    /// A doc inlining attribute is invalid if it is applied to a non-`use` item, or
    /// if there are conflicting attributes for one item.
    ///
    /// `specified_inline` is used to keep track of whether we have
    /// already seen an inlining attribute for this item.
    /// If so, `specified_inline` holds the value and the span of
    /// the first `inline`/`no_inline` attribute.
    fn check_doc_inline(
        &self,
        attr: &Attribute,
        meta: &MetaItemInner,
        hir_id: HirId,
        target: Target,
        specified_inline: &mut Option<(bool, Span)>,
    ) {
        match target {
            Target::Use | Target::ExternCrate => {
                let do_inline = meta.name_or_empty() == sym::inline;
                if let Some((prev_inline, prev_span)) = *specified_inline {
                    if do_inline != prev_inline {
                        let mut spans = MultiSpan::from_spans(vec![prev_span, meta.span()]);
                        spans.push_span_label(prev_span, fluent::passes_doc_inline_conflict_first);
                        spans.push_span_label(
                            meta.span(),
                            fluent::passes_doc_inline_conflict_second,
                        );
                        self.dcx().emit_err(errors::DocKeywordConflict { spans });
                    }
                } else {
                    *specified_inline = Some((do_inline, meta.span()));
                }
            }
            _ => {
                self.tcx.emit_node_span_lint(
                    INVALID_DOC_ATTRIBUTES,
                    hir_id,
                    meta.span(),
                    errors::DocInlineOnlyUse {
                        attr_span: meta.span(),
                        item_span: (attr.style == AttrStyle::Outer)
                            .then(|| self.tcx.hir().span(hir_id)),
                    },
                );
            }
        }
    }

    fn check_doc_masked(
        &self,
        attr: &Attribute,
        meta: &MetaItemInner,
        hir_id: HirId,
        target: Target,
    ) {
        if target != Target::ExternCrate {
            self.tcx.emit_node_span_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                errors::DocMaskedOnlyExternCrate {
                    attr_span: meta.span(),
                    item_span: (attr.style == AttrStyle::Outer)
                        .then(|| self.tcx.hir().span(hir_id)),
                },
            );
            return;
        }

        if self.tcx.extern_mod_stmt_cnum(hir_id.owner).is_none() {
            self.tcx.emit_node_span_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                errors::DocMaskedNotExternCrateSelf {
                    attr_span: meta.span(),
                    item_span: (attr.style == AttrStyle::Outer)
                        .then(|| self.tcx.hir().span(hir_id)),
                },
            );
        }
    }

    /// Checks that an attribute is *not* used at the crate level. Returns `true` if valid.
    fn check_attr_not_crate_level(
        &self,
        meta: &MetaItemInner,
        hir_id: HirId,
        attr_name: &str,
    ) -> bool {
        if CRATE_HIR_ID == hir_id {
            self.dcx().emit_err(errors::DocAttrNotCrateLevel { span: meta.span(), attr_name });
            return false;
        }
        true
    }

    /// Checks that an attribute is used at the crate level. Returns `true` if valid.
    fn check_attr_crate_level(
        &self,
        attr: &Attribute,
        meta: &MetaItemInner,
        hir_id: HirId,
    ) -> bool {
        if hir_id != CRATE_HIR_ID {
            // insert a bang between `#` and `[...`
            let bang_span = attr.span.lo() + BytePos(1);
            let sugg = (attr.style == AttrStyle::Outer
                && self.tcx.hir().get_parent_item(hir_id) == CRATE_OWNER_ID)
                .then_some(errors::AttrCrateLevelOnlySugg {
                    attr: attr.span.with_lo(bang_span).with_hi(bang_span),
                });
            self.tcx.emit_node_span_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                errors::AttrCrateLevelOnly { sugg },
            );
            return false;
        }
        true
    }

    /// Checks that `doc(test(...))` attribute contains only valid attributes. Returns `true` if
    /// valid.
    fn check_test_attr(&self, meta: &MetaItemInner, hir_id: HirId) {
        if let Some(metas) = meta.meta_item_list() {
            for i_meta in metas {
                match (i_meta.name_or_empty(), i_meta.meta_item()) {
                    (sym::attr | sym::no_crate_inject, _) => {}
                    (_, Some(m)) => {
                        self.tcx.emit_node_span_lint(
                            INVALID_DOC_ATTRIBUTES,
                            hir_id,
                            i_meta.span(),
                            errors::DocTestUnknown {
                                path: rustc_ast_pretty::pprust::path_to_string(&m.path),
                            },
                        );
                    }
                    (_, None) => {
                        self.tcx.emit_node_span_lint(
                            INVALID_DOC_ATTRIBUTES,
                            hir_id,
                            i_meta.span(),
                            errors::DocTestLiteral,
                        );
                    }
                }
            }
        } else {
            self.tcx.emit_node_span_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                errors::DocTestTakesList,
            );
        }
    }

    /// Check that the `#![doc(cfg_hide(...))]` attribute only contains a list of attributes.
    ///
    fn check_doc_cfg_hide(&self, meta: &MetaItemInner, hir_id: HirId) {
        if meta.meta_item_list().is_none() {
            self.tcx.emit_node_span_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                errors::DocCfgHideTakesList,
            );
        }
    }

    /// Runs various checks on `#[doc]` attributes.
    ///
    /// `specified_inline` should be initialized to `None` and kept for the scope
    /// of one item. Read the documentation of [`check_doc_inline`] for more information.
    ///
    /// [`check_doc_inline`]: Self::check_doc_inline
    fn check_doc_attrs(
        &self,
        attr: &Attribute,
        hir_id: HirId,
        target: Target,
        specified_inline: &mut Option<(bool, Span)>,
        aliases: &mut FxHashMap<String, Span>,
    ) {
        if let Some(mi) = attr.meta()
            && let Some(list) = mi.meta_item_list()
        {
            for meta in list {
                if let Some(i_meta) = meta.meta_item() {
                    match i_meta.name_or_empty() {
                        sym::alias => {
                            if self.check_attr_not_crate_level(meta, hir_id, "alias") {
                                self.check_doc_alias(meta, hir_id, target, aliases);
                            }
                        }

                        sym::keyword => {
                            if self.check_attr_not_crate_level(meta, hir_id, "keyword") {
                                self.check_doc_keyword(meta, hir_id);
                            }
                        }

                        sym::fake_variadic => {
                            if self.check_attr_not_crate_level(meta, hir_id, "fake_variadic") {
                                self.check_doc_fake_variadic(meta, hir_id);
                            }
                        }

                        sym::search_unbox => {
                            if self.check_attr_not_crate_level(meta, hir_id, "fake_variadic") {
                                self.check_doc_search_unbox(meta, hir_id);
                            }
                        }

                        sym::test => {
                            if self.check_attr_crate_level(attr, meta, hir_id) {
                                self.check_test_attr(meta, hir_id);
                            }
                        }

                        sym::html_favicon_url
                        | sym::html_logo_url
                        | sym::html_playground_url
                        | sym::issue_tracker_base_url
                        | sym::html_root_url
                        | sym::html_no_source => {
                            self.check_attr_crate_level(attr, meta, hir_id);
                        }

                        sym::cfg_hide => {
                            if self.check_attr_crate_level(attr, meta, hir_id) {
                                self.check_doc_cfg_hide(meta, hir_id);
                            }
                        }

                        sym::inline | sym::no_inline => {
                            self.check_doc_inline(attr, meta, hir_id, target, specified_inline)
                        }

                        sym::masked => self.check_doc_masked(attr, meta, hir_id, target),

                        sym::cfg | sym::hidden | sym::notable_trait => {}

                        sym::rust_logo => {
                            if self.check_attr_crate_level(attr, meta, hir_id)
                                && !self.tcx.features().rustdoc_internals()
                            {
                                feature_err(
                                    &self.tcx.sess,
                                    sym::rustdoc_internals,
                                    meta.span(),
                                    fluent::passes_doc_rust_logo,
                                )
                                .emit();
                            }
                        }

                        _ => {
                            let path = rustc_ast_pretty::pprust::path_to_string(&i_meta.path);
                            if i_meta.has_name(sym::spotlight) {
                                self.tcx.emit_node_span_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownSpotlight { path, span: i_meta.span },
                                );
                            } else if i_meta.has_name(sym::include)
                                && let Some(value) = i_meta.value_str()
                            {
                                let applicability = if list.len() == 1 {
                                    Applicability::MachineApplicable
                                } else {
                                    Applicability::MaybeIncorrect
                                };
                                // If there are multiple attributes, the suggestion would suggest
                                // deleting all of them, which is incorrect.
                                self.tcx.emit_node_span_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownInclude {
                                        path,
                                        value: value.to_string(),
                                        inner: match attr.style {
                                            AttrStyle::Inner => "!",
                                            AttrStyle::Outer => "",
                                        },
                                        sugg: (attr.meta().unwrap().span, applicability),
                                    },
                                );
                            } else if i_meta.has_name(sym::passes)
                                || i_meta.has_name(sym::no_default_passes)
                            {
                                self.tcx.emit_node_span_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownPasses { path, span: i_meta.span },
                                );
                            } else if i_meta.has_name(sym::plugins) {
                                self.tcx.emit_node_span_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownPlugins { path, span: i_meta.span },
                                );
                            } else {
                                self.tcx.emit_node_span_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownAny { path },
                                );
                            }
                        }
                    }
                } else {
                    self.tcx.emit_node_span_lint(
                        INVALID_DOC_ATTRIBUTES,
                        hir_id,
                        meta.span(),
                        errors::DocInvalid,
                    );
                }
            }
        }
    }

    /// Warns against some misuses of `#[pass_by_value]`
    fn check_pass_by_value(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Struct | Target::Enum | Target::TyAlias => {}
            _ => {
                self.dcx().emit_err(errors::PassByValue { attr_span: attr.span, span });
            }
        }
    }

    fn check_allow_incoherent_impl(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Method(MethodKind::Inherent) => {}
            _ => {
                self.dcx().emit_err(errors::AllowIncoherentImpl { attr_span: attr.span, span });
            }
        }
    }

    fn check_has_incoherent_inherent_impls(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Trait | Target::Struct | Target::Enum | Target::Union | Target::ForeignTy => {}
            _ => {
                self.tcx
                    .dcx()
                    .emit_err(errors::HasIncoherentInherentImpl { attr_span: attr.span, span });
            }
        }
    }

    fn check_ffi_pure(&self, attr_span: Span, attrs: &[Attribute], target: Target) {
        if target != Target::ForeignFn {
            self.dcx().emit_err(errors::FfiPureInvalidTarget { attr_span });
            return;
        }
        if attrs.iter().any(|a| a.has_name(sym::ffi_const)) {
            // `#[ffi_const]` functions cannot be `#[ffi_pure]`
            self.dcx().emit_err(errors::BothFfiConstAndPure { attr_span });
        }
    }

    fn check_ffi_const(&self, attr_span: Span, target: Target) {
        if target != Target::ForeignFn {
            self.dcx().emit_err(errors::FfiConstInvalidTarget { attr_span });
        }
    }

    /// Warns against some misuses of `#[must_use]`
    fn check_must_use(&self, hir_id: HirId, attr: &Attribute, target: Target) {
        if !matches!(
            target,
            Target::Fn
                | Target::Enum
                | Target::Struct
                | Target::Union
                | Target::Method(_)
                | Target::ForeignFn
                // `impl Trait` in return position can trip
                // `unused_must_use` if `Trait` is marked as
                // `#[must_use]`
                | Target::Trait
        ) {
            let article = match target {
                Target::ExternCrate
                | Target::Enum
                | Target::Impl
                | Target::Expression
                | Target::Arm
                | Target::AssocConst
                | Target::AssocTy => "an",
                _ => "a",
            };

            self.tcx.emit_node_span_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr.span,
                errors::MustUseNoEffect { article, target },
            );
        }
    }

    /// Checks if `#[must_not_suspend]` is applied to a struct, enum, union, or trait.
    fn check_must_not_suspend(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Struct | Target::Enum | Target::Union | Target::Trait => {}
            _ => {
                self.dcx().emit_err(errors::MustNotSuspend { attr_span: attr.span, span });
            }
        }
    }

    /// Checks if `#[may_dangle]` is applied to a lifetime or type generic parameter in `Drop` impl.
    fn check_may_dangle(&self, hir_id: HirId, attr: &Attribute) {
        if let hir::Node::GenericParam(param) = self.tcx.hir_node(hir_id)
            && matches!(
                param.kind,
                hir::GenericParamKind::Lifetime { .. } | hir::GenericParamKind::Type { .. }
            )
            && matches!(param.source, hir::GenericParamSource::Generics)
            && let parent_hir_id = self.tcx.parent_hir_id(hir_id)
            && let hir::Node::Item(item) = self.tcx.hir_node(parent_hir_id)
            && let hir::ItemKind::Impl(impl_) = item.kind
            && let Some(trait_) = impl_.of_trait
            && let Some(def_id) = trait_.trait_def_id()
            && self.tcx.is_lang_item(def_id, hir::LangItem::Drop)
        {
            return;
        }

        self.dcx().emit_err(errors::InvalidMayDangle { attr_span: attr.span });
    }

    /// Checks if `#[cold]` is applied to a non-function.
    fn check_cold(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Fn | Target::Method(..) | Target::ForeignFn | Target::Closure => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[cold]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "cold");
            }
            _ => {
                // FIXME: #[cold] was previously allowed on non-functions and some crates used
                // this, so only emit a warning.
                self.tcx.emit_node_span_lint(UNUSED_ATTRIBUTES, hir_id, attr.span, errors::Cold {
                    span,
                    on_crate: hir_id == CRATE_HIR_ID,
                });
            }
        }
    }

    /// Checks if `#[link]` is applied to an item other than a foreign module.
    fn check_link(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        if target == Target::ForeignMod
            && let hir::Node::Item(item) = self.tcx.hir_node(hir_id)
            && let Item { kind: ItemKind::ForeignMod { abi, .. }, .. } = item
            && !matches!(abi, Abi::Rust | Abi::RustIntrinsic)
        {
            return;
        }

        self.tcx.emit_node_span_lint(UNUSED_ATTRIBUTES, hir_id, attr.span, errors::Link {
            span: (target != Target::ForeignMod).then_some(span),
        });
    }

    /// Checks if `#[link_name]` is applied to an item other than a foreign function or static.
    fn check_link_name(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::ForeignFn | Target::ForeignStatic => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[link_name]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "link_name");
            }
            _ => {
                // FIXME: #[cold] was previously allowed on non-functions/statics and some crates
                // used this, so only emit a warning.
                let attr_span = matches!(target, Target::ForeignMod).then_some(attr.span);
                if let Some(s) = attr.value_str() {
                    self.tcx.emit_node_span_lint(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        attr.span,
                        errors::LinkName { span, attr_span, value: s.as_str() },
                    );
                } else {
                    self.tcx.emit_node_span_lint(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        attr.span,
                        errors::LinkName { span, attr_span, value: "..." },
                    );
                };
            }
        }
    }

    /// Checks if `#[no_link]` is applied to an `extern crate`.
    fn check_no_link(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::ExternCrate => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[no_link]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "no_link");
            }
            _ => {
                self.dcx().emit_err(errors::NoLink { attr_span: attr.span, span });
            }
        }
    }

    fn is_impl_item(&self, hir_id: HirId) -> bool {
        matches!(self.tcx.hir_node(hir_id), hir::Node::ImplItem(..))
    }

    /// Checks if `#[export_name]` is applied to a function or static.
    fn check_export_name(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Static | Target::Fn => {}
            Target::Method(..) if self.is_impl_item(hir_id) => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[export_name]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "export_name");
            }
            _ => {
                self.dcx().emit_err(errors::ExportName { attr_span: attr.span, span });
            }
        }
    }

    fn check_rustc_layout_scalar_valid_range(&self, attr: &Attribute, span: Span, target: Target) {
        if target != Target::Struct {
            self.dcx().emit_err(errors::RustcLayoutScalarValidRangeNotStruct {
                attr_span: attr.span,
                span,
            });
            return;
        }

        let Some(list) = attr.meta_item_list() else {
            return;
        };

        if !matches!(&list[..], &[MetaItemInner::Lit(MetaItemLit { kind: LitKind::Int(..), .. })]) {
            self.tcx
                .dcx()
                .emit_err(errors::RustcLayoutScalarValidRangeArg { attr_span: attr.span });
        }
    }

    /// Checks if `#[rustc_legacy_const_generics]` is applied to a function and has a valid argument.
    fn check_rustc_legacy_const_generics(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
        item: Option<ItemLike<'_>>,
    ) {
        let is_function = matches!(target, Target::Fn);
        if !is_function {
            self.dcx().emit_err(errors::AttrShouldBeAppliedToFn {
                attr_span: attr.span,
                defn_span: span,
                on_crate: hir_id == CRATE_HIR_ID,
            });
            return;
        }

        let Some(list) = attr.meta_item_list() else {
            // The attribute form is validated on AST.
            return;
        };

        let Some(ItemLike::Item(Item {
            kind: ItemKind::Fn(FnSig { decl, .. }, generics, _), ..
        })) = item
        else {
            bug!("should be a function item");
        };

        for param in generics.params {
            match param.kind {
                hir::GenericParamKind::Const { .. } => {}
                _ => {
                    self.dcx().emit_err(errors::RustcLegacyConstGenericsOnly {
                        attr_span: attr.span,
                        param_span: param.span,
                    });
                    return;
                }
            }
        }

        if list.len() != generics.params.len() {
            self.dcx().emit_err(errors::RustcLegacyConstGenericsIndex {
                attr_span: attr.span,
                generics_span: generics.span,
            });
            return;
        }

        let arg_count = decl.inputs.len() as u128 + generics.params.len() as u128;
        let mut invalid_args = vec![];
        for meta in list {
            if let Some(LitKind::Int(val, _)) = meta.lit().map(|lit| &lit.kind) {
                if *val >= arg_count {
                    let span = meta.span();
                    self.dcx().emit_err(errors::RustcLegacyConstGenericsIndexExceed {
                        span,
                        arg_count: arg_count as usize,
                    });
                    return;
                }
            } else {
                invalid_args.push(meta.span());
            }
        }

        if !invalid_args.is_empty() {
            self.dcx().emit_err(errors::RustcLegacyConstGenericsIndexNegative { invalid_args });
        }
    }

    /// Helper function for checking that the provided attribute is only applied to a function or
    /// method.
    fn check_applied_to_fn_or_method(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) {
        let is_function = matches!(target, Target::Fn | Target::Method(..));
        if !is_function {
            self.dcx().emit_err(errors::AttrShouldBeAppliedToFn {
                attr_span: attr.span,
                defn_span: span,
                on_crate: hir_id == CRATE_HIR_ID,
            });
        }
    }

    /// Checks that the `#[rustc_lint_opt_ty]` attribute is only applied to a struct.
    fn check_rustc_lint_opt_ty(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Struct => {}
            _ => {
                self.dcx().emit_err(errors::RustcLintOptTy { attr_span: attr.span, span });
            }
        }
    }

    /// Checks that the `#[rustc_lint_opt_deny_field_access]` attribute is only applied to a field.
    fn check_rustc_lint_opt_deny_field_access(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Field => {}
            _ => {
                self.tcx
                    .dcx()
                    .emit_err(errors::RustcLintOptDenyFieldAccess { attr_span: attr.span, span });
            }
        }
    }

    /// Checks that the dep-graph debugging attributes are only present when the query-dep-graph
    /// option is passed to the compiler.
    fn check_rustc_dirty_clean(&self, attr: &Attribute) {
        if !self.tcx.sess.opts.unstable_opts.query_dep_graph {
            self.dcx().emit_err(errors::RustcDirtyClean { span: attr.span });
        }
    }

    /// Checks if the attribute is applied to a trait.
    fn check_must_be_applied_to_trait(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Trait => {}
            _ => {
                self.dcx().emit_err(errors::AttrShouldBeAppliedToTrait {
                    attr_span: attr.span,
                    defn_span: span,
                });
            }
        }
    }

    /// Checks if `#[link_section]` is applied to a function or static.
    fn check_link_section(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Static | Target::Fn | Target::Method(..) => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[link_section]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "link_section");
            }
            _ => {
                // FIXME: #[link_section] was previously allowed on non-functions/statics and some
                // crates used this, so only emit a warning.
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::LinkSection { span },
                );
            }
        }
    }

    /// Checks if `#[no_mangle]` is applied to a function or static.
    fn check_no_mangle(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Static | Target::Fn => {}
            Target::Method(..) if self.is_impl_item(hir_id) => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[no_mangle]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "no_mangle");
            }
            // FIXME: #[no_mangle] was previously allowed on non-functions/statics, this should be an error
            // The error should specify that the item that is wrong is specifically a *foreign* fn/static
            // otherwise the error seems odd
            Target::ForeignFn | Target::ForeignStatic => {
                let foreign_item_kind = match target {
                    Target::ForeignFn => "function",
                    Target::ForeignStatic => "static",
                    _ => unreachable!(),
                };
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::NoMangleForeign { span, attr_span: attr.span, foreign_item_kind },
                );
            }
            _ => {
                // FIXME: #[no_mangle] was previously allowed on non-functions/statics and some
                // crates used this, so only emit a warning.
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::NoMangle { span },
                );
            }
        }
    }

    /// Checks if the `#[repr]` attributes on `item` are valid.
    fn check_repr(
        &self,
        attrs: &[Attribute],
        span: Span,
        target: Target,
        item: Option<ItemLike<'_>>,
        hir_id: HirId,
    ) {
        // Extract the names of all repr hints, e.g., [foo, bar, align] for:
        // ```
        // #[repr(foo)]
        // #[repr(bar, align(8))]
        // ```
        let hints: Vec<_> = attrs
            .iter()
            .filter(|attr| attr.has_name(sym::repr))
            .filter_map(|attr| attr.meta_item_list())
            .flatten()
            .collect();

        let mut int_reprs = 0;
        let mut is_explicit_rust = false;
        let mut is_c = false;
        let mut is_simd = false;
        let mut is_transparent = false;

        for hint in &hints {
            if !hint.is_meta_item() {
                self.dcx().emit_err(errors::ReprIdent { span: hint.span() });
                continue;
            }

            match hint.name_or_empty() {
                sym::Rust => {
                    is_explicit_rust = true;
                    match target {
                        Target::Struct | Target::Union | Target::Enum => continue,
                        _ => {
                            self.dcx().emit_err(errors::AttrApplication::StructEnumUnion {
                                hint_span: hint.span(),
                                span,
                            });
                        }
                    }
                }
                sym::C => {
                    is_c = true;
                    match target {
                        Target::Struct | Target::Union | Target::Enum => continue,
                        _ => {
                            self.dcx().emit_err(errors::AttrApplication::StructEnumUnion {
                                hint_span: hint.span(),
                                span,
                            });
                        }
                    }
                }
                sym::align => {
                    if let (Target::Fn | Target::Method(MethodKind::Inherent), false) =
                        (target, self.tcx.features().fn_align())
                    {
                        feature_err(
                            &self.tcx.sess,
                            sym::fn_align,
                            hint.span(),
                            fluent::passes_repr_align_function,
                        )
                        .emit();
                    }

                    match target {
                        Target::Struct
                        | Target::Union
                        | Target::Enum
                        | Target::Fn
                        | Target::Method(_) => {}
                        _ => {
                            self.dcx().emit_err(
                                errors::AttrApplication::StructEnumFunctionMethodUnion {
                                    hint_span: hint.span(),
                                    span,
                                },
                            );
                        }
                    }

                    self.check_align_value(hint);
                }
                sym::packed => {
                    if target != Target::Struct && target != Target::Union {
                        self.dcx().emit_err(errors::AttrApplication::StructUnion {
                            hint_span: hint.span(),
                            span,
                        });
                    } else {
                        continue;
                    }
                }
                sym::simd => {
                    is_simd = true;
                    if target != Target::Struct {
                        self.dcx().emit_err(errors::AttrApplication::Struct {
                            hint_span: hint.span(),
                            span,
                        });
                    } else {
                        continue;
                    }
                }
                sym::transparent => {
                    is_transparent = true;
                    match target {
                        Target::Struct | Target::Union | Target::Enum => continue,
                        _ => {
                            self.dcx().emit_err(errors::AttrApplication::StructEnumUnion {
                                hint_span: hint.span(),
                                span,
                            });
                        }
                    }
                }
                sym::i8
                | sym::u8
                | sym::i16
                | sym::u16
                | sym::i32
                | sym::u32
                | sym::i64
                | sym::u64
                | sym::i128
                | sym::u128
                | sym::isize
                | sym::usize => {
                    int_reprs += 1;
                    if target != Target::Enum {
                        self.dcx().emit_err(errors::AttrApplication::Enum {
                            hint_span: hint.span(),
                            span,
                        });
                    } else {
                        continue;
                    }
                }
                _ => {
                    self.dcx().emit_err(errors::UnrecognizedReprHint { span: hint.span() });
                    continue;
                }
            };
        }

        // Just point at all repr hints if there are any incompatibilities.
        // This is not ideal, but tracking precisely which ones are at fault is a huge hassle.
        let hint_spans = hints.iter().map(|hint| hint.span());

        // Error on repr(transparent, <anything else>).
        if is_transparent && hints.len() > 1 {
            let hint_spans = hint_spans.clone().collect();
            self.dcx().emit_err(errors::TransparentIncompatible {
                hint_spans,
                target: target.to_string(),
            });
        }
        if is_explicit_rust && (int_reprs > 0 || is_c || is_simd) {
            let hint_spans = hint_spans.clone().collect();
            self.dcx().emit_err(errors::ReprConflicting { hint_spans });
        }
        // Warn on repr(u8, u16), repr(C, simd), and c-like-enum-repr(C, u8)
        if (int_reprs > 1)
            || (is_simd && is_c)
            || (int_reprs == 1
                && is_c
                && item.is_some_and(|item| {
                    if let ItemLike::Item(item) = item { is_c_like_enum(item) } else { false }
                }))
        {
            self.tcx.emit_node_span_lint(
                CONFLICTING_REPR_HINTS,
                hir_id,
                hint_spans.collect::<Vec<Span>>(),
                errors::ReprConflictingLint,
            );
        }
    }

    fn check_align_value(&self, item: &MetaItemInner) {
        match item.singleton_lit_list() {
            Some((
                _,
                MetaItemLit {
                    kind: ast::LitKind::Int(literal, ast::LitIntType::Unsuffixed), ..
                },
            )) => {
                let val = literal.get() as u64;
                if val > 2_u64.pow(29) {
                    // for values greater than 2^29, a different error will be emitted, make sure that happens
                    self.dcx().span_delayed_bug(
                        item.span(),
                        "alignment greater than 2^29 should be errored on elsewhere",
                    );
                } else {
                    // only do this check when <= 2^29 to prevent duplicate errors:
                    // alignment greater than 2^29 not supported
                    // alignment is too large for the current target

                    let max =
                        Size::from_bits(self.tcx.sess.target.pointer_width).signed_int_max() as u64;
                    if val > max {
                        self.dcx().emit_err(errors::InvalidReprAlignForTarget {
                            span: item.span(),
                            size: max,
                        });
                    }
                }
            }

            // if the attribute is malformed, singleton_lit_list may not be of the expected type or may be None
            // but an error will have already been emitted, so this code should just skip such attributes
            Some((_, _)) | None => {
                self.dcx().span_delayed_bug(item.span(), "malformed repr(align(N))");
            }
        }
    }

    fn check_used(&self, attrs: &[Attribute], target: Target, target_span: Span) {
        let mut used_linker_span = None;
        let mut used_compiler_span = None;
        for attr in attrs.iter().filter(|attr| attr.has_name(sym::used)) {
            if target != Target::Static {
                self.dcx().emit_err(errors::UsedStatic {
                    attr_span: attr.span,
                    span: target_span,
                    target: target.name(),
                });
            }
            let inner = attr.meta_item_list();
            match inner.as_deref() {
                Some([item]) if item.has_name(sym::linker) => {
                    if used_linker_span.is_none() {
                        used_linker_span = Some(attr.span);
                    }
                }
                Some([item]) if item.has_name(sym::compiler) => {
                    if used_compiler_span.is_none() {
                        used_compiler_span = Some(attr.span);
                    }
                }
                Some(_) => {
                    // This error case is handled in rustc_hir_analysis::collect.
                }
                None => {
                    // Default case (compiler) when arg isn't defined.
                    if used_compiler_span.is_none() {
                        used_compiler_span = Some(attr.span);
                    }
                }
            }
        }
        if let (Some(linker_span), Some(compiler_span)) = (used_linker_span, used_compiler_span) {
            self.tcx
                .dcx()
                .emit_err(errors::UsedCompilerLinker { spans: vec![linker_span, compiler_span] });
        }
    }

    /// Outputs an error for `#[allow_internal_unstable]` which can only be applied to macros.
    /// (Allows proc_macro functions)
    fn check_allow_internal_unstable(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
        attrs: &[Attribute],
    ) {
        debug!("Checking target: {:?}", target);
        match target {
            Target::Fn => {
                for attr in attrs {
                    if attr.is_proc_macro_attr() {
                        debug!("Is proc macro attr");
                        return;
                    }
                }
                debug!("Is not proc macro attr");
            }
            Target::MacroDef => {}
            // FIXME(#80564): We permit struct fields and match arms to have an
            // `#[allow_internal_unstable]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm => self.inline_attr_str_error_without_macro_def(
                hir_id,
                attr,
                "allow_internal_unstable",
            ),
            _ => {
                self.tcx
                    .dcx()
                    .emit_err(errors::AllowInternalUnstable { attr_span: attr.span, span });
            }
        }
    }

    /// Checks if the items on the `#[debugger_visualizer]` attribute are valid.
    fn check_debugger_visualizer(&self, attr: &Attribute, target: Target) {
        // Here we only check that the #[debugger_visualizer] attribute is attached
        // to nothing other than a module. All other checks are done in the
        // `debugger_visualizer` query where they need to be done for decoding
        // anyway.
        match target {
            Target::Mod => {}
            _ => {
                self.dcx().emit_err(errors::DebugVisualizerPlacement { span: attr.span });
            }
        }
    }

    /// Outputs an error for `#[allow_internal_unstable]` which can only be applied to macros.
    /// (Allows proc_macro functions)
    fn check_rustc_allow_const_fn_unstable(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) {
        match target {
            Target::Fn | Target::Method(_)
                if self.tcx.is_const_fn(hir_id.expect_owner().to_def_id()) => {}
            // FIXME(#80564): We permit struct fields and match arms to have an
            // `#[allow_internal_unstable]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "allow_internal_unstable")
            }
            _ => {
                self.tcx
                    .dcx()
                    .emit_err(errors::RustcAllowConstFnUnstable { attr_span: attr.span, span });
            }
        }
    }

    fn check_rustc_std_internal_symbol(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Fn | Target::Static => {}
            _ => {
                self.tcx
                    .dcx()
                    .emit_err(errors::RustcStdInternalSymbol { attr_span: attr.span, span });
            }
        }
    }

    fn check_stability_promotable(&self, attr: &Attribute, target: Target) {
        match target {
            Target::Expression => {
                self.dcx().emit_err(errors::StabilityPromotable { attr_span: attr.span });
            }
            _ => {}
        }
    }

    fn check_link_ordinal(&self, attr: &Attribute, _span: Span, target: Target) {
        match target {
            Target::ForeignFn | Target::ForeignStatic => {}
            _ => {
                self.dcx().emit_err(errors::LinkOrdinal { attr_span: attr.span });
            }
        }
    }

    fn check_confusables(&self, attr: &Attribute, target: Target) {
        match target {
            Target::Method(MethodKind::Inherent) => {
                let Some(meta) = attr.meta() else {
                    return;
                };
                let ast::MetaItem { kind: MetaItemKind::List(ref metas), .. } = meta else {
                    return;
                };

                let mut candidates = Vec::new();

                for meta in metas {
                    let MetaItemInner::Lit(meta_lit) = meta else {
                        self.dcx().emit_err(errors::IncorrectMetaItem {
                            span: meta.span(),
                            suggestion: errors::IncorrectMetaItemSuggestion {
                                lo: meta.span().shrink_to_lo(),
                                hi: meta.span().shrink_to_hi(),
                            },
                        });
                        return;
                    };
                    candidates.push(meta_lit.symbol);
                }

                if candidates.is_empty() {
                    self.dcx().emit_err(errors::EmptyConfusables { span: attr.span });
                }
            }
            _ => {
                self.dcx().emit_err(errors::Confusables { attr_span: attr.span });
            }
        }
    }

    fn check_deprecated(&self, hir_id: HirId, attr: &Attribute, _span: Span, target: Target) {
        match target {
            Target::Closure | Target::Expression | Target::Statement | Target::Arm => {
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::Deprecated,
                );
            }
            _ => {}
        }
    }

    fn check_macro_use(&self, hir_id: HirId, attr: &Attribute, target: Target) {
        let name = attr.name_or_empty();
        match target {
            Target::ExternCrate | Target::Mod => {}
            _ => {
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::MacroUse { name },
                );
            }
        }
    }

    fn check_macro_export(&self, hir_id: HirId, attr: &Attribute, target: Target) {
        if target != Target::MacroDef {
            self.tcx.emit_node_span_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr.span,
                errors::MacroExport::Normal,
            );
        } else if let Some(meta_item_list) = attr.meta_item_list()
            && !meta_item_list.is_empty()
        {
            if meta_item_list.len() > 1 {
                self.tcx.emit_node_span_lint(
                    INVALID_MACRO_EXPORT_ARGUMENTS,
                    hir_id,
                    attr.span,
                    errors::MacroExport::TooManyItems,
                );
            } else if meta_item_list[0].name_or_empty() != sym::local_inner_macros {
                self.tcx.emit_node_span_lint(
                    INVALID_MACRO_EXPORT_ARGUMENTS,
                    hir_id,
                    meta_item_list[0].span(),
                    errors::MacroExport::UnknownItem { name: meta_item_list[0].name_or_empty() },
                );
            }
        } else {
            // special case when `#[macro_export]` is applied to a macro 2.0
            let (macro_definition, _) = self.tcx.hir_node(hir_id).expect_item().expect_macro();
            let is_decl_macro = !macro_definition.macro_rules;

            if is_decl_macro {
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::MacroExport::OnDeclMacro,
                );
            }
        }
    }

    fn check_unused_attribute(&self, hir_id: HirId, attr: &Attribute) {
        // Warn on useless empty attributes.
        let note = if matches!(
            attr.name_or_empty(),
            sym::macro_use
                | sym::allow
                | sym::expect
                | sym::warn
                | sym::deny
                | sym::forbid
                | sym::feature
                | sym::repr
                | sym::target_feature
        ) && attr.meta_item_list().is_some_and(|list| list.is_empty())
        {
            errors::UnusedNote::EmptyList { name: attr.name_or_empty() }
        } else if matches!(
            attr.name_or_empty(),
            sym::allow | sym::warn | sym::deny | sym::forbid | sym::expect
        ) && let Some(meta) = attr.meta_item_list()
            && let [meta] = meta.as_slice()
            && let Some(item) = meta.meta_item()
            && let MetaItemKind::NameValue(_) = &item.kind
            && item.path == sym::reason
        {
            errors::UnusedNote::NoLints { name: attr.name_or_empty() }
        } else if attr.name_or_empty() == sym::default_method_body_is_const {
            errors::UnusedNote::DefaultMethodBodyConst
        } else {
            return;
        };

        self.tcx.emit_node_span_lint(UNUSED_ATTRIBUTES, hir_id, attr.span, errors::Unused {
            attr_span: attr.span,
            note,
        });
    }

    /// A best effort attempt to create an error for a mismatching proc macro signature.
    ///
    /// If this best effort goes wrong, it will just emit a worse error later (see #102923)
    fn check_proc_macro(&self, hir_id: HirId, target: Target, kind: ProcMacroKind) {
        if target != Target::Fn {
            return;
        }

        let tcx = self.tcx;
        let Some(token_stream_def_id) = tcx.get_diagnostic_item(sym::TokenStream) else {
            return;
        };
        let Some(token_stream) = tcx.type_of(token_stream_def_id).no_bound_vars() else {
            return;
        };

        let def_id = hir_id.expect_owner().def_id;
        let param_env = ty::ParamEnv::empty();

        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

        let span = tcx.def_span(def_id);
        let fresh_args = infcx.fresh_args_for_item(span, def_id.to_def_id());
        let sig = tcx.liberate_late_bound_regions(
            def_id.to_def_id(),
            tcx.fn_sig(def_id).instantiate(tcx, fresh_args),
        );

        let mut cause = ObligationCause::misc(span, def_id);
        let sig = ocx.normalize(&cause, param_env, sig);

        // proc macro is not WF.
        let errors = ocx.select_where_possible();
        if !errors.is_empty() {
            return;
        }

        let expected_sig = tcx.mk_fn_sig(
            std::iter::repeat(token_stream).take(match kind {
                ProcMacroKind::Attribute => 2,
                ProcMacroKind::Derive | ProcMacroKind::FunctionLike => 1,
            }),
            token_stream,
            false,
            Safety::Safe,
            Abi::Rust,
        );

        if let Err(terr) = ocx.eq(&cause, param_env, expected_sig, sig) {
            let mut diag = tcx.dcx().create_err(errors::ProcMacroBadSig { span, kind });

            let hir_sig = tcx.hir().fn_sig_by_hir_id(hir_id);
            if let Some(hir_sig) = hir_sig {
                #[allow(rustc::diagnostic_outside_of_impl)] // FIXME
                match terr {
                    TypeError::ArgumentMutability(idx) | TypeError::ArgumentSorts(_, idx) => {
                        if let Some(ty) = hir_sig.decl.inputs.get(idx) {
                            diag.span(ty.span);
                            cause.span = ty.span;
                        } else if idx == hir_sig.decl.inputs.len() {
                            let span = hir_sig.decl.output.span();
                            diag.span(span);
                            cause.span = span;
                        }
                    }
                    TypeError::ArgCount => {
                        if let Some(ty) = hir_sig.decl.inputs.get(expected_sig.inputs().len()) {
                            diag.span(ty.span);
                            cause.span = ty.span;
                        }
                    }
                    TypeError::SafetyMismatch(_) => {
                        // FIXME: Would be nice if we had a span here..
                    }
                    TypeError::AbiMismatch(_) => {
                        // FIXME: Would be nice if we had a span here..
                    }
                    TypeError::VariadicMismatch(_) => {
                        // FIXME: Would be nice if we had a span here..
                    }
                    _ => {}
                }
            }

            infcx.err_ctxt().note_type_err(
                &mut diag,
                &cause,
                None,
                Some(param_env.and(ValuePairs::PolySigs(ExpectedFound {
                    expected: ty::Binder::dummy(expected_sig),
                    found: ty::Binder::dummy(sig),
                }))),
                terr,
                false,
            );
            diag.emit();
            self.abort.set(true);
        }

        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            infcx.err_ctxt().report_fulfillment_errors(errors);
            self.abort.set(true);
        }
    }

    fn check_coroutine(&self, attr: &Attribute, target: Target) {
        match target {
            Target::Closure => return,
            _ => {
                self.dcx().emit_err(errors::CoroutineOnNonClosure { span: attr.span });
            }
        }
    }

    fn check_linkage(&self, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Fn
            | Target::Method(..)
            | Target::Static
            | Target::ForeignStatic
            | Target::ForeignFn => {}
            _ => {
                self.dcx().emit_err(errors::Linkage { attr_span: attr.span, span });
            }
        }
    }

    fn check_rustc_pub_transparent(&self, attr_span: Span, span: Span, attrs: &[Attribute]) {
        if !attrs
            .iter()
            .filter(|attr| attr.has_name(sym::repr))
            .filter_map(|attr| attr.meta_item_list())
            .flatten()
            .any(|nmi| nmi.has_name(sym::transparent))
        {
            self.dcx().emit_err(errors::RustcPubTransparent { span, attr_span });
        }
    }

    /// Checks if `#[autodiff]` is applied to an item other than a function item.
    fn check_autodiff(&self, _hir_id: HirId, _attr: &Attribute, span: Span, target: Target) {
        debug!("check_autodiff");
        match target {
            Target::Fn => {}
            _ => {
                self.dcx().emit_err(errors::AutoDiffAttr { attr_span: span });
                self.abort.set(true);
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for CheckAttrVisitor<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        // Historically we've run more checks on non-exported than exported macros,
        // so this lets us continue to run them while maintaining backwards compatibility.
        // In the long run, the checks should be harmonized.
        if let ItemKind::Macro(macro_def, _) = item.kind {
            let def_id = item.owner_id.to_def_id();
            if macro_def.macro_rules && !self.tcx.has_attr(def_id, sym::macro_export) {
                check_non_exported_macro_for_invalid_attrs(self.tcx, item);
            }
        }

        let target = Target::from_item(item);
        self.check_attributes(item.hir_id(), item.span, target, Some(ItemLike::Item(item)));
        intravisit::walk_item(self, item)
    }

    fn visit_generic_param(&mut self, generic_param: &'tcx hir::GenericParam<'tcx>) {
        let target = Target::from_generic_param(generic_param);
        self.check_attributes(generic_param.hir_id, generic_param.span, target, None);
        intravisit::walk_generic_param(self, generic_param)
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx TraitItem<'tcx>) {
        let target = Target::from_trait_item(trait_item);
        self.check_attributes(trait_item.hir_id(), trait_item.span, target, None);
        intravisit::walk_trait_item(self, trait_item)
    }

    fn visit_field_def(&mut self, struct_field: &'tcx hir::FieldDef<'tcx>) {
        self.check_attributes(struct_field.hir_id, struct_field.span, Target::Field, None);
        intravisit::walk_field_def(self, struct_field);
    }

    fn visit_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) {
        self.check_attributes(arm.hir_id, arm.span, Target::Arm, None);
        intravisit::walk_arm(self, arm);
    }

    fn visit_foreign_item(&mut self, f_item: &'tcx ForeignItem<'tcx>) {
        let target = Target::from_foreign_item(f_item);
        self.check_attributes(f_item.hir_id(), f_item.span, target, Some(ItemLike::ForeignItem));
        intravisit::walk_foreign_item(self, f_item)
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        let target = target_from_impl_item(self.tcx, impl_item);
        self.check_attributes(impl_item.hir_id(), impl_item.span, target, None);
        intravisit::walk_impl_item(self, impl_item)
    }

    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
        // When checking statements ignore expressions, they will be checked later.
        if let hir::StmtKind::Let(l) = stmt.kind {
            self.check_attributes(l.hir_id, stmt.span, Target::Statement, None);
        }
        intravisit::walk_stmt(self, stmt)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let target = match expr.kind {
            hir::ExprKind::Closure { .. } => Target::Closure,
            _ => Target::Expression,
        };

        self.check_attributes(expr.hir_id, expr.span, target, None);
        intravisit::walk_expr(self, expr)
    }

    fn visit_expr_field(&mut self, field: &'tcx hir::ExprField<'tcx>) {
        self.check_attributes(field.hir_id, field.span, Target::ExprField, None);
        intravisit::walk_expr_field(self, field)
    }

    fn visit_variant(&mut self, variant: &'tcx hir::Variant<'tcx>) {
        self.check_attributes(variant.hir_id, variant.span, Target::Variant, None);
        intravisit::walk_variant(self, variant)
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        self.check_attributes(param.hir_id, param.span, Target::Param, None);

        intravisit::walk_param(self, param);
    }

    fn visit_pat_field(&mut self, field: &'tcx hir::PatField<'tcx>) {
        self.check_attributes(field.hir_id, field.span, Target::PatField, None);
        intravisit::walk_pat_field(self, field);
    }
}

fn is_c_like_enum(item: &Item<'_>) -> bool {
    if let ItemKind::Enum(ref def, _) = item.kind {
        for variant in def.variants {
            match variant.data {
                hir::VariantData::Unit(..) => { /* continue */ }
                _ => return false,
            }
        }
        true
    } else {
        false
    }
}

// FIXME: Fix "Cannot determine resolution" error and remove built-in macros
// from this check.
fn check_invalid_crate_level_attr(tcx: TyCtxt<'_>, attrs: &[Attribute]) {
    // Check for builtin attributes at the crate level
    // which were unsuccessfully resolved due to cannot determine
    // resolution for the attribute macro error.
    const ATTRS_TO_CHECK: &[Symbol] = &[
        sym::macro_export,
        sym::repr,
        sym::path,
        sym::automatically_derived,
        sym::start,
        sym::rustc_main,
        sym::derive,
        sym::test,
        sym::test_case,
        sym::global_allocator,
        sym::bench,
    ];

    for attr in attrs {
        // This function should only be called with crate attributes
        // which are inner attributes always but lets check to make sure
        if attr.style == AttrStyle::Inner {
            for attr_to_check in ATTRS_TO_CHECK {
                if attr.has_name(*attr_to_check) {
                    let item = tcx
                        .hir()
                        .items()
                        .map(|id| tcx.hir().item(id))
                        .find(|item| !item.span.is_dummy()) // Skip prelude `use`s
                        .map(|item| errors::ItemFollowingInnerAttr {
                            span: item.ident.span,
                            kind: item.kind.descr(),
                        });
                    let err = tcx.dcx().create_err(errors::InvalidAttrAtCrateLevel {
                        span: attr.span,
                        sugg_span: tcx
                            .sess
                            .source_map()
                            .span_to_snippet(attr.span)
                            .ok()
                            .filter(|src| src.starts_with("#!["))
                            .map(|_| {
                                attr.span
                                    .with_lo(attr.span.lo() + BytePos(1))
                                    .with_hi(attr.span.lo() + BytePos(2))
                            }),
                        name: *attr_to_check,
                        item,
                    });

                    if let AttrKind::Normal(ref p) = attr.kind {
                        tcx.dcx().try_steal_replace_and_emit_err(
                            p.item.path.span,
                            StashKey::UndeterminedMacroResolution,
                            err,
                        );
                    } else {
                        err.emit();
                    }
                }
            }
        }
    }
}

fn check_non_exported_macro_for_invalid_attrs(tcx: TyCtxt<'_>, item: &Item<'_>) {
    let attrs = tcx.hir().attrs(item.hir_id());

    for attr in attrs {
        if attr.has_name(sym::inline) {
            tcx.dcx().emit_err(errors::NonExportedMacroInvalidAttrs { attr_span: attr.span });
        }
    }
}

fn check_mod_attrs(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    let check_attr_visitor = &mut CheckAttrVisitor { tcx, abort: Cell::new(false) };
    tcx.hir().visit_item_likes_in_module(module_def_id, check_attr_visitor);
    if module_def_id.to_local_def_id().is_top_level_module() {
        check_attr_visitor.check_attributes(CRATE_HIR_ID, DUMMY_SP, Target::Mod, None);
        check_invalid_crate_level_attr(tcx, tcx.hir().krate_attrs());
    }
    if check_attr_visitor.abort.get() {
        tcx.dcx().abort_if_errors()
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_attrs, ..*providers };
}

fn check_duplicates(
    tcx: TyCtxt<'_>,
    attr: &Attribute,
    hir_id: HirId,
    duplicates: AttributeDuplicates,
    seen: &mut FxHashMap<Symbol, Span>,
) {
    use AttributeDuplicates::*;
    if matches!(duplicates, WarnFollowingWordOnly) && !attr.is_word() {
        return;
    }
    match duplicates {
        DuplicatesOk => {}
        WarnFollowing | FutureWarnFollowing | WarnFollowingWordOnly | FutureWarnPreceding => {
            match seen.entry(attr.name_or_empty()) {
                Entry::Occupied(mut entry) => {
                    let (this, other) = if matches!(duplicates, FutureWarnPreceding) {
                        let to_remove = entry.insert(attr.span);
                        (to_remove, attr.span)
                    } else {
                        (attr.span, *entry.get())
                    };
                    tcx.emit_node_span_lint(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        this,
                        errors::UnusedDuplicate {
                            this,
                            other,
                            warning: matches!(
                                duplicates,
                                FutureWarnFollowing | FutureWarnPreceding
                            ),
                        },
                    );
                }
                Entry::Vacant(entry) => {
                    entry.insert(attr.span);
                }
            }
        }
        ErrorFollowing | ErrorPreceding => match seen.entry(attr.name_or_empty()) {
            Entry::Occupied(mut entry) => {
                let (this, other) = if matches!(duplicates, ErrorPreceding) {
                    let to_remove = entry.insert(attr.span);
                    (to_remove, attr.span)
                } else {
                    (attr.span, *entry.get())
                };
                tcx.dcx().emit_err(errors::UnusedMultiple {
                    this,
                    other,
                    name: attr.name_or_empty(),
                });
            }
            Entry::Vacant(entry) => {
                entry.insert(attr.span);
            }
        },
    }
}

fn doc_fake_variadic_is_allowed_self_ty(self_ty: &hir::Ty<'_>) -> bool {
    matches!(&self_ty.kind, hir::TyKind::Tup([_]))
        || if let hir::TyKind::BareFn(bare_fn_ty) = &self_ty.kind {
            bare_fn_ty.decl.inputs.len() == 1
        } else {
            false
        }
        || (if let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = &self_ty.kind
            && let Some(&[hir::GenericArg::Type(ty)]) =
                path.segments.last().map(|last| last.args().args)
        {
            doc_fake_variadic_is_allowed_self_ty(ty)
        } else {
            false
        })
}

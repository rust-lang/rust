//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use crate::errors;
use rustc_ast::{ast, AttrStyle, Attribute, Lit, LitKind, MetaItemKind, NestedMetaItem};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{fluent, struct_span_err, Applicability, MultiSpan};
use rustc_expand::base::resolve_path;
use rustc_feature::{AttributeDuplicates, AttributeType, BuiltinAttribute, BUILTIN_ATTRIBUTE_MAP};
use rustc_hir as hir;
use rustc_hir::def_id::{LocalDefId, CRATE_DEF_ID};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self, FnSig, ForeignItem, HirId, Item, ItemKind, TraitItem, CRATE_HIR_ID};
use rustc_hir::{MethodKind, Target};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_lifetime::ObjectLifetimeDefault;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::{
    CONFLICTING_REPR_HINTS, INVALID_DOC_ATTRIBUTES, UNUSED_ATTRIBUTES,
};
use rustc_session::parse::feature_err;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::spec::abi::Abi;
use std::collections::hash_map::Entry;

pub(crate) fn target_from_impl_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_item: &hir::ImplItem<'_>,
) -> Target {
    match impl_item.kind {
        hir::ImplItemKind::Const(..) => Target::AssocConst,
        hir::ImplItemKind::Fn(..) => {
            let parent_hir_id = tcx.hir().get_parent_item(impl_item.hir_id());
            let containing_item = tcx.hir().expect_item(parent_hir_id);
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
        hir::ImplItemKind::TyAlias(..) => Target::AssocTy,
    }
}

#[derive(Clone, Copy)]
enum ItemLike<'tcx> {
    Item(&'tcx Item<'tcx>),
    ForeignItem,
}

struct CheckAttrVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl CheckAttrVisitor<'_> {
    /// Checks any attribute.
    fn check_attributes(
        &self,
        hir_id: HirId,
        span: Span,
        target: Target,
        item: Option<ItemLike<'_>>,
    ) {
        let mut doc_aliases = FxHashMap::default();
        let mut is_valid = true;
        let mut specified_inline = None;
        let mut seen = FxHashMap::default();
        let attrs = self.tcx.hir().attrs(hir_id);
        for attr in attrs {
            let attr_is_valid = match attr.name_or_empty() {
                sym::inline => self.check_inline(hir_id, attr, span, target),
                sym::no_coverage => self.check_no_coverage(hir_id, attr, span, target),
                sym::non_exhaustive => self.check_non_exhaustive(hir_id, attr, span, target),
                sym::marker => self.check_marker(hir_id, attr, span, target),
                sym::rustc_must_implement_one_of => {
                    self.check_rustc_must_implement_one_of(attr, span, target)
                }
                sym::target_feature => self.check_target_feature(hir_id, attr, span, target),
                sym::thread_local => self.check_thread_local(attr, span, target),
                sym::track_caller => {
                    self.check_track_caller(hir_id, attr.span, attrs, span, target)
                }
                sym::doc => self.check_doc_attrs(
                    attr,
                    hir_id,
                    target,
                    &mut specified_inline,
                    &mut doc_aliases,
                ),
                sym::no_link => self.check_no_link(hir_id, &attr, span, target),
                sym::export_name => self.check_export_name(hir_id, &attr, span, target),
                sym::rustc_layout_scalar_valid_range_start
                | sym::rustc_layout_scalar_valid_range_end => {
                    self.check_rustc_layout_scalar_valid_range(&attr, span, target)
                }
                sym::allow_internal_unstable => {
                    self.check_allow_internal_unstable(hir_id, &attr, span, target, &attrs)
                }
                sym::debugger_visualizer => self.check_debugger_visualizer(&attr, target),
                sym::rustc_allow_const_fn_unstable => {
                    self.check_rustc_allow_const_fn_unstable(hir_id, &attr, span, target)
                }
                sym::rustc_std_internal_symbol => {
                    self.check_rustc_std_internal_symbol(&attr, span, target)
                }
                sym::naked => self.check_naked(hir_id, attr, span, target),
                sym::rustc_legacy_const_generics => {
                    self.check_rustc_legacy_const_generics(&attr, span, target, item)
                }
                sym::rustc_lint_query_instability => {
                    self.check_rustc_lint_query_instability(&attr, span, target)
                }
                sym::rustc_lint_diagnostics => {
                    self.check_rustc_lint_diagnostics(&attr, span, target)
                }
                sym::rustc_lint_opt_ty => self.check_rustc_lint_opt_ty(&attr, span, target),
                sym::rustc_lint_opt_deny_field_access => {
                    self.check_rustc_lint_opt_deny_field_access(&attr, span, target)
                }
                sym::rustc_clean
                | sym::rustc_dirty
                | sym::rustc_if_this_changed
                | sym::rustc_then_this_would_need => self.check_rustc_dirty_clean(&attr),
                sym::cmse_nonsecure_entry => self.check_cmse_nonsecure_entry(attr, span, target),
                sym::const_trait => self.check_const_trait(attr, span, target),
                sym::must_not_suspend => self.check_must_not_suspend(&attr, span, target),
                sym::must_use => self.check_must_use(hir_id, &attr, span, target),
                sym::rustc_pass_by_value => self.check_pass_by_value(&attr, span, target),
                sym::rustc_allow_incoherent_impl => {
                    self.check_allow_incoherent_impl(&attr, span, target)
                }
                sym::rustc_has_incoherent_inherent_impls => {
                    self.check_has_incoherent_inherent_impls(&attr, span, target)
                }
                sym::rustc_const_unstable
                | sym::rustc_const_stable
                | sym::unstable
                | sym::stable
                | sym::rustc_allowed_through_unstable_modules
                | sym::rustc_promotable => self.check_stability_promotable(&attr, span, target),
                sym::link_ordinal => self.check_link_ordinal(&attr, span, target),
                _ => true,
            };
            is_valid &= attr_is_valid;

            // lint-only checks
            match attr.name_or_empty() {
                sym::cold => self.check_cold(hir_id, attr, span, target),
                sym::link => self.check_link(hir_id, attr, span, target),
                sym::link_name => self.check_link_name(hir_id, attr, span, target),
                sym::link_section => self.check_link_section(hir_id, attr, span, target),
                sym::no_mangle => self.check_no_mangle(hir_id, attr, span, target),
                sym::deprecated => self.check_deprecated(hir_id, attr, span, target),
                sym::macro_use | sym::macro_escape => self.check_macro_use(hir_id, attr, target),
                sym::path => self.check_generic_attr(hir_id, attr, target, &[Target::Mod]),
                sym::plugin_registrar => self.check_plugin_registrar(hir_id, attr, target),
                sym::macro_export => self.check_macro_export(hir_id, attr, target),
                sym::ignore | sym::should_panic | sym::proc_macro_derive => {
                    self.check_generic_attr(hir_id, attr, target, &[Target::Fn])
                }
                sym::automatically_derived => {
                    self.check_generic_attr(hir_id, attr, target, &[Target::Impl])
                }
                sym::no_implicit_prelude => {
                    self.check_generic_attr(hir_id, attr, target, &[Target::Mod])
                }
                sym::rustc_object_lifetime_default => self.check_object_lifetime_default(hir_id),
                _ => {}
            }

            let builtin = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));

            if hir_id != CRATE_HIR_ID {
                if let Some(BuiltinAttribute { type_: AttributeType::CrateLevel, .. }) =
                    attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name))
                {
                    match attr.style {
                        ast::AttrStyle::Outer => self.tcx.emit_spanned_lint(
                            UNUSED_ATTRIBUTES,
                            hir_id,
                            attr.span,
                            errors::OuterCrateLevelAttr,
                        ),
                        ast::AttrStyle::Inner => self.tcx.emit_spanned_lint(
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

        if !is_valid {
            return;
        }

        // FIXME(@lcnr): this doesn't belong here.
        if matches!(
            target,
            Target::Closure
                | Target::Fn
                | Target::Method(_)
                | Target::ForeignFn
                | Target::ForeignStatic
        ) {
            self.tcx.ensure().codegen_fn_attrs(self.tcx.hir().local_def_id(hir_id));
        }

        self.check_repr(attrs, span, target, item, hir_id);
        self.check_used(attrs, target);
    }

    fn inline_attr_str_error_with_macro_def(&self, hir_id: HirId, attr: &Attribute, sym: &str) {
        self.tcx.emit_spanned_lint(
            UNUSED_ATTRIBUTES,
            hir_id,
            attr.span,
            errors::IgnoredAttrWithMacro { sym },
        );
    }

    fn inline_attr_str_error_without_macro_def(&self, hir_id: HirId, attr: &Attribute, sym: &str) {
        self.tcx.emit_spanned_lint(
            UNUSED_ATTRIBUTES,
            hir_id,
            attr.span,
            errors::IgnoredAttr { sym },
        );
    }

    /// Checks if an `#[inline]` is applied to a function or a closure. Returns `true` if valid.
    fn check_inline(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Fn
            | Target::Closure
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
            Target::Method(MethodKind::Trait { body: false }) | Target::ForeignFn => {
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::IgnoredInlineAttrFnProto,
                );
                true
            }
            // FIXME(#65833): We permit associated consts to have an `#[inline]` attribute with
            // just a lint, because we previously erroneously allowed it and some crates used it
            // accidentally, to to be compatible with crates depending on them, we can't throw an
            // error here.
            Target::AssocConst => {
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::IgnoredInlineAttrConstants,
                );
                true
            }
            // FIXME(#80564): Same for fields, arms, and macro defs
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "inline");
                true
            }
            _ => {
                self.tcx.sess.emit_err(errors::InlineNotFnOrClosure {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    /// Checks if a `#[no_coverage]` is applied directly to a function
    fn check_no_coverage(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            // no_coverage on function is fine
            Target::Fn
            | Target::Closure
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,

            // function prototypes can't be covered
            Target::Method(MethodKind::Trait { body: false }) | Target::ForeignFn => {
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::IgnoredNoCoverageFnProto,
                );
                true
            }

            Target::Mod | Target::ForeignMod | Target::Impl | Target::Trait => {
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::IgnoredNoCoveragePropagate,
                );
                true
            }

            Target::Expression | Target::Statement | Target::Arm => {
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::IgnoredNoCoverageFnDefn,
                );
                true
            }

            _ => {
                self.tcx.sess.emit_err(errors::IgnoredNoCoverageNotCoverable {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    fn check_generic_attr(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        target: Target,
        allowed_targets: &[Target],
    ) {
        if !allowed_targets.iter().any(|t| t == &target) {
            let name = attr.name_or_empty();
            let mut i = allowed_targets.iter();
            // Pluralize
            let b = i.next().map_or_else(String::new, |t| t.to_string() + "s");
            let supported_names = i.enumerate().fold(b, |mut b, (i, allowed_target)| {
                if allowed_targets.len() > 2 && i == allowed_targets.len() - 2 {
                    b.push_str(", and ");
                } else if allowed_targets.len() == 2 && i == allowed_targets.len() - 2 {
                    b.push_str(" and ");
                } else {
                    b.push_str(", ");
                }
                // Pluralize
                b.push_str(&(allowed_target.to_string() + "s"));
                b
            });
            self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                lint.build(&format!("`#[{name}]` only has an effect on {}", supported_names))
                    .emit();
            });
        }
    }

    /// Checks if `#[naked]` is applied to a function definition.
    fn check_naked(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[allow_internal_unstable]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "naked");
                true
            }
            _ => {
                self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToFn {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    /// Checks if `#[cmse_nonsecure_entry]` is applied to a function definition.
    fn check_cmse_nonsecure_entry(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
            _ => {
                self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToFn {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    /// Debugging aid for `object_lifetime_default` query.
    fn check_object_lifetime_default(&self, hir_id: HirId) {
        let tcx = self.tcx;
        if let Some(generics) = tcx.hir().get_generics(tcx.hir().local_def_id(hir_id)) {
            for p in generics.params {
                let hir::GenericParamKind::Type { .. } = p.kind else { continue };
                let param_id = tcx.hir().local_def_id(p.hir_id);
                let default = tcx.object_lifetime_default(param_id);
                let repr = match default {
                    ObjectLifetimeDefault::Empty => "BaseDefault".to_owned(),
                    ObjectLifetimeDefault::Static => "'static".to_owned(),
                    ObjectLifetimeDefault::Param(def_id) => tcx.item_name(def_id).to_string(),
                    ObjectLifetimeDefault::Ambiguous => "Ambiguous".to_owned(),
                };
                tcx.sess.span_err(p.span, &repr);
            }
        }
    }

    /// Checks if a `#[track_caller]` is applied to a non-naked function. Returns `true` if valid.
    fn check_track_caller(
        &self,
        hir_id: HirId,
        attr_span: Span,
        attrs: &[Attribute],
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            _ if attrs.iter().any(|attr| attr.has_name(sym::naked)) => {
                self.tcx.sess.emit_err(errors::NakedTrackedCaller { attr_span });
                false
            }
            Target::Fn | Target::Method(..) | Target::ForeignFn | Target::Closure => true,
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[track_caller]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                for attr in attrs {
                    self.inline_attr_str_error_with_macro_def(hir_id, attr, "track_caller");
                }
                true
            }
            _ => {
                self.tcx
                    .sess
                    .emit_err(errors::TrackedCallerWrongLocation { attr_span, defn_span: span });
                false
            }
        }
    }

    /// Checks if the `#[non_exhaustive]` attribute on an `item` is valid. Returns `true` if valid.
    fn check_non_exhaustive(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Struct | Target::Enum | Target::Variant => true,
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[non_exhaustive]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "non_exhaustive");
                true
            }
            _ => {
                self.tcx.sess.emit_err(errors::NonExhaustiveWrongLocation {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    /// Checks if the `#[marker]` attribute on an `item` is valid. Returns `true` if valid.
    fn check_marker(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Trait => true,
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[marker]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "marker");
                true
            }
            _ => {
                self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToTrait {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    /// Checks if the `#[rustc_must_implement_one_of]` attribute on a `target` is valid. Returns `true` if valid.
    fn check_rustc_must_implement_one_of(
        &self,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Trait => true,
            _ => {
                self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToTrait {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    /// Checks if the `#[target_feature]` attribute on `item` is valid. Returns `true` if valid.
    fn check_target_feature(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
            // FIXME: #[target_feature] was previously erroneously allowed on statements and some
            // crates used this, so only emit a warning.
            Target::Statement => {
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::TargetFeatureOnStatement,
                );
                true
            }
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[target_feature]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "target_feature");
                true
            }
            _ => {
                self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToFn {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    /// Checks if the `#[thread_local]` attribute on `item` is valid. Returns `true` if valid.
    fn check_thread_local(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::ForeignStatic | Target::Static => true,
            _ => {
                self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToStatic {
                    attr_span: attr.span,
                    defn_span: span,
                });
                false
            }
        }
    }

    fn doc_attr_str_error(&self, meta: &NestedMetaItem, attr_name: &str) {
        self.tcx.sess.emit_err(errors::DocExpectStr { attr_span: meta.span(), attr_name });
    }

    fn check_doc_alias_value(
        &self,
        meta: &NestedMetaItem,
        doc_alias: Symbol,
        hir_id: HirId,
        target: Target,
        is_list: bool,
        aliases: &mut FxHashMap<String, Span>,
    ) -> bool {
        let tcx = self.tcx;
        let span = meta.name_value_literal_span().unwrap_or_else(|| meta.span());
        let attr_str =
            &format!("`#[doc(alias{})]`", if is_list { "(\"...\")" } else { " = \"...\"" });
        if doc_alias == kw::Empty {
            tcx.sess.emit_err(errors::DocAliasEmpty { span, attr_str });
            return false;
        }

        let doc_alias_str = doc_alias.as_str();
        if let Some(c) = doc_alias_str
            .chars()
            .find(|&c| c == '"' || c == '\'' || (c.is_whitespace() && c != ' '))
        {
            tcx.sess.emit_err(errors::DocAliasBadChar { span, attr_str, char_: c });
            return false;
        }
        if doc_alias_str.starts_with(' ') || doc_alias_str.ends_with(' ') {
            tcx.sess.emit_err(errors::DocAliasStartEnd { span, attr_str });
            return false;
        }

        let span = meta.span();
        if let Some(location) = match target {
            Target::AssocTy => {
                let parent_hir_id = self.tcx.hir().get_parent_item(hir_id);
                let containing_item = self.tcx.hir().expect_item(parent_hir_id);
                if Target::from_item(containing_item) == Target::Impl {
                    Some("type alias in implementation block")
                } else {
                    None
                }
            }
            Target::AssocConst => {
                let parent_hir_id = self.tcx.hir().get_parent_item(hir_id);
                let containing_item = self.tcx.hir().expect_item(parent_hir_id);
                // We can't link to trait impl's consts.
                let err = "associated constant in trait implementation block";
                match containing_item.kind {
                    ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) => Some(err),
                    _ => None,
                }
            }
            // we check the validity of params elsewhere
            Target::Param => return false,
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
            | Target::OpaqueTy
            | Target::ImplTraitPlaceholder
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
            tcx.sess.emit_err(errors::DocAliasBadLocation { span, attr_str, location });
            return false;
        }
        let item_name = self.tcx.hir().name(hir_id);
        if item_name == doc_alias {
            tcx.sess.emit_err(errors::DocAliasNotAnAlias { span, attr_str });
            return false;
        }
        if let Err(entry) = aliases.try_insert(doc_alias_str.to_owned(), span) {
            self.tcx.emit_spanned_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                span,
                errors::DocAliasDuplicated { first_defn: *entry.entry.get() },
            );
        }
        true
    }

    fn check_doc_alias(
        &self,
        meta: &NestedMetaItem,
        hir_id: HirId,
        target: Target,
        aliases: &mut FxHashMap<String, Span>,
    ) -> bool {
        if let Some(values) = meta.meta_item_list() {
            let mut errors = 0;
            for v in values {
                match v.literal() {
                    Some(l) => match l.kind {
                        LitKind::Str(s, _) => {
                            if !self.check_doc_alias_value(v, s, hir_id, target, true, aliases) {
                                errors += 1;
                            }
                        }
                        _ => {
                            self.tcx
                                .sess
                                .emit_err(errors::DocAliasNotStringLiteral { span: v.span() });
                            errors += 1;
                        }
                    },
                    None => {
                        self.tcx.sess.emit_err(errors::DocAliasNotStringLiteral { span: v.span() });
                        errors += 1;
                    }
                }
            }
            errors == 0
        } else if let Some(doc_alias) = meta.value_str() {
            self.check_doc_alias_value(meta, doc_alias, hir_id, target, false, aliases)
        } else {
            self.tcx.sess.emit_err(errors::DocAliasMalformed { span: meta.span() });
            false
        }
    }

    fn check_doc_keyword(&self, meta: &NestedMetaItem, hir_id: HirId) -> bool {
        let doc_keyword = meta.value_str().unwrap_or(kw::Empty);
        if doc_keyword == kw::Empty {
            self.doc_attr_str_error(meta, "keyword");
            return false;
        }
        match self.tcx.hir().find(hir_id).and_then(|node| match node {
            hir::Node::Item(item) => Some(&item.kind),
            _ => None,
        }) {
            Some(ItemKind::Mod(ref module)) => {
                if !module.item_ids.is_empty() {
                    self.tcx.sess.emit_err(errors::DocKeywordEmptyMod { span: meta.span() });
                    return false;
                }
            }
            _ => {
                self.tcx.sess.emit_err(errors::DocKeywordNotMod { span: meta.span() });
                return false;
            }
        }
        if !rustc_lexer::is_ident(doc_keyword.as_str()) {
            self.tcx.sess.emit_err(errors::DocKeywordInvalidIdent {
                span: meta.name_value_literal_span().unwrap_or_else(|| meta.span()),
                doc_keyword,
            });
            return false;
        }
        true
    }

    fn check_doc_fake_variadic(&self, meta: &NestedMetaItem, hir_id: HirId) -> bool {
        match self.tcx.hir().find(hir_id).and_then(|node| match node {
            hir::Node::Item(item) => Some(&item.kind),
            _ => None,
        }) {
            Some(ItemKind::Impl(ref i)) => {
                let is_valid = matches!(&i.self_ty.kind, hir::TyKind::Tup([_]))
                    || if let hir::TyKind::BareFn(bare_fn_ty) = &i.self_ty.kind {
                        bare_fn_ty.decl.inputs.len() == 1
                    } else {
                        false
                    };
                if !is_valid {
                    self.tcx.sess.emit_err(errors::DocFakeVariadicNotValid { span: meta.span() });
                    return false;
                }
            }
            _ => {
                self.tcx.sess.emit_err(errors::DocKeywordOnlyImpl { span: meta.span() });
                return false;
            }
        }
        true
    }

    /// Checks `#[doc(inline)]`/`#[doc(no_inline)]` attributes. Returns `true` if valid.
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
        meta: &NestedMetaItem,
        hir_id: HirId,
        target: Target,
        specified_inline: &mut Option<(bool, Span)>,
    ) -> bool {
        if target == Target::Use || target == Target::ExternCrate {
            let do_inline = meta.name_or_empty() == sym::inline;
            if let Some((prev_inline, prev_span)) = *specified_inline {
                if do_inline != prev_inline {
                    let mut spans = MultiSpan::from_spans(vec![prev_span, meta.span()]);
                    spans.push_span_label(prev_span, fluent::passes::doc_inline_conflict_first);
                    spans.push_span_label(meta.span(), fluent::passes::doc_inline_conflict_second);
                    self.tcx.sess.emit_err(errors::DocKeywordConflict { spans });
                    return false;
                }
                true
            } else {
                *specified_inline = Some((do_inline, meta.span()));
                true
            }
        } else {
            self.tcx.emit_spanned_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                errors::DocInlineOnlyUse {
                    attr_span: meta.span(),
                    item_span: (attr.style == AttrStyle::Outer)
                        .then(|| self.tcx.hir().span(hir_id)),
                },
            );
            false
        }
    }

    /// Checks that an attribute is *not* used at the crate level. Returns `true` if valid.
    fn check_attr_not_crate_level(
        &self,
        meta: &NestedMetaItem,
        hir_id: HirId,
        attr_name: &str,
    ) -> bool {
        if CRATE_HIR_ID == hir_id {
            self.tcx.sess.emit_err(errors::DocAttrNotCrateLevel { span: meta.span(), attr_name });
            return false;
        }
        true
    }

    /// Checks that an attribute is used at the crate level. Returns `true` if valid.
    fn check_attr_crate_level(
        &self,
        attr: &Attribute,
        meta: &NestedMetaItem,
        hir_id: HirId,
    ) -> bool {
        if hir_id != CRATE_HIR_ID {
            self.tcx.struct_span_lint_hir(INVALID_DOC_ATTRIBUTES, hir_id, meta.span(), |lint| {
                let mut err = lint.build(fluent::passes::attr_crate_level);
                if attr.style == AttrStyle::Outer
                    && self.tcx.hir().get_parent_item(hir_id) == CRATE_DEF_ID
                {
                    if let Ok(mut src) = self.tcx.sess.source_map().span_to_snippet(attr.span) {
                        src.insert(1, '!');
                        err.span_suggestion_verbose(
                            attr.span,
                            fluent::passes::suggestion,
                            src,
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.span_help(attr.span, fluent::passes::help);
                    }
                }
                err.note(fluent::passes::note).emit();
            });
            return false;
        }
        true
    }

    /// Checks that `doc(test(...))` attribute contains only valid attributes. Returns `true` if
    /// valid.
    fn check_test_attr(&self, meta: &NestedMetaItem, hir_id: HirId) -> bool {
        let mut is_valid = true;
        if let Some(metas) = meta.meta_item_list() {
            for i_meta in metas {
                match i_meta.name_or_empty() {
                    sym::attr | sym::no_crate_inject => {}
                    _ => {
                        self.tcx.emit_spanned_lint(
                            INVALID_DOC_ATTRIBUTES,
                            hir_id,
                            i_meta.span(),
                            errors::DocTestUnknown {
                                path: rustc_ast_pretty::pprust::path_to_string(
                                    &i_meta.meta_item().unwrap().path,
                                ),
                            },
                        );
                        is_valid = false;
                    }
                }
            }
        } else {
            self.tcx.emit_spanned_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                errors::DocTestTakesList,
            );
            is_valid = false;
        }
        is_valid
    }

    /// Runs various checks on `#[doc]` attributes. Returns `true` if valid.
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
    ) -> bool {
        let mut is_valid = true;

        if let Some(mi) = attr.meta() && let Some(list) = mi.meta_item_list() {
            for meta in list {
                if let Some(i_meta) = meta.meta_item() {
                    match i_meta.name_or_empty() {
                        sym::alias
                            if !self.check_attr_not_crate_level(meta, hir_id, "alias")
                                || !self.check_doc_alias(meta, hir_id, target, aliases) =>
                        {
                            is_valid = false
                        }

                        sym::keyword
                            if !self.check_attr_not_crate_level(meta, hir_id, "keyword")
                                || !self.check_doc_keyword(meta, hir_id) =>
                        {
                            is_valid = false
                        }

                        sym::fake_variadic
                            if !self.check_attr_not_crate_level(meta, hir_id, "fake_variadic")
                                || !self.check_doc_fake_variadic(meta, hir_id) =>
                        {
                            is_valid = false
                        }

                        sym::html_favicon_url
                        | sym::html_logo_url
                        | sym::html_playground_url
                        | sym::issue_tracker_base_url
                        | sym::html_root_url
                        | sym::html_no_source
                        | sym::test
                            if !self.check_attr_crate_level(attr, meta, hir_id) =>
                        {
                            is_valid = false;
                        }

                        sym::inline | sym::no_inline
                            if !self.check_doc_inline(
                                attr,
                                meta,
                                hir_id,
                                target,
                                specified_inline,
                            ) =>
                        {
                            is_valid = false;
                        }

                        // no_default_passes: deprecated
                        // passes: deprecated
                        // plugins: removed, but rustdoc warns about it itself
                        sym::alias
                        | sym::cfg
                        | sym::cfg_hide
                        | sym::hidden
                        | sym::html_favicon_url
                        | sym::html_logo_url
                        | sym::html_no_source
                        | sym::html_playground_url
                        | sym::html_root_url
                        | sym::inline
                        | sym::issue_tracker_base_url
                        | sym::keyword
                        | sym::masked
                        | sym::no_default_passes
                        | sym::no_inline
                        | sym::notable_trait
                        | sym::passes
                        | sym::plugins
                        | sym::fake_variadic => {}

                        sym::test => {
                            if !self.check_test_attr(meta, hir_id) {
                                is_valid = false;
                            }
                        }

                        sym::primitive => {
                            if !self.tcx.features().rustdoc_internals {
                                self.tcx.emit_spanned_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocPrimitive,
                                );
                            }
                        }

                        _ => {
                            let path = rustc_ast_pretty::pprust::path_to_string(&i_meta.path);
                            if i_meta.has_name(sym::spotlight) {
                                self.tcx.emit_spanned_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownSpotlight {
                                        path,
                                        span: i_meta.span
                                    }
                                );
                            } else if i_meta.has_name(sym::include) &&
                                    let Some(value) = i_meta.value_str() {
                                let applicability = if list.len() == 1 {
                                    Applicability::MachineApplicable
                                } else {
                                    Applicability::MaybeIncorrect
                                };
                                // If there are multiple attributes, the suggestion would suggest
                                // deleting all of them, which is incorrect.
                                self.tcx.emit_spanned_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownInclude {
                                        path,
                                        value: value.to_string(),
                                        inner: (attr.style == AttrStyle::Inner)
                                            .then_some("!")
                                            .unwrap_or(""),
                                        sugg: (attr.meta().unwrap().span, applicability),
                                    }
                                );
                            } else {
                                self.tcx.emit_spanned_lint(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    errors::DocTestUnknownAny { path }
                                );
                            }
                            is_valid = false;
                        }
                    }
                } else {
                    self.tcx.emit_spanned_lint(
                        INVALID_DOC_ATTRIBUTES,
                        hir_id,
                        meta.span(),
                        errors::DocInvalid,
                    );
                    is_valid = false;
                }
            }
        }

        is_valid
    }

    /// Warns against some misuses of `#[pass_by_value]`
    fn check_pass_by_value(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Struct | Target::Enum | Target::TyAlias => true,
            _ => {
                self.tcx.sess.emit_err(errors::PassByValue { attr_span: attr.span, span });
                false
            }
        }
    }

    fn check_allow_incoherent_impl(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Method(MethodKind::Inherent) => true,
            _ => {
                self.tcx.sess.emit_err(errors::AllowIncoherentImpl { attr_span: attr.span, span });
                false
            }
        }
    }

    fn check_has_incoherent_inherent_impls(
        &self,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Trait | Target::Struct | Target::Enum | Target::Union | Target::ForeignTy => {
                true
            }
            _ => {
                self.tcx
                    .sess
                    .emit_err(errors::HasIncoherentInherentImpl { attr_span: attr.span, span });
                false
            }
        }
    }

    /// Warns against some misuses of `#[must_use]`
    fn check_must_use(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) -> bool {
        let node = self.tcx.hir().get(hir_id);
        if let Some(kind) = node.fn_kind() && let rustc_hir::IsAsync::Async = kind.asyncness() {
            self.tcx.emit_spanned_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr.span,
                errors::MustUseAsync { span }
            );
        }

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
                | Target::OpaqueTy
                | Target::Enum
                | Target::Impl
                | Target::Expression
                | Target::Arm
                | Target::AssocConst
                | Target::AssocTy => "an",
                _ => "a",
            };

            self.tcx.emit_spanned_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr.span,
                errors::MustUseNoEffect { article, target },
            );
        }

        // For now, its always valid
        true
    }

    /// Checks if `#[must_not_suspend]` is applied to a function. Returns `true` if valid.
    fn check_must_not_suspend(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Struct | Target::Enum | Target::Union | Target::Trait => true,
            _ => {
                self.tcx.sess.emit_err(errors::MustNotSuspend { attr_span: attr.span, span });
                false
            }
        }
    }

    /// Checks if `#[cold]` is applied to a non-function. Returns `true` if valid.
    fn check_cold(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Fn | Target::Method(..) | Target::ForeignFn | Target::Closure => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[cold]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "cold");
            }
            _ => {
                // FIXME: #[cold] was previously allowed on non-functions and some crates used
                // this, so only emit a warning.
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::Cold { span },
                );
            }
        }
    }

    /// Checks if `#[link]` is applied to an item other than a foreign module.
    fn check_link(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        if target == Target::ForeignMod
            && let hir::Node::Item(item) = self.tcx.hir().get(hir_id)
            && let Item { kind: ItemKind::ForeignMod { abi, .. }, .. } = item
            && !matches!(abi, Abi::Rust | Abi::RustIntrinsic | Abi::PlatformIntrinsic)
        {
            return;
        }

        self.tcx.emit_spanned_lint(
            UNUSED_ATTRIBUTES,
            hir_id,
            attr.span,
            errors::Link { span: (target != Target::ForeignMod).then_some(span) },
        );
    }

    /// Checks if `#[link_name]` is applied to an item other than a foreign function or static.
    fn check_link_name(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::ForeignFn | Target::ForeignStatic => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[link_name]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "link_name");
            }
            _ => {
                // FIXME: #[cold] was previously allowed on non-functions/statics and some crates
                // used this, so only emit a warning.
                let attr_span = matches!(target, Target::ForeignMod).then_some(attr.span);
                if let Some(s) = attr.value_str() {
                    self.tcx.emit_spanned_lint(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        attr.span,
                        errors::LinkName { span, attr_span, value: s.as_str() },
                    );
                } else {
                    self.tcx.emit_spanned_lint(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        attr.span,
                        errors::LinkName { span, attr_span, value: "..." },
                    );
                };
            }
        }
    }

    /// Checks if `#[no_link]` is applied to an `extern crate`. Returns `true` if valid.
    fn check_no_link(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::ExternCrate => true,
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[no_link]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "no_link");
                true
            }
            _ => {
                self.tcx.sess.emit_err(errors::NoLink { attr_span: attr.span, span });
                false
            }
        }
    }

    fn is_impl_item(&self, hir_id: HirId) -> bool {
        matches!(self.tcx.hir().get(hir_id), hir::Node::ImplItem(..))
    }

    /// Checks if `#[export_name]` is applied to a function or static. Returns `true` if valid.
    fn check_export_name(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Static | Target::Fn => true,
            Target::Method(..) if self.is_impl_item(hir_id) => true,
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[export_name]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "export_name");
                true
            }
            _ => {
                self.tcx.sess.emit_err(errors::ExportName { attr_span: attr.span, span });
                false
            }
        }
    }

    fn check_rustc_layout_scalar_valid_range(
        &self,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        if target != Target::Struct {
            self.tcx.sess.emit_err(errors::RustcLayoutScalarValidRangeNotStruct {
                attr_span: attr.span,
                span,
            });
            return false;
        }

        let Some(list) = attr.meta_item_list() else {
            return false;
        };

        if matches!(&list[..], &[NestedMetaItem::Literal(Lit { kind: LitKind::Int(..), .. })]) {
            true
        } else {
            self.tcx.sess.emit_err(errors::RustcLayoutScalarValidRangeArg { attr_span: attr.span });
            false
        }
    }

    /// Checks if `#[rustc_legacy_const_generics]` is applied to a function and has a valid argument.
    fn check_rustc_legacy_const_generics(
        &self,
        attr: &Attribute,
        span: Span,
        target: Target,
        item: Option<ItemLike<'_>>,
    ) -> bool {
        let is_function = matches!(target, Target::Fn);
        if !is_function {
            self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToFn {
                attr_span: attr.span,
                defn_span: span,
            });
            return false;
        }

        let Some(list) = attr.meta_item_list() else {
            // The attribute form is validated on AST.
            return false;
        };

        let Some(ItemLike::Item(Item {
            kind: ItemKind::Fn(FnSig { decl, .. }, generics, _),
            ..
        }))  = item else {
            bug!("should be a function item");
        };

        for param in generics.params {
            match param.kind {
                hir::GenericParamKind::Const { .. } => {}
                _ => {
                    self.tcx.sess.emit_err(errors::RustcLegacyConstGenericsOnly {
                        attr_span: attr.span,
                        param_span: param.span,
                    });
                    return false;
                }
            }
        }

        if list.len() != generics.params.len() {
            self.tcx.sess.emit_err(errors::RustcLegacyConstGenericsIndex {
                attr_span: attr.span,
                generics_span: generics.span,
            });
            return false;
        }

        let arg_count = decl.inputs.len() as u128 + generics.params.len() as u128;
        let mut invalid_args = vec![];
        for meta in list {
            if let Some(LitKind::Int(val, _)) = meta.literal().map(|lit| &lit.kind) {
                if *val >= arg_count {
                    let span = meta.span();
                    self.tcx.sess.emit_err(errors::RustcLegacyConstGenericsIndexExceed {
                        span,
                        arg_count: arg_count as usize,
                    });
                    return false;
                }
            } else {
                invalid_args.push(meta.span());
            }
        }

        if !invalid_args.is_empty() {
            self.tcx.sess.emit_err(errors::RustcLegacyConstGenericsIndexNegative { invalid_args });
            false
        } else {
            true
        }
    }

    /// Helper function for checking that the provided attribute is only applied to a function or
    /// method.
    fn check_applied_to_fn_or_method(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        let is_function = matches!(target, Target::Fn | Target::Method(..));
        if !is_function {
            self.tcx.sess.emit_err(errors::AttrShouldBeAppliedToFn {
                attr_span: attr.span,
                defn_span: span,
            });
            false
        } else {
            true
        }
    }

    /// Checks that the `#[rustc_lint_query_instability]` attribute is only applied to a function
    /// or method.
    fn check_rustc_lint_query_instability(
        &self,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        self.check_applied_to_fn_or_method(attr, span, target)
    }

    /// Checks that the `#[rustc_lint_diagnostics]` attribute is only applied to a function or
    /// method.
    fn check_rustc_lint_diagnostics(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        self.check_applied_to_fn_or_method(attr, span, target)
    }

    /// Checks that the `#[rustc_lint_opt_ty]` attribute is only applied to a struct.
    fn check_rustc_lint_opt_ty(&self, attr: &Attribute, span: Span, target: Target) -> bool {
        match target {
            Target::Struct => true,
            _ => {
                self.tcx.sess.emit_err(errors::RustcLintOptTy { attr_span: attr.span, span });
                false
            }
        }
    }

    /// Checks that the `#[rustc_lint_opt_deny_field_access]` attribute is only applied to a field.
    fn check_rustc_lint_opt_deny_field_access(
        &self,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Field => true,
            _ => {
                self.tcx
                    .sess
                    .emit_err(errors::RustcLintOptDenyFieldAccess { attr_span: attr.span, span });
                false
            }
        }
    }

    /// Checks that the dep-graph debugging attributes are only present when the query-dep-graph
    /// option is passed to the compiler.
    fn check_rustc_dirty_clean(&self, attr: &Attribute) -> bool {
        if self.tcx.sess.opts.unstable_opts.query_dep_graph {
            true
        } else {
            self.tcx.sess.emit_err(errors::RustcDirtyClean { span: attr.span });
            false
        }
    }

    /// Checks if `#[link_section]` is applied to a function or static.
    fn check_link_section(&self, hir_id: HirId, attr: &Attribute, span: Span, target: Target) {
        match target {
            Target::Static | Target::Fn | Target::Method(..) => {}
            // FIXME(#80564): We permit struct fields, match arms and macro defs to have an
            // `#[link_section]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "link_section");
            }
            _ => {
                // FIXME: #[link_section] was previously allowed on non-functions/statics and some
                // crates used this, so only emit a warning.
                self.tcx.emit_spanned_lint(
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
            // erroneously allowed it and some crates used it accidentally, to to be compatible
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
                self.tcx.emit_spanned_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr.span,
                    errors::NoMangleForeign { span, attr_span: attr.span, foreign_item_kind },
                );
            }
            _ => {
                // FIXME: #[no_mangle] was previously allowed on non-functions/statics and some
                // crates used this, so only emit a warning.
                self.tcx.emit_spanned_lint(
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
        let mut is_c = false;
        let mut is_simd = false;
        let mut is_transparent = false;

        for hint in &hints {
            if !hint.is_meta_item() {
                self.tcx.sess.emit_err(errors::ReprIdent { span: hint.span() });
                continue;
            }

            let (article, allowed_targets) = match hint.name_or_empty() {
                sym::C => {
                    is_c = true;
                    match target {
                        Target::Struct | Target::Union | Target::Enum => continue,
                        _ => ("a", "struct, enum, or union"),
                    }
                }
                sym::align => {
                    if let (Target::Fn, false) = (target, self.tcx.features().fn_align) {
                        feature_err(
                            &self.tcx.sess.parse_sess,
                            sym::fn_align,
                            hint.span(),
                            "`repr(align)` attributes on functions are unstable",
                        )
                        .emit();
                    }

                    match target {
                        Target::Struct | Target::Union | Target::Enum | Target::Fn => continue,
                        _ => ("a", "struct, enum, function, or union"),
                    }
                }
                sym::packed => {
                    if target != Target::Struct && target != Target::Union {
                        ("a", "struct or union")
                    } else {
                        continue;
                    }
                }
                sym::simd => {
                    is_simd = true;
                    if target != Target::Struct {
                        ("a", "struct")
                    } else {
                        continue;
                    }
                }
                sym::transparent => {
                    is_transparent = true;
                    match target {
                        Target::Struct | Target::Union | Target::Enum => continue,
                        _ => ("a", "struct, enum, or union"),
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
                        ("an", "enum")
                    } else {
                        continue;
                    }
                }
                _ => {
                    struct_span_err!(
                        self.tcx.sess,
                        hint.span(),
                        E0552,
                        "unrecognized representation hint"
                    )
                    .help("valid reprs are `C`, `align`, `packed`, `transparent`, `simd`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `i128`, `u128`, `isize`, `usize`")
                    .emit();

                    continue;
                }
            };

            struct_span_err!(
                self.tcx.sess,
                hint.span(),
                E0517,
                "{}",
                &format!("attribute should be applied to {article} {allowed_targets}")
            )
            .span_label(span, &format!("not {article} {allowed_targets}"))
            .emit();
        }

        // Just point at all repr hints if there are any incompatibilities.
        // This is not ideal, but tracking precisely which ones are at fault is a huge hassle.
        let hint_spans = hints.iter().map(|hint| hint.span());

        // Error on repr(transparent, <anything else>).
        if is_transparent && hints.len() > 1 {
            let hint_spans: Vec<_> = hint_spans.clone().collect();
            struct_span_err!(
                self.tcx.sess,
                hint_spans,
                E0692,
                "transparent {} cannot have other repr hints",
                target
            )
            .emit();
        }
        // Warn on repr(u8, u16), repr(C, simd), and c-like-enum-repr(C, u8)
        if (int_reprs > 1)
            || (is_simd && is_c)
            || (int_reprs == 1
                && is_c
                && item.map_or(false, |item| {
                    if let ItemLike::Item(item) = item {
                        return is_c_like_enum(item);
                    }
                    return false;
                }))
        {
            self.tcx.emit_spanned_lint(
                CONFLICTING_REPR_HINTS,
                hir_id,
                hint_spans.collect::<Vec<Span>>(),
                errors::ReprConflicting,
            );
        }
    }

    fn check_used(&self, attrs: &[Attribute], target: Target) {
        let mut used_linker_span = None;
        let mut used_compiler_span = None;
        for attr in attrs.iter().filter(|attr| attr.has_name(sym::used)) {
            if target != Target::Static {
                self.tcx.sess.emit_err(errors::UsedStatic { span: attr.span });
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
                    // This error case is handled in rustc_typeck::collect.
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
                .sess
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
    ) -> bool {
        debug!("Checking target: {:?}", target);
        match target {
            Target::Fn => {
                for attr in attrs {
                    if self.tcx.sess.is_proc_macro_attr(attr) {
                        debug!("Is proc macro attr");
                        return true;
                    }
                }
                debug!("Is not proc macro attr");
                false
            }
            Target::MacroDef => true,
            // FIXME(#80564): We permit struct fields and match arms to have an
            // `#[allow_internal_unstable]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm => {
                self.inline_attr_str_error_without_macro_def(
                    hir_id,
                    attr,
                    "allow_internal_unstable",
                );
                true
            }
            _ => {
                self.tcx
                    .sess
                    .emit_err(errors::AllowInternalUnstable { attr_span: attr.span, span });
                false
            }
        }
    }

    /// Checks if the items on the `#[debugger_visualizer]` attribute are valid.
    fn check_debugger_visualizer(&self, attr: &Attribute, target: Target) -> bool {
        match target {
            Target::Mod => {}
            _ => {
                self.tcx.sess.emit_err(errors::DebugVisualizerPlacement { span: attr.span });
                return false;
            }
        }

        let Some(hints) = attr.meta_item_list() else {
            self.tcx.sess.emit_err(errors::DebugVisualizerInvalid { span: attr.span });
            return false;
        };

        let hint = match hints.len() {
            1 => &hints[0],
            _ => {
                self.tcx.sess.emit_err(errors::DebugVisualizerInvalid { span: attr.span });
                return false;
            }
        };

        let Some(meta_item) = hint.meta_item() else {
            self.tcx.sess.emit_err(errors::DebugVisualizerInvalid { span: attr.span });
            return false;
        };

        let visualizer_path = match (meta_item.name_or_empty(), meta_item.value_str()) {
            (sym::natvis_file, Some(value)) => value,
            (sym::gdb_script_file, Some(value)) => value,
            (_, _) => {
                self.tcx.sess.emit_err(errors::DebugVisualizerInvalid { span: meta_item.span });
                return false;
            }
        };

        let file =
            match resolve_path(&self.tcx.sess.parse_sess, visualizer_path.as_str(), attr.span) {
                Ok(file) => file,
                Err(mut err) => {
                    err.emit();
                    return false;
                }
            };

        match std::fs::File::open(&file) {
            Ok(_) => true,
            Err(err) => {
                self.tcx
                    .sess
                    .struct_span_err(
                        meta_item.span,
                        &format!("couldn't read {}: {}", file.display(), err),
                    )
                    .emit();
                false
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
    ) -> bool {
        match target {
            Target::Fn | Target::Method(_)
                if self.tcx.is_const_fn_raw(self.tcx.hir().local_def_id(hir_id).to_def_id()) =>
            {
                true
            }
            // FIXME(#80564): We permit struct fields and match arms to have an
            // `#[allow_internal_unstable]` attribute with just a lint, because we previously
            // erroneously allowed it and some crates used it accidentally, to to be compatible
            // with crates depending on them, we can't throw an error here.
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "allow_internal_unstable");
                true
            }
            _ => {
                self.tcx
                    .sess
                    .emit_err(errors::RustcAllowConstFnUnstable { attr_span: attr.span, span });
                false
            }
        }
    }

    fn check_rustc_std_internal_symbol(
        &self,
        attr: &Attribute,
        span: Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Fn | Target::Static => true,
            _ => {
                self.tcx
                    .sess
                    .emit_err(errors::RustcStdInternalSymbol { attr_span: attr.span, span });
                false
            }
        }
    }

    /// `#[const_trait]` only applies to traits.
    fn check_const_trait(&self, attr: &Attribute, _span: Span, target: Target) -> bool {
        match target {
            Target::Trait => true,
            _ => {
                self.tcx.sess.emit_err(errors::ConstTrait { attr_span: attr.span });
                false
            }
        }
    }

    fn check_stability_promotable(&self, attr: &Attribute, _span: Span, target: Target) -> bool {
        match target {
            Target::Expression => {
                self.tcx.sess.emit_err(errors::StabilityPromotable { attr_span: attr.span });
                false
            }
            _ => true,
        }
    }

    fn check_link_ordinal(&self, attr: &Attribute, _span: Span, target: Target) -> bool {
        match target {
            Target::ForeignFn | Target::ForeignStatic => true,
            _ => {
                self.tcx.sess.emit_err(errors::LinkOrdinal { attr_span: attr.span });
                false
            }
        }
    }

    fn check_deprecated(&self, hir_id: HirId, attr: &Attribute, _span: Span, target: Target) {
        match target {
            Target::Closure | Target::Expression | Target::Statement | Target::Arm => {
                self.tcx.emit_spanned_lint(
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
                self.tcx.emit_spanned_lint(
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
            self.tcx.emit_spanned_lint(UNUSED_ATTRIBUTES, hir_id, attr.span, errors::MacroExport);
        }
    }

    fn check_plugin_registrar(&self, hir_id: HirId, attr: &Attribute, target: Target) {
        if target != Target::Fn {
            self.tcx.emit_spanned_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr.span,
                errors::PluginRegistrar,
            );
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
        ) && attr.meta_item_list().map_or(false, |list| list.is_empty())
        {
            errors::UnusedNote::EmptyList { name: attr.name_or_empty() }
        } else if matches!(
                attr.name_or_empty(),
                sym::allow | sym::warn | sym::deny | sym::forbid | sym::expect
            ) && let Some(meta) = attr.meta_item_list()
            && meta.len() == 1
            && let Some(item) = meta[0].meta_item()
            && let MetaItemKind::NameValue(_) = &item.kind
            && item.path == sym::reason
        {
            errors::UnusedNote::NoLints { name: attr.name_or_empty() }
        } else if attr.name_or_empty() == sym::default_method_body_is_const {
            errors::UnusedNote::DefaultMethodBodyConst
        } else {
            return;
        };

        self.tcx.emit_spanned_lint(
            UNUSED_ATTRIBUTES,
            hir_id,
            attr.span,
            errors::Unused { attr_span: attr.span, note },
        );
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
        if let ItemKind::Macro(ref macro_def, _) = item.kind {
            let def_id = item.def_id.to_def_id();
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
        if let hir::StmtKind::Local(ref l) = stmt.kind {
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
        self.check_attributes(variant.id, variant.span, Target::Variant, None);
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
        sym::unix_sigpipe,
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
                    let mut err = tcx.sess.struct_span_err(
                        attr.span,
                        &format!(
                            "`{}` attribute cannot be used at crate level",
                            attr_to_check.to_ident_string()
                        ),
                    );
                    // Only emit an error with a suggestion if we can create a
                    // string out of the attribute span
                    if let Ok(src) = tcx.sess.source_map().span_to_snippet(attr.span) {
                        let replacement = src.replace("#!", "#");
                        err.span_suggestion_verbose(
                            attr.span,
                            "perhaps you meant to use an outer attribute",
                            replacement,
                            rustc_errors::Applicability::MachineApplicable,
                        );
                    }
                    err.emit();
                }
            }
        }
    }
}

fn check_non_exported_macro_for_invalid_attrs(tcx: TyCtxt<'_>, item: &Item<'_>) {
    let attrs = tcx.hir().attrs(item.hir_id());

    for attr in attrs {
        if attr.has_name(sym::inline) {
            tcx.sess.emit_err(errors::NonExportedMacroInvalidAttrs { attr_span: attr.span });
        }
    }
}

fn check_mod_attrs(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    let check_attr_visitor = &mut CheckAttrVisitor { tcx };
    tcx.hir().visit_item_likes_in_module(module_def_id, check_attr_visitor);
    if module_def_id.is_top_level_module() {
        check_attr_visitor.check_attributes(CRATE_HIR_ID, DUMMY_SP, Target::Mod, None);
        check_invalid_crate_level_attr(tcx, tcx.hir().krate_attrs());
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
                    tcx.emit_spanned_lint(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        this,
                        errors::UnusedDuplicate {
                            this,
                            other,
                            warning: matches!(
                                duplicates,
                                FutureWarnFollowing | FutureWarnPreceding
                            )
                            .then_some(()),
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
                tcx.sess.emit_err(errors::UnusedMultiple {
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

//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use rustc_middle::hir::map::Map;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;

use rustc_ast::{ast, AttrStyle, Attribute, Lit, LitKind, NestedMetaItem};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{pluralize, struct_span_err, Applicability};
use rustc_feature::{AttributeDuplicates, AttributeType, BuiltinAttribute, BUILTIN_ATTRIBUTE_MAP};
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{self, FnSig, ForeignItem, HirId, Item, ItemKind, TraitItem, CRATE_HIR_ID};
use rustc_hir::{MethodKind, Target};
use rustc_session::lint::builtin::{
    CONFLICTING_REPR_HINTS, INVALID_DOC_ATTRIBUTES, UNUSED_ATTRIBUTES,
};
use rustc_session::parse::feature_err;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{MultiSpan, Span, DUMMY_SP};
use std::collections::hash_map::Entry;

pub(crate) fn target_from_impl_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_item: &hir::ImplItem<'_>,
) -> Target {
    match impl_item.kind {
        hir::ImplItemKind::Const(..) => Target::AssocConst,
        hir::ImplItemKind::Fn(..) => {
            let parent_hir_id = tcx.hir().get_parent_item(impl_item.hir_id()).expect_owner();
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
    ForeignItem(&'tcx ForeignItem<'tcx>),
}

struct CheckAttrVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl CheckAttrVisitor<'_> {
    /// Checks any attribute.
    fn check_attributes(
        &self,
        hir_id: HirId,
        span: &Span,
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
                sym::non_exhaustive => self.check_non_exhaustive(hir_id, attr, span, target),
                sym::marker => self.check_marker(hir_id, attr, span, target),
                sym::target_feature => self.check_target_feature(hir_id, attr, span, target),
                sym::track_caller => {
                    self.check_track_caller(hir_id, &attr.span, attrs, span, target)
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
                sym::rustc_allow_const_fn_unstable => {
                    self.check_rustc_allow_const_fn_unstable(hir_id, &attr, span, target)
                }
                sym::naked => self.check_naked(hir_id, attr, span, target),
                sym::rustc_legacy_const_generics => {
                    self.check_rustc_legacy_const_generics(&attr, span, target, item)
                }
                sym::rustc_clean
                | sym::rustc_dirty
                | sym::rustc_if_this_changed
                | sym::rustc_then_this_would_need => self.check_rustc_dirty_clean(&attr),
                sym::cmse_nonsecure_entry => self.check_cmse_nonsecure_entry(attr, span, target),
                sym::default_method_body_is_const => {
                    self.check_default_method_body_is_const(attr, span, target)
                }
                sym::must_not_suspend => self.check_must_not_suspend(&attr, span, target),
                sym::must_use => self.check_must_use(hir_id, &attr, span, target),
                sym::rustc_const_unstable
                | sym::rustc_const_stable
                | sym::unstable
                | sym::stable
                | sym::rustc_promotable => self.check_stability_promotable(&attr, span, target),
                _ => true,
            };
            is_valid &= attr_is_valid;

            // lint-only checks
            match attr.name_or_empty() {
                sym::cold => self.check_cold(hir_id, attr, span, target),
                sym::link_name => self.check_link_name(hir_id, attr, span, target),
                sym::link_section => self.check_link_section(hir_id, attr, span, target),
                sym::no_mangle => self.check_no_mangle(hir_id, attr, span, target),
                sym::deprecated | sym::rustc_deprecated => {
                    self.check_deprecated(hir_id, attr, span, target)
                }
                sym::macro_use | sym::macro_escape => self.check_macro_use(hir_id, attr, target),
                sym::path => self.check_generic_attr(hir_id, attr, target, &[Target::Mod]),
                sym::cfg_attr => self.check_cfg_attr(hir_id, attr),
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
                _ => {}
            }

            let builtin = attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name));

            if hir_id != CRATE_HIR_ID {
                if let Some(BuiltinAttribute { type_: AttributeType::CrateLevel, .. }) =
                    attr.ident().and_then(|ident| BUILTIN_ATTRIBUTE_MAP.get(&ident.name))
                {
                    self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                        let msg = match attr.style {
                            ast::AttrStyle::Outer => {
                                "crate-level attribute should be an inner attribute: add an exclamation \
                                 mark: `#![foo]`"
                            }
                            ast::AttrStyle::Inner => "crate-level attribute should be in the root module",
                        };
                        lint.build(msg).emit()
                    });
                }
            }

            if let Some(BuiltinAttribute { duplicates, .. }) = builtin {
                check_duplicates(self.tcx, attr, hir_id, *duplicates, &mut seen);
            }

            // Warn on useless empty attributes.
            if matches!(
                attr.name_or_empty(),
                sym::macro_use
                    | sym::allow
                    | sym::warn
                    | sym::deny
                    | sym::forbid
                    | sym::feature
                    | sym::repr
                    | sym::target_feature
            ) && attr.meta_item_list().map_or(false, |list| list.is_empty())
            {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("unused attribute")
                        .span_suggestion(
                            attr.span,
                            "remove this attribute",
                            String::new(),
                            Applicability::MachineApplicable,
                        )
                        .note(&format!(
                            "attribute `{}` with an empty list has no effect",
                            attr.name_or_empty()
                        ))
                        .emit();
                });
            }
        }

        if !is_valid {
            return;
        }

        if matches!(target, Target::Closure | Target::Fn | Target::Method(_) | Target::ForeignFn) {
            self.tcx.ensure().codegen_fn_attrs(self.tcx.hir().local_def_id(hir_id));
        }

        self.check_repr(attrs, span, target, item, hir_id);
        self.check_used(attrs, target);
    }

    fn inline_attr_str_error_with_macro_def(&self, hir_id: HirId, attr: &Attribute, sym: &str) {
        self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
            lint.build(&format!(
                "`#[{}]` is ignored on struct fields, match arms and macro defs",
                sym,
            ))
            .warn(
                "this was previously accepted by the compiler but is \
                 being phased out; it will become a hard error in \
                 a future release!",
            )
            .note(
                "see issue #80564 <https://github.com/rust-lang/rust/issues/80564> \
                 for more information",
            )
            .emit();
        });
    }

    fn inline_attr_str_error_without_macro_def(&self, hir_id: HirId, attr: &Attribute, sym: &str) {
        self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
            lint.build(&format!("`#[{}]` is ignored on struct fields and match arms", sym))
                .warn(
                    "this was previously accepted by the compiler but is \
                 being phased out; it will become a hard error in \
                 a future release!",
                )
                .note(
                    "see issue #80564 <https://github.com/rust-lang/rust/issues/80564> \
                 for more information",
                )
                .emit();
        });
    }

    /// Checks if an `#[inline]` is applied to a function or a closure. Returns `true` if valid.
    fn check_inline(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Fn
            | Target::Closure
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
            Target::Method(MethodKind::Trait { body: false }) | Target::ForeignFn => {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("`#[inline]` is ignored on function prototypes").emit()
                });
                true
            }
            // FIXME(#65833): We permit associated consts to have an `#[inline]` attribute with
            // just a lint, because we previously erroneously allowed it and some crates used it
            // accidentally, to to be compatible with crates depending on them, we can't throw an
            // error here.
            Target::AssocConst => {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("`#[inline]` is ignored on constants")
                        .warn(
                            "this was previously accepted by the compiler but is \
                             being phased out; it will become a hard error in \
                             a future release!",
                        )
                        .note(
                            "see issue #65833 <https://github.com/rust-lang/rust/issues/65833> \
                             for more information",
                        )
                        .emit();
                });
                true
            }
            // FIXME(#80564): Same for fields, arms, and macro defs
            Target::Field | Target::Arm | Target::MacroDef => {
                self.inline_attr_str_error_with_macro_def(hir_id, attr, "inline");
                true
            }
            _ => {
                struct_span_err!(
                    self.tcx.sess,
                    attr.span,
                    E0518,
                    "attribute should be applied to function or closure",
                )
                .span_label(*span, "not a function or closure")
                .emit();
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
    fn check_naked(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) -> bool {
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
                self.tcx
                    .sess
                    .struct_span_err(
                        attr.span,
                        "attribute should be applied to a function definition",
                    )
                    .span_label(*span, "not a function definition")
                    .emit();
                false
            }
        }
    }

    /// Checks if `#[cmse_nonsecure_entry]` is applied to a function definition.
    fn check_cmse_nonsecure_entry(&self, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(
                        attr.span,
                        "attribute should be applied to a function definition",
                    )
                    .span_label(*span, "not a function definition")
                    .emit();
                false
            }
        }
    }

    /// Checks if a `#[track_caller]` is applied to a non-naked function. Returns `true` if valid.
    fn check_track_caller(
        &self,
        hir_id: HirId,
        attr_span: &Span,
        attrs: &[Attribute],
        span: &Span,
        target: Target,
    ) -> bool {
        match target {
            _ if attrs.iter().any(|attr| attr.has_name(sym::naked)) => {
                struct_span_err!(
                    self.tcx.sess,
                    *attr_span,
                    E0736,
                    "cannot use `#[track_caller]` with `#[naked]`",
                )
                .emit();
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
                struct_span_err!(
                    self.tcx.sess,
                    *attr_span,
                    E0739,
                    "attribute should be applied to function"
                )
                .span_label(*span, "not a function")
                .emit();
                false
            }
        }
    }

    /// Checks if the `#[non_exhaustive]` attribute on an `item` is valid. Returns `true` if valid.
    fn check_non_exhaustive(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: &Span,
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
                struct_span_err!(
                    self.tcx.sess,
                    attr.span,
                    E0701,
                    "attribute can only be applied to a struct or enum"
                )
                .span_label(*span, "not a struct or enum")
                .emit();
                false
            }
        }
    }

    /// Checks if the `#[marker]` attribute on an `item` is valid. Returns `true` if valid.
    fn check_marker(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) -> bool {
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
                self.tcx
                    .sess
                    .struct_span_err(attr.span, "attribute can only be applied to a trait")
                    .span_label(*span, "not a trait")
                    .emit();
                false
            }
        }
    }

    /// Checks if the `#[target_feature]` attribute on `item` is valid. Returns `true` if valid.
    fn check_target_feature(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: &Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
            // FIXME: #[target_feature] was previously erroneously allowed on statements and some
            // crates used this, so only emit a warning.
            Target::Statement => {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("attribute should be applied to a function")
                        .warn(
                            "this was previously accepted by the compiler but is \
                             being phased out; it will become a hard error in \
                             a future release!",
                        )
                        .span_label(*span, "not a function")
                        .emit();
                });
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
                self.tcx
                    .sess
                    .struct_span_err(attr.span, "attribute should be applied to a function")
                    .span_label(*span, "not a function")
                    .emit();
                false
            }
        }
    }

    fn doc_attr_str_error(&self, meta: &NestedMetaItem, attr_name: &str) {
        self.tcx
            .sess
            .struct_span_err(
                meta.span(),
                &format!("doc {0} attribute expects a string: #[doc({0} = \"a\")]", attr_name),
            )
            .emit();
    }

    fn check_doc_alias_value(
        &self,
        meta: &NestedMetaItem,
        doc_alias: &str,
        hir_id: HirId,
        target: Target,
        is_list: bool,
        aliases: &mut FxHashMap<String, Span>,
    ) -> bool {
        let tcx = self.tcx;
        let err_fn = move |span: Span, msg: &str| {
            tcx.sess.span_err(
                span,
                &format!(
                    "`#[doc(alias{})]` {}",
                    if is_list { "(\"...\")" } else { " = \"...\"" },
                    msg,
                ),
            );
            false
        };
        if doc_alias.is_empty() {
            return err_fn(
                meta.name_value_literal_span().unwrap_or_else(|| meta.span()),
                "attribute cannot have empty value",
            );
        }
        if let Some(c) =
            doc_alias.chars().find(|&c| c == '"' || c == '\'' || (c.is_whitespace() && c != ' '))
        {
            self.tcx.sess.span_err(
                meta.name_value_literal_span().unwrap_or_else(|| meta.span()),
                &format!(
                    "{:?} character isn't allowed in `#[doc(alias{})]`",
                    c,
                    if is_list { "(\"...\")" } else { " = \"...\"" },
                ),
            );
            return false;
        }
        if doc_alias.starts_with(' ') || doc_alias.ends_with(' ') {
            return err_fn(
                meta.name_value_literal_span().unwrap_or_else(|| meta.span()),
                "cannot start or end with ' '",
            );
        }
        if let Some(err) = match target {
            Target::Impl => Some("implementation block"),
            Target::ForeignMod => Some("extern block"),
            Target::AssocTy => {
                let parent_hir_id = self.tcx.hir().get_parent_item(hir_id).expect_owner();
                let containing_item = self.tcx.hir().expect_item(parent_hir_id);
                if Target::from_item(containing_item) == Target::Impl {
                    Some("type alias in implementation block")
                } else {
                    None
                }
            }
            Target::AssocConst => {
                let parent_hir_id = self.tcx.hir().get_parent_item(hir_id).expect_owner();
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
            _ => None,
        } {
            return err_fn(meta.span(), &format!("isn't allowed on {}", err));
        }
        let item_name = self.tcx.hir().name(hir_id);
        if item_name.as_str() == doc_alias {
            return err_fn(meta.span(), "is the same as the item's name");
        }
        let span = meta.span();
        if let Err(entry) = aliases.try_insert(doc_alias.to_owned(), span) {
            self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, span, |lint| {
                lint.build("doc alias is duplicated")
                    .span_label(*entry.entry.get(), "first defined here")
                    .emit();
            });
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
                            if !self.check_doc_alias_value(
                                v,
                                s.as_str(),
                                hir_id,
                                target,
                                true,
                                aliases,
                            ) {
                                errors += 1;
                            }
                        }
                        _ => {
                            self.tcx
                                .sess
                                .struct_span_err(
                                    v.span(),
                                    "`#[doc(alias(\"a\"))]` expects string literals",
                                )
                                .emit();
                            errors += 1;
                        }
                    },
                    None => {
                        self.tcx
                            .sess
                            .struct_span_err(
                                v.span(),
                                "`#[doc(alias(\"a\"))]` expects string literals",
                            )
                            .emit();
                        errors += 1;
                    }
                }
            }
            errors == 0
        } else if let Some(doc_alias) = meta.value_str().map(|s| s.to_string()) {
            self.check_doc_alias_value(meta, &doc_alias, hir_id, target, false, aliases)
        } else {
            self.tcx
                .sess
                .struct_span_err(
                    meta.span(),
                    "doc alias attribute expects a string `#[doc(alias = \"a\")]` or a list of \
                     strings `#[doc(alias(\"a\", \"b\"))]`",
                )
                .emit();
            false
        }
    }

    fn check_doc_keyword(&self, meta: &NestedMetaItem, hir_id: HirId) -> bool {
        let doc_keyword = meta.value_str().map(|s| s.to_string()).unwrap_or_else(String::new);
        if doc_keyword.is_empty() {
            self.doc_attr_str_error(meta, "keyword");
            return false;
        }
        match self.tcx.hir().find(hir_id).and_then(|node| match node {
            hir::Node::Item(item) => Some(&item.kind),
            _ => None,
        }) {
            Some(ItemKind::Mod(ref module)) => {
                if !module.item_ids.is_empty() {
                    self.tcx
                        .sess
                        .struct_span_err(
                            meta.span(),
                            "`#[doc(keyword = \"...\")]` can only be used on empty modules",
                        )
                        .emit();
                    return false;
                }
            }
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(
                        meta.span(),
                        "`#[doc(keyword = \"...\")]` can only be used on modules",
                    )
                    .emit();
                return false;
            }
        }
        if !rustc_lexer::is_ident(&doc_keyword) {
            self.tcx
                .sess
                .struct_span_err(
                    meta.name_value_literal_span().unwrap_or_else(|| meta.span()),
                    &format!("`{}` is not a valid identifier", doc_keyword),
                )
                .emit();
            return false;
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
                    spans.push_span_label(prev_span, String::from("this attribute..."));
                    spans.push_span_label(
                        meta.span(),
                        String::from("...conflicts with this attribute"),
                    );
                    self.tcx
                        .sess
                        .struct_span_err(spans, "conflicting doc inlining attributes")
                        .help("remove one of the conflicting attributes")
                        .emit();
                    return false;
                }
                true
            } else {
                *specified_inline = Some((do_inline, meta.span()));
                true
            }
        } else {
            self.tcx.struct_span_lint_hir(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                |lint| {
                    let mut err = lint.build(
                        "this attribute can only be applied to a `use` item",
                    );
                    err.span_label(meta.span(), "only applicable on `use` items");
                    if attr.style == AttrStyle::Outer {
                        err.span_label(
                            self.tcx.hir().span(hir_id),
                            "not a `use` item",
                        );
                    }
                    err.note("read https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#inline-and-no_inline for more information")
                        .emit();
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
            self.tcx
                .sess
                .struct_span_err(
                    meta.span(),
                    &format!(
                        "`#![doc({} = \"...\")]` isn't allowed as a crate-level attribute",
                        attr_name,
                    ),
                )
                .emit();
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
            self.tcx.struct_span_lint_hir(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                meta.span(),
                |lint| {
                    let mut err = lint.build(
                        "this attribute can only be applied at the crate level",
                    );
                    if attr.style == AttrStyle::Outer && self.tcx.hir().get_parent_item(hir_id) == CRATE_HIR_ID {
                        if let Ok(mut src) =
                            self.tcx.sess.source_map().span_to_snippet(attr.span)
                        {
                            src.insert(1, '!');
                            err.span_suggestion_verbose(
                                attr.span,
                                "to apply to the crate, use an inner attribute",
                                src,
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            err.span_help(
                                attr.span,
                                "to apply to the crate, use an inner attribute",
                            );
                        }
                    }
                    err.note("read https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level for more information")
                        .emit();
                },
            );
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
                        self.tcx.struct_span_lint_hir(
                            INVALID_DOC_ATTRIBUTES,
                            hir_id,
                            i_meta.span(),
                            |lint| {
                                lint.build(&format!(
                                    "unknown `doc(test)` attribute `{}`",
                                    rustc_ast_pretty::pprust::path_to_string(
                                        &i_meta.meta_item().unwrap().path
                                    ),
                                ))
                                .emit();
                            },
                        );
                        is_valid = false;
                    }
                }
            }
        } else {
            self.tcx.struct_span_lint_hir(INVALID_DOC_ATTRIBUTES, hir_id, meta.span(), |lint| {
                lint.build("`#[doc(test(...)]` takes a list of attributes").emit();
            });
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

        if let Some(list) = attr.meta().and_then(|mi| mi.meta_item_list().map(|l| l.to_vec())) {
            for meta in &list {
                if let Some(i_meta) = meta.meta_item() {
                    match i_meta.name_or_empty() {
                        sym::alias
                            if !self.check_attr_not_crate_level(&meta, hir_id, "alias")
                                || !self.check_doc_alias(&meta, hir_id, target, aliases) =>
                        {
                            is_valid = false
                        }

                        sym::keyword
                            if !self.check_attr_not_crate_level(&meta, hir_id, "keyword")
                                || !self.check_doc_keyword(&meta, hir_id) =>
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
                            if !self.check_attr_crate_level(&attr, &meta, hir_id) =>
                        {
                            is_valid = false;
                        }

                        sym::inline | sym::no_inline
                            if !self.check_doc_inline(
                                &attr,
                                &meta,
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
                        | sym::plugins => {}

                        sym::test => {
                            if !self.check_test_attr(&meta, hir_id) {
                                is_valid = false;
                            }
                        }

                        sym::primitive => {
                            if !self.tcx.features().rustdoc_internals {
                                self.tcx.struct_span_lint_hir(
                                    INVALID_DOC_ATTRIBUTES,
                                    hir_id,
                                    i_meta.span,
                                    |lint| {
                                        let mut diag = lint.build(
                                            "`doc(primitive)` should never have been stable",
                                        );
                                        diag.emit();
                                    },
                                );
                            }
                        }

                        _ => {
                            self.tcx.struct_span_lint_hir(
                                INVALID_DOC_ATTRIBUTES,
                                hir_id,
                                i_meta.span,
                                |lint| {
                                    let mut diag = lint.build(&format!(
                                        "unknown `doc` attribute `{}`",
                                        rustc_ast_pretty::pprust::path_to_string(&i_meta.path),
                                    ));
                                    if i_meta.has_name(sym::spotlight) {
                                        diag.note(
                                            "`doc(spotlight)` was renamed to `doc(notable_trait)`",
                                        );
                                        diag.span_suggestion_short(
                                            i_meta.span,
                                            "use `notable_trait` instead",
                                            String::from("notable_trait"),
                                            Applicability::MachineApplicable,
                                        );
                                        diag.note("`doc(spotlight)` is now a no-op");
                                    }
                                    if i_meta.has_name(sym::include) {
                                        if let Some(value) = i_meta.value_str() {
                                            // if there are multiple attributes, the suggestion would suggest deleting all of them, which is incorrect
                                            let applicability = if list.len() == 1 {
                                                Applicability::MachineApplicable
                                            } else {
                                                Applicability::MaybeIncorrect
                                            };
                                            let inner = if attr.style == AttrStyle::Inner {
                                                "!"
                                            } else {
                                                ""
                                            };
                                            diag.span_suggestion(
                                                attr.meta().unwrap().span,
                                                "use `doc = include_str!` instead",
                                                format!(
                                                    "#{}[doc = include_str!(\"{}\")]",
                                                    inner, value
                                                ),
                                                applicability,
                                            );
                                        }
                                    }
                                    diag.emit();
                                },
                            );
                            is_valid = false;
                        }
                    }
                } else {
                    self.tcx.struct_span_lint_hir(
                        INVALID_DOC_ATTRIBUTES,
                        hir_id,
                        meta.span(),
                        |lint| {
                            lint.build(&"invalid `doc` attribute").emit();
                        },
                    );
                    is_valid = false;
                }
            }
        }

        is_valid
    }

    /// Warns against some misuses of `#[must_use]`
    fn check_must_use(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: &Span,
        _target: Target,
    ) -> bool {
        let node = self.tcx.hir().get(hir_id);
        if let Some(fn_node) = node.fn_kind() {
            if let rustc_hir::IsAsync::Async = fn_node.asyncness() {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build(
                        "`must_use` attribute on `async` functions \
                              applies to the anonymous `Future` returned by the \
                              function, not the value within",
                    )
                    .span_label(
                        *span,
                        "this attribute does nothing, the `Future`s \
                                returned by async functions are already `must_use`",
                    )
                    .emit();
                });
            }
        }

        // For now, its always valid
        true
    }

    /// Checks if `#[must_not_suspend]` is applied to a function. Returns `true` if valid.
    fn check_must_not_suspend(&self, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Struct | Target::Enum | Target::Union | Target::Trait => true,
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(attr.span, "`must_not_suspend` attribute should be applied to a struct, enum, or trait")
                        .span_label(*span, "is not a struct, enum, or trait")
                        .emit();
                false
            }
        }
    }

    /// Checks if `#[cold]` is applied to a non-function. Returns `true` if valid.
    fn check_cold(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) {
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
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("attribute should be applied to a function")
                        .warn(
                            "this was previously accepted by the compiler but is \
                             being phased out; it will become a hard error in \
                             a future release!",
                        )
                        .span_label(*span, "not a function")
                        .emit();
                });
            }
        }
    }

    /// Checks if `#[link_name]` is applied to an item other than a foreign function or static.
    fn check_link_name(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) {
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
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    let mut diag =
                        lint.build("attribute should be applied to a foreign function or static");
                    diag.warn(
                        "this was previously accepted by the compiler but is \
                         being phased out; it will become a hard error in \
                         a future release!",
                    );

                    // See issue #47725
                    if let Target::ForeignMod = target {
                        if let Some(value) = attr.value_str() {
                            diag.span_help(
                                attr.span,
                                &format!(r#"try `#[link(name = "{}")]` instead"#, value),
                            );
                        } else {
                            diag.span_help(attr.span, r#"try `#[link(name = "...")]` instead"#);
                        }
                    }

                    diag.span_label(*span, "not a foreign function or static");
                    diag.emit();
                });
            }
        }
    }

    /// Checks if `#[no_link]` is applied to an `extern crate`. Returns `true` if valid.
    fn check_no_link(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) -> bool {
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
                self.tcx
                    .sess
                    .struct_span_err(
                        attr.span,
                        "attribute should be applied to an `extern crate` item",
                    )
                    .span_label(*span, "not an `extern crate` item")
                    .emit();
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
        span: &Span,
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
                self.tcx
                    .sess
                    .struct_span_err(
                        attr.span,
                        "attribute should be applied to a free function, impl method or static",
                    )
                    .span_label(*span, "not a free function, impl method or static")
                    .emit();
                false
            }
        }
    }

    fn check_rustc_layout_scalar_valid_range(
        &self,
        attr: &Attribute,
        span: &Span,
        target: Target,
    ) -> bool {
        if target != Target::Struct {
            self.tcx
                .sess
                .struct_span_err(attr.span, "attribute should be applied to a struct")
                .span_label(*span, "not a struct")
                .emit();
            return false;
        }

        let list = match attr.meta_item_list() {
            None => return false,
            Some(it) => it,
        };

        if matches!(&list[..], &[NestedMetaItem::Literal(Lit { kind: LitKind::Int(..), .. })]) {
            true
        } else {
            self.tcx
                .sess
                .struct_span_err(attr.span, "expected exactly one integer literal argument")
                .emit();
            false
        }
    }

    /// Checks if `#[rustc_legacy_const_generics]` is applied to a function and has a valid argument.
    fn check_rustc_legacy_const_generics(
        &self,
        attr: &Attribute,
        span: &Span,
        target: Target,
        item: Option<ItemLike<'_>>,
    ) -> bool {
        let is_function = matches!(target, Target::Fn | Target::Method(..));
        if !is_function {
            self.tcx
                .sess
                .struct_span_err(attr.span, "attribute should be applied to a function")
                .span_label(*span, "not a function")
                .emit();
            return false;
        }

        let list = match attr.meta_item_list() {
            // The attribute form is validated on AST.
            None => return false,
            Some(it) => it,
        };

        let (decl, generics) = match item {
            Some(ItemLike::Item(Item {
                kind: ItemKind::Fn(FnSig { decl, .. }, generics, _),
                ..
            })) => (decl, generics),
            _ => bug!("should be a function item"),
        };

        for param in generics.params {
            match param.kind {
                hir::GenericParamKind::Const { .. } => {}
                _ => {
                    self.tcx
                        .sess
                        .struct_span_err(
                            attr.span,
                            "#[rustc_legacy_const_generics] functions must \
                             only have const generics",
                        )
                        .span_label(param.span, "non-const generic parameter")
                        .emit();
                    return false;
                }
            }
        }

        if list.len() != generics.params.len() {
            self.tcx
                .sess
                .struct_span_err(
                    attr.span,
                    "#[rustc_legacy_const_generics] must have one index for each generic parameter",
                )
                .span_label(generics.span, "generic parameters")
                .emit();
            return false;
        }

        let arg_count = decl.inputs.len() as u128 + generics.params.len() as u128;
        let mut invalid_args = vec![];
        for meta in list {
            if let Some(LitKind::Int(val, _)) = meta.literal().map(|lit| &lit.kind) {
                if *val >= arg_count {
                    let span = meta.span();
                    self.tcx
                        .sess
                        .struct_span_err(span, "index exceeds number of arguments")
                        .span_label(
                            span,
                            format!(
                                "there {} only {} argument{}",
                                if arg_count != 1 { "are" } else { "is" },
                                arg_count,
                                pluralize!(arg_count)
                            ),
                        )
                        .emit();
                    return false;
                }
            } else {
                invalid_args.push(meta.span());
            }
        }

        if !invalid_args.is_empty() {
            self.tcx
                .sess
                .struct_span_err(invalid_args, "arguments should be non-negative integers")
                .emit();
            false
        } else {
            true
        }
    }

    /// Checks that the dep-graph debugging attributes are only present when the query-dep-graph
    /// option is passed to the compiler.
    fn check_rustc_dirty_clean(&self, attr: &Attribute) -> bool {
        if self.tcx.sess.opts.debugging_opts.query_dep_graph {
            true
        } else {
            self.tcx
                .sess
                .struct_span_err(attr.span, "attribute requires -Z query-dep-graph to be enabled")
                .emit();
            false
        }
    }

    /// Checks if `#[link_section]` is applied to a function or static.
    fn check_link_section(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) {
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
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("attribute should be applied to a function or static")
                        .warn(
                            "this was previously accepted by the compiler but is \
                             being phased out; it will become a hard error in \
                             a future release!",
                        )
                        .span_label(*span, "not a function or static")
                        .emit();
                });
            }
        }
    }

    /// Checks if `#[no_mangle]` is applied to a function or static.
    fn check_no_mangle(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) {
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
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build(&format!(
                        "`#[no_mangle]` has no effect on a foreign {}",
                        foreign_item_kind
                    ))
                    .warn(
                        "this was previously accepted by the compiler but is \
                            being phased out; it will become a hard error in \
                            a future release!",
                    )
                    .span_label(*span, format!("foreign {}", foreign_item_kind))
                    .note("symbol names in extern blocks are not mangled")
                    .span_suggestion(
                        attr.span,
                        "remove this attribute",
                        String::new(),
                        Applicability::MachineApplicable,
                    )
                    .emit();
                });
            }
            _ => {
                // FIXME: #[no_mangle] was previously allowed on non-functions/statics and some
                // crates used this, so only emit a warning.
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build(
                        "attribute should be applied to a free function, impl method or static",
                    )
                    .warn(
                        "this was previously accepted by the compiler but is \
                         being phased out; it will become a hard error in \
                         a future release!",
                    )
                    .span_label(*span, "not a free function, impl method or static")
                    .emit();
                });
            }
        }
    }

    /// Checks if the `#[repr]` attributes on `item` are valid.
    fn check_repr(
        &self,
        attrs: &[Attribute],
        span: &Span,
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
                struct_span_err!(
                    self.tcx.sess,
                    hint.span(),
                    E0565,
                    "meta item in `repr` must be an identifier"
                )
                .emit();
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
                    if let (Target::Fn, true) = (target, !self.tcx.features().fn_align) {
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
                sym::no_niche => {
                    if !self.tcx.features().enabled(sym::no_niche) {
                        feature_err(
                            &self.tcx.sess.parse_sess,
                            sym::no_niche,
                            hint.span(),
                            "the attribute `repr(no_niche)` is currently unstable",
                        )
                        .emit();
                    }
                    match target {
                        Target::Struct | Target::Enum => continue,
                        _ => ("a", "struct or enum"),
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
                    .emit();

                    continue;
                }
            };

            struct_span_err!(
                self.tcx.sess,
                hint.span(),
                E0517,
                "{}",
                &format!("attribute should be applied to {} {}", article, allowed_targets)
            )
            .span_label(*span, &format!("not {} {}", article, allowed_targets))
            .emit();
        }

        // Just point at all repr hints if there are any incompatibilities.
        // This is not ideal, but tracking precisely which ones are at fault is a huge hassle.
        let hint_spans = hints.iter().map(|hint| hint.span());

        // Error on repr(transparent, <anything else apart from no_niche>).
        let non_no_niche = |hint: &&NestedMetaItem| hint.name_or_empty() != sym::no_niche;
        let non_no_niche_count = hints.iter().filter(non_no_niche).count();
        if is_transparent && non_no_niche_count > 1 {
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
            self.tcx.struct_span_lint_hir(
                CONFLICTING_REPR_HINTS,
                hir_id,
                hint_spans.collect::<Vec<Span>>(),
                |lint| {
                    lint.build("conflicting representation hints")
                        .code(rustc_errors::error_code!(E0566))
                        .emit();
                },
            );
        }
    }

    fn check_used(&self, attrs: &[Attribute], target: Target) {
        for attr in attrs {
            if attr.has_name(sym::used) && target != Target::Static {
                self.tcx
                    .sess
                    .span_err(attr.span, "attribute must be applied to a `static` variable");
            }
        }
    }

    /// Outputs an error for `#[allow_internal_unstable]` which can only be applied to macros.
    /// (Allows proc_macro functions)
    fn check_allow_internal_unstable(
        &self,
        hir_id: HirId,
        attr: &Attribute,
        span: &Span,
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
                    .struct_span_err(attr.span, "attribute should be applied to a macro")
                    .span_label(*span, "not a macro")
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
        span: &Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Fn | Target::Method(_)
                if self.tcx.is_const_fn_raw(self.tcx.hir().local_def_id(hir_id)) =>
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
                    .struct_span_err(attr.span, "attribute should be applied to `const fn`")
                    .span_label(*span, "not a `const fn`")
                    .emit();
                false
            }
        }
    }

    /// default_method_body_is_const should only be applied to trait methods with default bodies.
    fn check_default_method_body_is_const(
        &self,
        attr: &Attribute,
        span: &Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Method(MethodKind::Trait { body: true }) => true,
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(
                        attr.span,
                        "attribute should be applied to a trait method with body",
                    )
                    .span_label(*span, "not a trait method or missing a body")
                    .emit();
                false
            }
        }
    }

    fn check_stability_promotable(&self, attr: &Attribute, _span: &Span, target: Target) -> bool {
        match target {
            Target::Expression => {
                self.tcx
                    .sess
                    .struct_span_err(attr.span, "attribute cannot be applied to an expression")
                    .emit();
                false
            }
            _ => true,
        }
    }

    fn check_deprecated(&self, hir_id: HirId, attr: &Attribute, _span: &Span, target: Target) {
        match target {
            Target::Closure | Target::Expression | Target::Statement | Target::Arm => {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("attribute is ignored here").emit();
                });
            }
            _ => {}
        }
    }

    fn check_macro_use(&self, hir_id: HirId, attr: &Attribute, target: Target) {
        let name = attr.name_or_empty();
        match target {
            Target::ExternCrate | Target::Mod => {}
            _ => {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build(&format!(
                        "`#[{name}]` only has an effect on `extern crate` and modules"
                    ))
                    .emit();
                });
            }
        }
    }

    fn check_macro_export(&self, hir_id: HirId, attr: &Attribute, target: Target) {
        if target != Target::MacroDef {
            self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                lint.build("`#[macro_export]` only has an effect on macro definitions").emit();
            });
        }
    }

    fn check_cfg_attr(&self, hir_id: HirId, attr: &Attribute) {
        if let Some((_, attrs)) = rustc_parse::parse_cfg_attr(&attr, &self.tcx.sess.parse_sess) {
            if attrs.is_empty() {
                self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                    lint.build("`#[cfg_attr]` does not expand to any attributes").emit();
                });
            }
        }
    }

    fn check_plugin_registrar(&self, hir_id: HirId, attr: &Attribute, target: Target) {
        if target != Target::Fn {
            self.tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, attr.span, |lint| {
                lint.build("`#[plugin_registrar]` only has an effect on functions").emit();
            });
        }
    }
}

impl<'tcx> Visitor<'tcx> for CheckAttrVisitor<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        // Historically we've run more checks on non-exported than exported macros,
        // so this lets us continue to run them while maintaining backwards compatibility.
        // In the long run, the checks should be harmonized.
        if let ItemKind::Macro(ref macro_def) = item.kind {
            let def_id = item.def_id.to_def_id();
            if macro_def.macro_rules && !self.tcx.has_attr(def_id, sym::macro_export) {
                check_non_exported_macro_for_invalid_attrs(self.tcx, item);
            }
        }

        let target = Target::from_item(item);
        self.check_attributes(item.hir_id(), &item.span, target, Some(ItemLike::Item(item)));
        intravisit::walk_item(self, item)
    }

    fn visit_generic_param(&mut self, generic_param: &'tcx hir::GenericParam<'tcx>) {
        let target = Target::from_generic_param(generic_param);
        self.check_attributes(generic_param.hir_id, &generic_param.span, target, None);
        intravisit::walk_generic_param(self, generic_param)
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx TraitItem<'tcx>) {
        let target = Target::from_trait_item(trait_item);
        self.check_attributes(trait_item.hir_id(), &trait_item.span, target, None);
        intravisit::walk_trait_item(self, trait_item)
    }

    fn visit_field_def(&mut self, struct_field: &'tcx hir::FieldDef<'tcx>) {
        self.check_attributes(struct_field.hir_id, &struct_field.span, Target::Field, None);
        intravisit::walk_field_def(self, struct_field);
    }

    fn visit_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) {
        self.check_attributes(arm.hir_id, &arm.span, Target::Arm, None);
        intravisit::walk_arm(self, arm);
    }

    fn visit_foreign_item(&mut self, f_item: &'tcx ForeignItem<'tcx>) {
        let target = Target::from_foreign_item(f_item);
        self.check_attributes(
            f_item.hir_id(),
            &f_item.span,
            target,
            Some(ItemLike::ForeignItem(f_item)),
        );
        intravisit::walk_foreign_item(self, f_item)
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        let target = target_from_impl_item(self.tcx, impl_item);
        self.check_attributes(impl_item.hir_id(), &impl_item.span, target, None);
        intravisit::walk_impl_item(self, impl_item)
    }

    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
        // When checking statements ignore expressions, they will be checked later.
        if let hir::StmtKind::Local(ref l) = stmt.kind {
            self.check_attributes(l.hir_id, &stmt.span, Target::Statement, None);
        }
        intravisit::walk_stmt(self, stmt)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let target = match expr.kind {
            hir::ExprKind::Closure(..) => Target::Closure,
            _ => Target::Expression,
        };

        self.check_attributes(expr.hir_id, &expr.span, target, None);
        intravisit::walk_expr(self, expr)
    }

    fn visit_variant(
        &mut self,
        variant: &'tcx hir::Variant<'tcx>,
        generics: &'tcx hir::Generics<'tcx>,
        item_id: HirId,
    ) {
        self.check_attributes(variant.id, &variant.span, Target::Variant, None);
        intravisit::walk_variant(self, variant, generics, item_id)
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        self.check_attributes(param.hir_id, &param.span, Target::Param, None);

        intravisit::walk_param(self, param);
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
                    err.emit()
                }
            }
        }
    }
}

fn check_non_exported_macro_for_invalid_attrs(tcx: TyCtxt<'_>, item: &Item<'_>) {
    let attrs = tcx.hir().attrs(item.hir_id());

    for attr in attrs {
        if attr.has_name(sym::inline) {
            struct_span_err!(
                tcx.sess,
                attr.span,
                E0518,
                "attribute should be applied to function or closure",
            )
            .span_label(attr.span, "not a function or closure")
            .emit();
        }
    }
}

fn check_mod_attrs(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    let check_attr_visitor = &mut CheckAttrVisitor { tcx };
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut check_attr_visitor.as_deep_visitor());
    if module_def_id.is_top_level_module() {
        check_attr_visitor.check_attributes(CRATE_HIR_ID, &DUMMY_SP, Target::Mod, None);
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
                    tcx.struct_span_lint_hir(UNUSED_ATTRIBUTES, hir_id, this, |lint| {
                        let mut db = lint.build("unused attribute");
                        db.span_note(other, "attribute also specified here").span_suggestion(
                            this,
                            "remove this attribute",
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                        if matches!(duplicates, FutureWarnFollowing | FutureWarnPreceding) {
                            db.warn(
                                "this was previously accepted by the compiler but is \
                                 being phased out; it will become a hard error in \
                                 a future release!",
                            );
                        }
                        db.emit();
                    });
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
                tcx.sess
                    .struct_span_err(
                        this,
                        &format!("multiple `{}` attributes", attr.name_or_empty()),
                    )
                    .span_note(other, "attribute also specified here")
                    .span_suggestion(
                        this,
                        "remove this attribute",
                        String::new(),
                        Applicability::MachineApplicable,
                    )
                    .emit();
            }
            Entry::Vacant(entry) => {
                entry.insert(attr.span);
            }
        },
    }
}

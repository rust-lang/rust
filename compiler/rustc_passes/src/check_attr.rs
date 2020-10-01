//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use rustc_middle::hir::map::Map;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;

use rustc_ast::{Attribute, LitKind, NestedMetaItem};
use rustc_errors::{pluralize, struct_span_err};
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{self, FnSig, ForeignItem, ForeignItemKind, HirId, Item, ItemKind, TraitItem};
use rustc_hir::{MethodKind, Target};
use rustc_session::lint::builtin::{CONFLICTING_REPR_HINTS, UNUSED_ATTRIBUTES};
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::Span;

pub(crate) fn target_from_impl_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_item: &hir::ImplItem<'_>,
) -> Target {
    match impl_item.kind {
        hir::ImplItemKind::Const(..) => Target::AssocConst,
        hir::ImplItemKind::Fn(..) => {
            let parent_hir_id = tcx.hir().get_parent_item(impl_item.hir_id);
            let containing_item = tcx.hir().expect_item(parent_hir_id);
            let containing_impl_is_for_trait = match &containing_item.kind {
                hir::ItemKind::Impl { ref of_trait, .. } => of_trait.is_some(),
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

impl CheckAttrVisitor<'tcx> {
    /// Checks any attribute.
    fn check_attributes(
        &self,
        hir_id: HirId,
        attrs: &'hir [Attribute],
        span: &Span,
        target: Target,
        item: Option<ItemLike<'_>>,
    ) {
        let mut is_valid = true;
        for attr in attrs {
            is_valid &= if self.tcx.sess.check_name(attr, sym::inline) {
                self.check_inline(hir_id, attr, span, target)
            } else if self.tcx.sess.check_name(attr, sym::non_exhaustive) {
                self.check_non_exhaustive(attr, span, target)
            } else if self.tcx.sess.check_name(attr, sym::marker) {
                self.check_marker(attr, span, target)
            } else if self.tcx.sess.check_name(attr, sym::target_feature) {
                self.check_target_feature(hir_id, attr, span, target)
            } else if self.tcx.sess.check_name(attr, sym::track_caller) {
                self.check_track_caller(&attr.span, attrs, span, target)
            } else if self.tcx.sess.check_name(attr, sym::doc) {
                self.check_doc_alias(attr, hir_id, target)
            } else if self.tcx.sess.check_name(attr, sym::no_link) {
                self.check_no_link(&attr, span, target)
            } else if self.tcx.sess.check_name(attr, sym::export_name) {
                self.check_export_name(&attr, span, target)
            } else if self.tcx.sess.check_name(attr, sym::rustc_args_required_const) {
                self.check_rustc_args_required_const(&attr, span, target, item)
            } else {
                // lint-only checks
                if self.tcx.sess.check_name(attr, sym::cold) {
                    self.check_cold(hir_id, attr, span, target);
                } else if self.tcx.sess.check_name(attr, sym::link_name) {
                    self.check_link_name(hir_id, attr, span, target);
                } else if self.tcx.sess.check_name(attr, sym::link_section) {
                    self.check_link_section(hir_id, attr, span, target);
                } else if self.tcx.sess.check_name(attr, sym::no_mangle) {
                    self.check_no_mangle(hir_id, attr, span, target);
                }
                true
            };
        }

        if !is_valid {
            return;
        }

        if matches!(target, Target::Fn | Target::Method(_) | Target::ForeignFn) {
            self.tcx.ensure().codegen_fn_attrs(self.tcx.hir().local_def_id(hir_id));
        }

        self.check_repr(attrs, span, target, item, hir_id);
        self.check_used(attrs, target);
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

    /// Checks if a `#[track_caller]` is applied to a non-naked function. Returns `true` if valid.
    fn check_track_caller(
        &self,
        attr_span: &Span,
        attrs: &'hir [Attribute],
        span: &Span,
        target: Target,
    ) -> bool {
        match target {
            _ if self.tcx.sess.contains_name(attrs, sym::naked) => {
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
    fn check_non_exhaustive(&self, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Struct | Target::Enum => true,
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
    fn check_marker(&self, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Trait => true,
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

    fn check_doc_alias(&self, attr: &Attribute, hir_id: HirId, target: Target) -> bool {
        if let Some(mi) = attr.meta() {
            if let Some(list) = mi.meta_item_list() {
                for meta in list {
                    if meta.has_name(sym::alias) {
                        if !meta.is_value_str()
                            || meta
                                .value_str()
                                .map(|s| s.to_string())
                                .unwrap_or_else(String::new)
                                .is_empty()
                        {
                            self.tcx
                                .sess
                                .struct_span_err(
                                    meta.span(),
                                    "doc alias attribute expects a string: #[doc(alias = \"0\")]",
                                )
                                .emit();
                            return false;
                        }
                        if let Some(err) = match target {
                            Target::Impl => Some("implementation block"),
                            Target::ForeignMod => Some("extern block"),
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
                                    ItemKind::Impl { of_trait: Some(_), .. } => Some(err),
                                    _ => None,
                                }
                            }
                            _ => None,
                        } {
                            self.tcx
                                .sess
                                .struct_span_err(
                                    meta.span(),
                                    &format!("`#[doc(alias = \"...\")]` isn't allowed on {}", err),
                                )
                                .emit();
                        }
                    }
                }
            }
        }
        true
    }

    /// Checks if `#[cold]` is applied to a non-function. Returns `true` if valid.
    fn check_cold(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) {
        match target {
            Target::Fn | Target::Method(..) | Target::ForeignFn | Target::Closure => {}
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
    fn check_no_link(&self, attr: &Attribute, span: &Span, target: Target) -> bool {
        if target == Target::ExternCrate {
            true
        } else {
            self.tcx
                .sess
                .struct_span_err(attr.span, "attribute should be applied to an `extern crate` item")
                .span_label(*span, "not an `extern crate` item")
                .emit();
            false
        }
    }

    /// Checks if `#[export_name]` is applied to a function or static. Returns `true` if valid.
    fn check_export_name(&self, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Static | Target::Fn | Target::Method(..) => true,
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(
                        attr.span,
                        "attribute should be applied to a function or static",
                    )
                    .span_label(*span, "not a function or static")
                    .emit();
                false
            }
        }
    }

    /// Checks if `#[rustc_args_required_const]` is applied to a function and has a valid argument.
    fn check_rustc_args_required_const(
        &self,
        attr: &Attribute,
        span: &Span,
        target: Target,
        item: Option<ItemLike<'_>>,
    ) -> bool {
        if let Target::Fn | Target::Method(..) | Target::ForeignFn = target {
            let mut invalid_args = vec![];
            for meta in attr.meta_item_list().expect("no meta item list") {
                if let Some(LitKind::Int(val, _)) = meta.literal().map(|lit| &lit.kind) {
                    if let Some(ItemLike::Item(Item {
                        kind: ItemKind::Fn(FnSig { decl, .. }, ..),
                        ..
                    }))
                    | Some(ItemLike::ForeignItem(ForeignItem {
                        kind: ForeignItemKind::Fn(decl, ..),
                        ..
                    })) = item
                    {
                        let arg_count = decl.inputs.len() as u128;
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
                        bug!("should be a function item");
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
        } else {
            self.tcx
                .sess
                .struct_span_err(attr.span, "attribute should be applied to a function")
                .span_label(*span, "not a function")
                .emit();
            false
        }
    }

    /// Checks if `#[link_section]` is applied to a function or static.
    fn check_link_section(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) {
        match target {
            Target::Static | Target::Fn | Target::Method(..) => {}
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
            Target::Static | Target::Fn | Target::Method(..) => {}
            _ => {
                // FIXME: #[no_mangle] was previously allowed on non-functions/statics and some
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

    /// Checks if the `#[repr]` attributes on `item` are valid.
    fn check_repr(
        &self,
        attrs: &'hir [Attribute],
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
            .filter(|attr| self.tcx.sess.check_name(attr, sym::repr))
            .filter_map(|attr| attr.meta_item_list())
            .flatten()
            .collect();

        let mut int_reprs = 0;
        let mut is_c = false;
        let mut is_simd = false;
        let mut is_transparent = false;

        for hint in &hints {
            let (article, allowed_targets) = match hint.name_or_empty() {
                name @ sym::C | name @ sym::align => {
                    is_c |= name == sym::C;
                    match target {
                        Target::Struct | Target::Union | Target::Enum => continue,
                        _ => ("a", "struct, enum, or union"),
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
                _ => continue,
            };
            self.emit_repr_error(
                hint.span(),
                *span,
                &format!("attribute should be applied to {}", allowed_targets),
                &format!("not {} {}", article, allowed_targets),
            )
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

    fn emit_repr_error(
        &self,
        hint_span: Span,
        label_span: Span,
        hint_message: &str,
        label_message: &str,
    ) {
        struct_span_err!(self.tcx.sess, hint_span, E0517, "{}", hint_message)
            .span_label(label_span, label_message)
            .emit();
    }

    fn check_stmt_attributes(&self, stmt: &hir::Stmt<'_>) {
        // When checking statements ignore expressions, they will be checked later
        if let hir::StmtKind::Local(ref l) = stmt.kind {
            self.check_attributes(l.hir_id, &l.attrs, &stmt.span, Target::Statement, None);
            for attr in l.attrs.iter() {
                if self.tcx.sess.check_name(attr, sym::repr) {
                    self.emit_repr_error(
                        attr.span,
                        stmt.span,
                        "attribute should not be applied to a statement",
                        "not a struct, enum, or union",
                    );
                }
            }
        }
    }

    fn check_expr_attributes(&self, expr: &hir::Expr<'_>) {
        let target = match expr.kind {
            hir::ExprKind::Closure(..) => Target::Closure,
            _ => Target::Expression,
        };
        self.check_attributes(expr.hir_id, &expr.attrs, &expr.span, target, None);
        for attr in expr.attrs.iter() {
            if self.tcx.sess.check_name(attr, sym::repr) {
                self.emit_repr_error(
                    attr.span,
                    expr.span,
                    "attribute should not be applied to an expression",
                    "not defining a struct, enum, or union",
                );
            }
        }
        if target == Target::Closure {
            self.tcx.ensure().codegen_fn_attrs(self.tcx.hir().local_def_id(expr.hir_id));
        }
    }

    fn check_used(&self, attrs: &'hir [Attribute], target: Target) {
        for attr in attrs {
            if self.tcx.sess.check_name(attr, sym::used) && target != Target::Static {
                self.tcx
                    .sess
                    .span_err(attr.span, "attribute must be applied to a `static` variable");
            }
        }
    }
}

impl Visitor<'tcx> for CheckAttrVisitor<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        let target = Target::from_item(item);
        self.check_attributes(
            item.hir_id,
            item.attrs,
            &item.span,
            target,
            Some(ItemLike::Item(item)),
        );
        intravisit::walk_item(self, item)
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx TraitItem<'tcx>) {
        let target = Target::from_trait_item(trait_item);
        self.check_attributes(trait_item.hir_id, &trait_item.attrs, &trait_item.span, target, None);
        intravisit::walk_trait_item(self, trait_item)
    }

    fn visit_foreign_item(&mut self, f_item: &'tcx ForeignItem<'tcx>) {
        let target = Target::from_foreign_item(f_item);
        self.check_attributes(
            f_item.hir_id,
            &f_item.attrs,
            &f_item.span,
            target,
            Some(ItemLike::ForeignItem(f_item)),
        );
        intravisit::walk_foreign_item(self, f_item)
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        let target = target_from_impl_item(self.tcx, impl_item);
        self.check_attributes(impl_item.hir_id, &impl_item.attrs, &impl_item.span, target, None);
        intravisit::walk_impl_item(self, impl_item)
    }

    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
        self.check_stmt_attributes(stmt);
        intravisit::walk_stmt(self, stmt)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        self.check_expr_attributes(expr);
        intravisit::walk_expr(self, expr)
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

fn check_mod_attrs(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir()
        .visit_item_likes_in_module(module_def_id, &mut CheckAttrVisitor { tcx }.as_deep_visitor());
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_attrs, ..*providers };
}

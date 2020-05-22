//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use rustc_middle::hir::map::Map;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;

use rustc_ast::ast::{Attribute, NestedMetaItem};
use rustc_ast::attr;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{self, HirId, Item, ItemKind, TraitItem};
use rustc_hir::{MethodKind, Target};
use rustc_session::lint::builtin::{CONFLICTING_REPR_HINTS, UNUSED_ATTRIBUTES};
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::Span;

fn target_from_impl_item<'tcx>(tcx: TyCtxt<'tcx>, impl_item: &hir::ImplItem<'_>) -> Target {
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
        hir::ImplItemKind::TyAlias(..) | hir::ImplItemKind::OpaqueTy(..) => Target::AssocTy,
    }
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
        item: Option<&Item<'_>>,
    ) {
        let mut is_valid = true;
        for attr in attrs {
            is_valid &= if attr.check_name(sym::inline) {
                self.check_inline(hir_id, attr, span, target)
            } else if attr.check_name(sym::non_exhaustive) {
                self.check_non_exhaustive(attr, span, target)
            } else if attr.check_name(sym::marker) {
                self.check_marker(attr, span, target)
            } else if attr.check_name(sym::target_feature) {
                self.check_target_feature(attr, span, target)
            } else if attr.check_name(sym::track_caller) {
                self.check_track_caller(&attr.span, attrs, span, target)
            } else {
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
            _ if attr::contains_name(attrs, sym::naked) => {
                struct_span_err!(
                    self.tcx.sess,
                    *attr_span,
                    E0736,
                    "cannot use `#[track_caller]` with `#[naked]`",
                )
                .emit();
                false
            }
            Target::Fn | Target::Method(..) | Target::ForeignFn => true,
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
    fn check_target_feature(&self, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => true,
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

    /// Checks if the `#[repr]` attributes on `item` are valid.
    fn check_repr(
        &self,
        attrs: &'hir [Attribute],
        span: &Span,
        target: Target,
        item: Option<&Item<'_>>,
        hir_id: HirId,
    ) {
        // Extract the names of all repr hints, e.g., [foo, bar, align] for:
        // ```
        // #[repr(foo)]
        // #[repr(bar, align(8))]
        // ```
        let hints: Vec<_> = attrs
            .iter()
            .filter(|attr| attr.check_name(sym::repr))
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
                    if target != Target::Struct { ("a", "struct") } else { continue }
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
                | sym::isize
                | sym::usize => {
                    int_reprs += 1;
                    if target != Target::Enum { ("an", "enum") } else { continue }
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
            || (int_reprs == 1 && is_c && item.map_or(false, |item| is_c_like_enum(item)))
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
            for attr in l.attrs.iter() {
                if attr.check_name(sym::inline) {
                    self.check_inline(l.hir_id, attr, &stmt.span, Target::Statement);
                }
                if attr.check_name(sym::repr) {
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
        for attr in expr.attrs.iter() {
            if attr.check_name(sym::inline) {
                self.check_inline(expr.hir_id, attr, &expr.span, target);
            }
            if attr.check_name(sym::repr) {
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
            if attr.check_name(sym::used) && target != Target::Static {
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
        self.check_attributes(item.hir_id, item.attrs, &item.span, target, Some(item));
        intravisit::walk_item(self, item)
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx TraitItem<'tcx>) {
        let target = Target::from_trait_item(trait_item);
        self.check_attributes(trait_item.hir_id, &trait_item.attrs, &trait_item.span, target, None);
        intravisit::walk_trait_item(self, trait_item)
    }

    fn visit_foreign_item(&mut self, f_item: &'tcx hir::ForeignItem<'tcx>) {
        let target = Target::from_foreign_item(f_item);
        self.check_attributes(f_item.hir_id, &f_item.attrs, &f_item.span, target, None);
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

fn check_mod_attrs(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir()
        .visit_item_likes_in_module(module_def_id, &mut CheckAttrVisitor { tcx }.as_deep_visitor());
}

pub(crate) fn provide(providers: &mut Providers<'_>) {
    *providers = Providers { check_mod_attrs, ..*providers };
}

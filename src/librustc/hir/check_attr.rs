//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use crate::hir::def_id::DefId;
use crate::hir::intravisit::{self, NestedVisitorMap, Visitor};
use crate::hir::DUMMY_HIR_ID;
use crate::hir::{self, HirId, Item, ItemKind, TraitItem, TraitItemKind};
use crate::lint::builtin::UNUSED_ATTRIBUTES;
use crate::ty::query::Providers;
use crate::ty::TyCtxt;

use rustc_error_codes::*;
use rustc_span::symbol::sym;
use rustc_span::Span;
use syntax::ast::Attribute;
use syntax::attr;

use std::fmt::{self, Display};

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum MethodKind {
    Trait { body: bool },
    Inherent,
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum Target {
    ExternCrate,
    Use,
    Static,
    Const,
    Fn,
    Closure,
    Mod,
    ForeignMod,
    GlobalAsm,
    TyAlias,
    OpaqueTy,
    Enum,
    Struct,
    Union,
    Trait,
    TraitAlias,
    Impl,
    Expression,
    Statement,
    AssocConst,
    Method(MethodKind),
    AssocTy,
    ForeignFn,
    ForeignStatic,
    ForeignTy,
}

impl Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                Target::ExternCrate => "extern crate",
                Target::Use => "use",
                Target::Static => "static item",
                Target::Const => "constant item",
                Target::Fn => "function",
                Target::Closure => "closure",
                Target::Mod => "module",
                Target::ForeignMod => "foreign module",
                Target::GlobalAsm => "global asm",
                Target::TyAlias => "type alias",
                Target::OpaqueTy => "opaque type",
                Target::Enum => "enum",
                Target::Struct => "struct",
                Target::Union => "union",
                Target::Trait => "trait",
                Target::TraitAlias => "trait alias",
                Target::Impl => "item",
                Target::Expression => "expression",
                Target::Statement => "statement",
                Target::AssocConst => "associated const",
                Target::Method(_) => "method",
                Target::AssocTy => "associated type",
                Target::ForeignFn => "foreign function",
                Target::ForeignStatic => "foreign static item",
                Target::ForeignTy => "foreign type",
            }
        )
    }
}

impl Target {
    pub(crate) fn from_item(item: &Item<'_>) -> Target {
        match item.kind {
            ItemKind::ExternCrate(..) => Target::ExternCrate,
            ItemKind::Use(..) => Target::Use,
            ItemKind::Static(..) => Target::Static,
            ItemKind::Const(..) => Target::Const,
            ItemKind::Fn(..) => Target::Fn,
            ItemKind::Mod(..) => Target::Mod,
            ItemKind::ForeignMod(..) => Target::ForeignMod,
            ItemKind::GlobalAsm(..) => Target::GlobalAsm,
            ItemKind::TyAlias(..) => Target::TyAlias,
            ItemKind::OpaqueTy(..) => Target::OpaqueTy,
            ItemKind::Enum(..) => Target::Enum,
            ItemKind::Struct(..) => Target::Struct,
            ItemKind::Union(..) => Target::Union,
            ItemKind::Trait(..) => Target::Trait,
            ItemKind::TraitAlias(..) => Target::TraitAlias,
            ItemKind::Impl(..) => Target::Impl,
        }
    }

    fn from_trait_item(trait_item: &TraitItem<'_>) -> Target {
        match trait_item.kind {
            TraitItemKind::Const(..) => Target::AssocConst,
            TraitItemKind::Method(_, hir::TraitMethod::Required(_)) => {
                Target::Method(MethodKind::Trait { body: false })
            }
            TraitItemKind::Method(_, hir::TraitMethod::Provided(_)) => {
                Target::Method(MethodKind::Trait { body: true })
            }
            TraitItemKind::Type(..) => Target::AssocTy,
        }
    }

    fn from_foreign_item(foreign_item: &hir::ForeignItem<'_>) -> Target {
        match foreign_item.kind {
            hir::ForeignItemKind::Fn(..) => Target::ForeignFn,
            hir::ForeignItemKind::Static(..) => Target::ForeignStatic,
            hir::ForeignItemKind::Type => Target::ForeignTy,
        }
    }

    fn from_impl_item<'tcx>(tcx: TyCtxt<'tcx>, impl_item: &hir::ImplItem<'_>) -> Target {
        match impl_item.kind {
            hir::ImplItemKind::Const(..) => Target::AssocConst,
            hir::ImplItemKind::Method(..) => {
                let parent_hir_id = tcx.hir().get_parent_item(impl_item.hir_id);
                let containing_item = tcx.hir().expect_item(parent_hir_id);
                let containing_impl_is_for_trait = match &containing_item.kind {
                    hir::ItemKind::Impl(_, _, _, _, tr, _, _) => tr.is_some(),
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

        if target == Target::Fn {
            self.tcx.codegen_fn_attrs(self.tcx.hir().local_def_id(hir_id));
        }

        self.check_repr(attrs, span, target, item);
        self.check_used(attrs, target);
    }

    /// Checks if an `#[inline]` is applied to a function or a closure. Returns `true` if valid.
    fn check_inline(&self, hir_id: HirId, attr: &Attribute, span: &Span, target: Target) -> bool {
        match target {
            Target::Fn
            | Target::Closure
            | Target::Method(MethodKind::Trait { body: true })
            | Target::Method(MethodKind::Inherent) => true,
            Target::Method(MethodKind::Trait { body: false }) | Target::ForeignFn => {
                self.tcx
                    .struct_span_lint_hir(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        attr.span,
                        "`#[inline]` is ignored on function prototypes",
                    )
                    .emit();
                true
            }
            // FIXME(#65833): We permit associated consts to have an `#[inline]` attribute with
            // just a lint, because we previously erroneously allowed it and some crates used it
            // accidentally, to to be compatible with crates depending on them, we can't throw an
            // error here.
            Target::AssocConst => {
                self.tcx
                    .struct_span_lint_hir(
                        UNUSED_ATTRIBUTES,
                        hir_id,
                        attr.span,
                        "`#[inline]` is ignored on constants",
                    )
                    .warn(
                        "this was previously accepted by the compiler but is \
                       being phased out; it will become a hard error in \
                       a future release!",
                    )
                    .note(
                        "for more information, see issue #65833 \
                       <https://github.com/rust-lang/rust/issues/65833>",
                    )
                    .emit();
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
            Target::Fn if attr::contains_name(attrs, sym::naked) => {
                struct_span_err!(
                    self.tcx.sess,
                    *attr_span,
                    E0736,
                    "cannot use `#[track_caller]` with `#[naked]`",
                )
                .emit();
                false
            }
            Target::Fn | Target::Method(MethodKind::Inherent) => true,
            Target::Method(_) => {
                struct_span_err!(
                    self.tcx.sess,
                    *attr_span,
                    E0738,
                    "`#[track_caller]` may not be used on trait methods",
                )
                .emit();
                false
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
            | Target::Method(MethodKind::Trait { body: true })
            | Target::Method(MethodKind::Inherent) => true,
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

        // Error on repr(transparent, <anything else>).
        if is_transparent && hints.len() > 1 {
            let hint_spans: Vec<_> = hint_spans.clone().collect();
            span_err!(
                self.tcx.sess,
                hint_spans,
                E0692,
                "transparent {} cannot have other repr hints",
                target
            );
        }
        // Warn on repr(u8, u16), repr(C, simd), and c-like-enum-repr(C, u8)
        if (int_reprs > 1)
            || (is_simd && is_c)
            || (int_reprs == 1 && is_c && item.map_or(false, |item| is_c_like_enum(item)))
        {
            let hint_spans: Vec<_> = hint_spans.collect();
            span_warn!(self.tcx.sess, hint_spans, E0566, "conflicting representation hints");
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
                    self.check_inline(DUMMY_HIR_ID, attr, &stmt.span, Target::Statement);
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
                self.check_inline(DUMMY_HIR_ID, attr, &expr.span, target);
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
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
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
        let target = Target::from_impl_item(self.tcx, impl_item);
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

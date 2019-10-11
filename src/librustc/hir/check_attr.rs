//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use crate::hir::{self, HirId, HirVec, Attribute, Item, ItemKind, TraitItem, TraitItemKind};
use crate::hir::DUMMY_HIR_ID;
use crate::hir::def_id::DefId;
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::ty::TyCtxt;
use crate::ty::query::Providers;

use std::fmt::{self, Display};
use syntax::{attr, symbol::sym};
use syntax_pos::Span;

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
}

impl Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match *self {
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
        })
    }
}

impl Target {
    pub(crate) fn from_item(item: &Item) -> Target {
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
        attrs: &HirVec<Attribute>,
        span: &Span,
        target: Target,
        item: Option<&Item>,
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
                self.check_track_caller(attr, &item, target)
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
            Target::Fn | Target::Closure | Target::Method { body: true } => true,
            _ => {
                struct_span_err!(self.tcx.sess,
                                 attr.span,
                                 E0518,
                                 "attribute should be applied to function or closure")
                    .span_label(*span, "not a function or closure")
                    .emit();
                false
            }
        }
    }

    /// Checks if a `#[track_caller]` is applied to a non-naked function. Returns `true` if valid.
    fn check_track_caller(&self, attr: &hir::Attribute, item: &hir::Item, target: Target) -> bool {
        if target != Target::Fn {
            struct_span_err!(
                self.tcx.sess,
                attr.span,
                E0739,
                "attribute should be applied to function"
            )
            .span_label(item.span, "not a function")
            .emit();
            false
        } else if attr::contains_name(&item.attrs, sym::naked) {
            struct_span_err!(
                self.tcx.sess,
                attr.span,
                E0736,
                "cannot use `#[track_caller]` with `#[naked]`",
            )
            .emit();
            false
        } else {
            true
        }
    }

    /// Checks if the `#[non_exhaustive]` attribute on an `item` is valid. Returns `true` if valid.
    fn check_non_exhaustive(
        &self,
        attr: &Attribute,
        span: &Span,
        target: Target,
    ) -> bool {
        match target {
            Target::Struct | Target::Enum => true,
            _ => {
                struct_span_err!(self.tcx.sess,
                                 attr.span,
                                 E0701,
                                 "attribute can only be applied to a struct or enum")
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
                self.tcx.sess
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
            Target::Fn => true,
            _ => {
                self.tcx.sess
                    .struct_span_err(attr.span, "attribute should be applied to a function")
                    .span_label(*span, "not a function")
                    .emit();
                false
            },
        }
    }

    /// Checks if the `#[repr]` attributes on `item` are valid.
    fn check_repr(
        &self,
        attrs: &HirVec<Attribute>,
        span: &Span,
        target: Target,
        item: Option<&Item>,
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
                    if target != Target::Struct &&
                            target != Target::Union {
                                ("a", "struct or union")
                    } else {
                        continue
                    }
                }
                sym::simd => {
                    is_simd = true;
                    if target != Target::Struct {
                        ("a", "struct")
                    } else {
                        continue
                    }
                }
                sym::transparent => {
                    is_transparent = true;
                    match target {
                        Target::Struct | Target::Union | Target::Enum => continue,
                        _ => ("a", "struct, enum, or union"),
                    }
                }
                sym::i8  | sym::u8  | sym::i16 | sym::u16 |
                sym::i32 | sym::u32 | sym::i64 | sym::u64 |
                sym::isize | sym::usize => {
                    int_reprs += 1;
                    if target != Target::Enum {
                        ("an", "enum")
                    } else {
                        continue
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

        // Error on repr(transparent, <anything else>).
        if is_transparent && hints.len() > 1 {
            let hint_spans: Vec<_> = hint_spans.clone().collect();
            span_err!(self.tcx.sess, hint_spans, E0692,
                      "transparent {} cannot have other repr hints", target);
        }
        // Warn on repr(u8, u16), repr(C, simd), and c-like-enum-repr(C, u8)
        if (int_reprs > 1)
           || (is_simd && is_c)
           || (int_reprs == 1 && is_c && item.map(|item| is_c_like_enum(item)).unwrap_or(false)) {
            let hint_spans: Vec<_> = hint_spans.collect();
            span_warn!(self.tcx.sess, hint_spans, E0566,
                       "conflicting representation hints");
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

    fn check_stmt_attributes(&self, stmt: &hir::Stmt) {
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

    fn check_expr_attributes(&self, expr: &hir::Expr) {
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

    fn check_used(&self, attrs: &HirVec<Attribute>, target: Target) {
        for attr in attrs {
            if attr.check_name(sym::used) && target != Target::Static {
                self.tcx.sess
                    .span_err(attr.span, "attribute must be applied to a `static` variable");
            }
        }
    }
}

impl Visitor<'tcx> for CheckAttrVisitor<'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx Item) {
        let target = Target::from_item(item);
        self.check_attributes(item.hir_id, &item.attrs, &item.span, target, Some(item));
        intravisit::walk_item(self, item)
    }


    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt) {
        self.check_stmt_attributes(stmt);
        intravisit::walk_stmt(self, stmt)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        self.check_expr_attributes(expr);
        intravisit::walk_expr(self, expr)
    }
}

fn is_c_like_enum(item: &Item) -> bool {
    if let ItemKind::Enum(ref def, _) = item.kind {
        for variant in &def.variants {
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
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut CheckAttrVisitor { tcx }.as_deep_visitor()
    );
}

pub(crate) fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        check_mod_attrs,
        ..*providers
    };
}

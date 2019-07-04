//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.


use crate::ty::TyCtxt;
use crate::ty::query::Providers;

use crate::hir;
use crate::hir::def_id::DefId;
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use std::fmt::{self, Display};
use syntax::symbol::sym;
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
    Ty,
    Existential,
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
            Target::Ty => "type alias",
            Target::Existential => "existential type",
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
    pub(crate) fn from_item(item: &hir::Item) -> Target {
        match item.node {
            hir::ItemKind::ExternCrate(..) => Target::ExternCrate,
            hir::ItemKind::Use(..) => Target::Use,
            hir::ItemKind::Static(..) => Target::Static,
            hir::ItemKind::Const(..) => Target::Const,
            hir::ItemKind::Fn(..) => Target::Fn,
            hir::ItemKind::Mod(..) => Target::Mod,
            hir::ItemKind::ForeignMod(..) => Target::ForeignMod,
            hir::ItemKind::GlobalAsm(..) => Target::GlobalAsm,
            hir::ItemKind::Ty(..) => Target::Ty,
            hir::ItemKind::Existential(..) => Target::Existential,
            hir::ItemKind::Enum(..) => Target::Enum,
            hir::ItemKind::Struct(..) => Target::Struct,
            hir::ItemKind::Union(..) => Target::Union,
            hir::ItemKind::Trait(..) => Target::Trait,
            hir::ItemKind::TraitAlias(..) => Target::TraitAlias,
            hir::ItemKind::Impl(..) => Target::Impl,
        }
    }
}

struct CheckAttrVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl CheckAttrVisitor<'tcx> {
    /// Checks any attribute.
    fn check_attributes(&self, item: &hir::Item, target: Target) {
        if target == Target::Fn || target == Target::Const {
            self.tcx.codegen_fn_attrs(self.tcx.hir().local_def_id_from_hir_id(item.hir_id));
        } else if let Some(a) = item.attrs.iter().find(|a| a.check_name(sym::target_feature)) {
            self.tcx.sess.struct_span_err(a.span, "attribute should be applied to a function")
                .span_label(item.span, "not a function")
                .emit();
        }

        for attr in &item.attrs {
            if attr.check_name(sym::inline) {
                self.check_inline(attr, &item.span, target)
            } else if attr.check_name(sym::non_exhaustive) {
                self.check_non_exhaustive(attr, item, target)
            } else if attr.check_name(sym::marker) {
                self.check_marker(attr, item, target)
            }
        }

        self.check_repr(item, target);
        self.check_used(item, target);
    }

    /// Checks if an `#[inline]` is applied to a function or a closure.
    fn check_inline(&self, attr: &hir::Attribute, span: &Span, target: Target) {
        if target != Target::Fn && target != Target::Closure {
            struct_span_err!(self.tcx.sess,
                             attr.span,
                             E0518,
                             "attribute should be applied to function or closure")
                .span_label(*span, "not a function or closure")
                .emit();
        }
    }

    /// Checks if the `#[non_exhaustive]` attribute on an `item` is valid.
    fn check_non_exhaustive(&self, attr: &hir::Attribute, item: &hir::Item, target: Target) {
        match target {
            Target::Struct | Target::Enum => { /* Valid */ },
            _ => {
                struct_span_err!(self.tcx.sess,
                                 attr.span,
                                 E0701,
                                 "attribute can only be applied to a struct or enum")
                    .span_label(item.span, "not a struct or enum")
                    .emit();
                return;
            }
        }
    }

    /// Checks if the `#[marker]` attribute on an `item` is valid.
    fn check_marker(&self, attr: &hir::Attribute, item: &hir::Item, target: Target) {
        match target {
            Target::Trait => { /* Valid */ },
            _ => {
                self.tcx.sess
                    .struct_span_err(attr.span, "attribute can only be applied to a trait")
                    .span_label(item.span, "not a trait")
                    .emit();
                return;
            }
        }
    }

    /// Checks if the `#[repr]` attributes on `item` are valid.
    fn check_repr(&self, item: &hir::Item, target: Target) {
        // Extract the names of all repr hints, e.g., [foo, bar, align] for:
        // ```
        // #[repr(foo)]
        // #[repr(bar, align(8))]
        // ```
        let hints: Vec<_> = item.attrs
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
                item.span,
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
           || (int_reprs == 1 && is_c && is_c_like_enum(item)) {
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
        if let hir::StmtKind::Local(ref l) = stmt.node {
            for attr in l.attrs.iter() {
                if attr.check_name(sym::inline) {
                    self.check_inline(attr, &stmt.span, Target::Statement);
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
        let target = match expr.node {
            hir::ExprKind::Closure(..) => Target::Closure,
            _ => Target::Expression,
        };
        for attr in expr.attrs.iter() {
            if attr.check_name(sym::inline) {
                self.check_inline(attr, &expr.span, target);
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

    fn check_used(&self, item: &hir::Item, target: Target) {
        for attr in &item.attrs {
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

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let target = Target::from_item(item);
        self.check_attributes(item, target);
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

fn is_c_like_enum(item: &hir::Item) -> bool {
    if let hir::ItemKind::Enum(ref def, _) = item.node {
        for variant in &def.variants {
            match variant.node.data {
                hir::VariantData::Unit(..) => { /* continue */ }
                _ => { return false; }
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

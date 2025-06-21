//! A `MutVisitor` represents an AST modification; it accepts an AST piece and
//! mutates it in place. So, for instance, macro expansion is a `MutVisitor`
//! that walks over an AST and modifies it.
//!
//! Note: using a `MutVisitor` (other than the `MacroExpander` `MutVisitor`) on
//! an AST before macro expansion is probably a bad idea. For instance,
//! a `MutVisitor` renaming item names in a module will miss all of those
//! that are created by the expansion of a macro.

use std::ops::DerefMut;
use std::panic;

use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span};
use smallvec::{SmallVec, smallvec};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::ptr::P;
use crate::tokenstream::*;
use crate::visit::{AssocCtxt, BoundKind, FnCtxt, VisitorResult, try_visit, visit_opt, walk_list};

mod sealed {
    use rustc_ast_ir::visit::VisitorResult;

    /// This is for compatibility with the regular `Visitor`.
    pub trait MutVisitorResult {
        type Result: VisitorResult;
    }

    impl<T> MutVisitorResult for T {
        type Result = ();
    }
}

use sealed::MutVisitorResult;

super::common_visitor_and_walkers!((mut) MutVisitor);

macro_rules! generate_flat_map_visitor_fns {
    ($($name:ident, $Ty:ty, $flat_map_fn:ident$(, $param:ident: $ParamTy:ty)*;)+) => {
        $(
            fn $name<V: MutVisitor>(
                vis: &mut V,
                values: &mut ThinVec<$Ty>,
                $(
                    $param: $ParamTy,
                )*
            ) {
                values.flat_map_in_place(|value| vis.$flat_map_fn(value$(,$param)*));
            }
        )+
    }
}

generate_flat_map_visitor_fns! {
    visit_items, P<Item>, flat_map_item;
    visit_foreign_items, P<ForeignItem>, flat_map_foreign_item;
    visit_generic_params, GenericParam, flat_map_generic_param;
    visit_stmts, Stmt, flat_map_stmt;
    visit_exprs, P<Expr>, filter_map_expr;
    visit_expr_fields, ExprField, flat_map_expr_field;
    visit_pat_fields, PatField, flat_map_pat_field;
    visit_variants, Variant, flat_map_variant;
    visit_assoc_items, P<AssocItem>, flat_map_assoc_item, ctxt: AssocCtxt;
    visit_where_predicates, WherePredicate, flat_map_where_predicate;
    visit_params, Param, flat_map_param;
    visit_field_defs, FieldDef, flat_map_field_def;
    visit_arms, Arm, flat_map_arm;
}

pub fn walk_flat_map_pat_field<T: MutVisitor>(
    vis: &mut T,
    mut fp: PatField,
) -> SmallVec<[PatField; 1]> {
    vis.visit_pat_field(&mut fp);
    smallvec![fp]
}

fn visit_nested_use_tree<V: MutVisitor>(
    vis: &mut V,
    nested_tree: &mut UseTree,
    nested_id: &mut NodeId,
) {
    vis.visit_id(nested_id);
    vis.visit_use_tree(nested_tree);
}

macro_rules! generate_walk_flat_map_fns {
    ($($fn_name:ident($Ty:ty$(,$extra_name:ident: $ExtraTy:ty)*) => $visit_fn_name:ident;)+) => {$(
        pub fn $fn_name<V: MutVisitor>(vis: &mut V, mut value: $Ty$(,$extra_name: $ExtraTy)*) -> SmallVec<[$Ty; 1]> {
            vis.$visit_fn_name(&mut value$(,$extra_name)*);
            smallvec![value]
        }
    )+};
}

generate_walk_flat_map_fns! {
    walk_flat_map_arm(Arm) => visit_arm;
    walk_flat_map_variant(Variant) => visit_variant;
    walk_flat_map_param(Param) => visit_param;
    walk_flat_map_generic_param(GenericParam) => visit_generic_param;
    walk_flat_map_where_predicate(WherePredicate) => visit_where_predicate;
    walk_flat_map_field_def(FieldDef) => visit_field_def;
    walk_flat_map_expr_field(ExprField) => visit_expr_field;
    walk_flat_map_item(P<Item>) => visit_item;
    walk_flat_map_foreign_item(P<ForeignItem>) => visit_foreign_item;
    walk_flat_map_assoc_item(P<AssocItem>, ctxt: AssocCtxt) => visit_assoc_item;
}

fn walk_ty_alias_where_clauses<T: MutVisitor>(vis: &mut T, tawcs: &mut TyAliasWhereClauses) {
    let TyAliasWhereClauses { before, after, split: _ } = tawcs;
    let TyAliasWhereClause { has_where_token: _, span: span_before } = before;
    let TyAliasWhereClause { has_where_token: _, span: span_after } = after;
    vis.visit_span(span_before);
    vis.visit_span(span_after);
}

pub fn walk_filter_map_expr<T: MutVisitor>(vis: &mut T, mut e: P<Expr>) -> Option<P<Expr>> {
    vis.visit_expr(&mut e);
    Some(e)
}

pub fn walk_flat_map_stmt<T: MutVisitor>(
    vis: &mut T,
    Stmt { kind, span, mut id }: Stmt,
) -> SmallVec<[Stmt; 1]> {
    vis.visit_id(&mut id);
    let mut stmts: SmallVec<[Stmt; 1]> = walk_flat_map_stmt_kind(vis, kind)
        .into_iter()
        .map(|kind| Stmt { id, kind, span })
        .collect();
    match &mut stmts[..] {
        [] => {}
        [stmt] => vis.visit_span(&mut stmt.span),
        _ => panic!(
            "cloning statement `NodeId`s is prohibited by default, \
             the visitor should implement custom statement visiting"
        ),
    }
    stmts
}

fn walk_flat_map_stmt_kind<T: MutVisitor>(vis: &mut T, kind: StmtKind) -> SmallVec<[StmtKind; 1]> {
    match kind {
        StmtKind::Let(mut local) => smallvec![StmtKind::Let({
            vis.visit_local(&mut local);
            local
        })],
        StmtKind::Item(item) => vis.flat_map_item(item).into_iter().map(StmtKind::Item).collect(),
        StmtKind::Expr(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Expr).collect(),
        StmtKind::Semi(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Semi).collect(),
        StmtKind::Empty => smallvec![StmtKind::Empty],
        StmtKind::MacCall(mut mac) => {
            let MacCallStmt { mac: mac_, style: _, attrs, tokens: _ } = mac.deref_mut();
            for attr in attrs {
                vis.visit_attribute(attr);
            }
            vis.visit_mac_call(mac_);
            smallvec![StmtKind::MacCall(mac)]
        }
    }
}

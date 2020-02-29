//! Assorted functions shared by several assists.
pub(crate) mod insert_use;

use hir::Semantics;
use ra_ide_db::RootDatabase;
use ra_syntax::{
    ast::{self, make, NameOwner},
    AstNode, T,
};
use rustc_hash::FxHashSet;

pub use insert_use::insert_use_statement;

pub fn get_missing_impl_items(
    sema: &Semantics<RootDatabase>,
    impl_def: &ast::ImplDef,
) -> Vec<hir::AssocItem> {
    // Names must be unique between constants and functions. However, type aliases
    // may share the same name as a function or constant.
    let mut impl_fns_consts = FxHashSet::default();
    let mut impl_type = FxHashSet::default();

    if let Some(item_list) = impl_def.item_list() {
        for item in item_list.impl_items() {
            match item {
                ast::ImplItem::FnDef(f) => {
                    if let Some(n) = f.name() {
                        impl_fns_consts.insert(n.syntax().to_string());
                    }
                }

                ast::ImplItem::TypeAliasDef(t) => {
                    if let Some(n) = t.name() {
                        impl_type.insert(n.syntax().to_string());
                    }
                }

                ast::ImplItem::ConstDef(c) => {
                    if let Some(n) = c.name() {
                        impl_fns_consts.insert(n.syntax().to_string());
                    }
                }
            }
        }
    }

    resolve_target_trait(sema, impl_def).map_or(vec![], |target_trait| {
        target_trait
            .items(sema.db)
            .iter()
            .filter(|i| match i {
                hir::AssocItem::Function(f) => {
                    !impl_fns_consts.contains(&f.name(sema.db).to_string())
                }
                hir::AssocItem::TypeAlias(t) => !impl_type.contains(&t.name(sema.db).to_string()),
                hir::AssocItem::Const(c) => c
                    .name(sema.db)
                    .map(|n| !impl_fns_consts.contains(&n.to_string()))
                    .unwrap_or_default(),
            })
            .cloned()
            .collect()
    })
}

pub(crate) fn resolve_target_trait(
    sema: &Semantics<RootDatabase>,
    impl_def: &ast::ImplDef,
) -> Option<hir::Trait> {
    let ast_path = impl_def
        .target_trait()
        .map(|it| it.syntax().clone())
        .and_then(ast::PathType::cast)?
        .path()?;

    match sema.resolve_path(&ast_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(def))) => Some(def),
        _ => None,
    }
}

pub(crate) fn invert_boolean_expression(expr: ast::Expr) -> ast::Expr {
    if let Some(expr) = invert_special_case(&expr) {
        return expr;
    }
    make::expr_prefix(T![!], expr)
}

fn invert_special_case(expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => match bin.op_kind()? {
            ast::BinOp::NegatedEqualityTest => bin.replace_op(T![==]).map(|it| it.into()),
            ast::BinOp::EqualityTest => bin.replace_op(T![!=]).map(|it| it.into()),
            _ => None,
        },
        ast::Expr::PrefixExpr(pe) if pe.op_kind()? == ast::PrefixOp::Not => pe.expr(),
        // FIXME:
        // ast::Expr::Literal(true | false )
        _ => None,
    }
}

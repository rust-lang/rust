//! Assorted functions shared by several assists.

use ra_syntax::{
    ast::{self, make, NameOwner},
    AstNode, T,
};

use hir::db::HirDatabase;

use rustc_hash::FxHashSet;

pub fn get_missing_impl_items(
    db: &impl HirDatabase,
    analyzer: &hir::SourceAnalyzer,
    impl_block: &ast::ImplBlock,
) -> Vec<hir::AssocItem> {
    // since the names are unique only to each associated type (fn/type/const),
    // create buckets of each already implemented type that we'll use in the
    // lookup later.
    let mut impl_fns = FxHashSet::default();
    let mut impl_type = FxHashSet::default();
    let mut impl_const = FxHashSet::default();

    if let Some(item_list) = impl_block.item_list() {
        for item in item_list.impl_items() {
            match item {
                ast::ImplItem::FnDef(f) => {
                    if let Some(n) = f.name() {
                        impl_fns.insert(n.syntax().to_string());
                    }
                }

                ast::ImplItem::TypeAliasDef(t) => {
                    if let Some(n) = t.name() {
                        impl_type.insert(n.syntax().to_string());
                    }
                }

                ast::ImplItem::ConstDef(c) => {
                    if let Some(n) = c.name() {
                        impl_const.insert(n.syntax().to_string());
                    }
                }
            }
        }
    }

    resolve_target_trait(db, analyzer, impl_block).map_or(vec![], |target_trait| {
        target_trait
            .items(db)
            .iter()
            .filter(|i| match i {
                hir::AssocItem::Function(f) => !impl_fns.contains(&f.name(db).to_string()),
                hir::AssocItem::TypeAlias(t) => !impl_type.contains(&t.name(db).to_string()),
                hir::AssocItem::Const(c) => {
                    c.name(db).map(|n| !impl_const.contains(&n.to_string())).unwrap_or_default()
                }
            })
            .map(|i| i.clone())
            .collect()
    })
}

fn resolve_target_trait(
    db: &impl HirDatabase,
    analyzer: &hir::SourceAnalyzer,
    impl_block: &ast::ImplBlock,
) -> Option<hir::Trait> {
    let ast_path = impl_block
        .target_trait()
        .map(|it| it.syntax().clone())
        .and_then(ast::PathType::cast)?
        .path()?;

    match analyzer.resolve_path(db, &ast_path) {
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

use itertools::Itertools;
use rustc_hash::FxHashMap;

use hir::{PathResolution, Semantics};
use ide_db::RootDatabase;
use syntax::{
    algo,
    ast::{self, NameOwner},
    AstNode,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: reorder_impl
//
// Reorder the methods of an `impl Trait`. The methods will be ordered
// in the same order as in the trait definition.
//
// ```
// trait Foo {
//     fn a() {}
//     fn b() {}
//     fn c() {}
// }
//
// struct Bar;
// $0impl Foo for Bar {
//     fn b() {}
//     fn c() {}
//     fn a() {}
// }
// ```
// ->
// ```
// trait Foo {
//     fn a() {}
//     fn b() {}
//     fn c() {}
// }
//
// struct Bar;
// impl Foo for Bar {
//     fn a() {}
//     fn b() {}
//     fn c() {}
// }
// ```
//
pub(crate) fn reorder_impl(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let impl_ast = ctx.find_node_at_offset::<ast::Impl>()?;
    let items = impl_ast.assoc_item_list()?;
    let methods = get_methods(&items);

    let path = impl_ast
        .trait_()
        .and_then(|t| match t {
            ast::Type::PathType(path) => Some(path),
            _ => None,
        })?
        .path()?;

    let ranks = compute_method_ranks(&path, ctx)?;
    let sorted: Vec<_> = methods
        .iter()
        .cloned()
        .sorted_by_key(|f| {
            f.name().and_then(|n| ranks.get(&n.to_string()).copied()).unwrap_or(usize::max_value())
        })
        .collect();

    // Don't edit already sorted methods:
    if methods == sorted {
        cov_mark::hit!(not_applicable_if_sorted);
        return None;
    }

    let target = items.syntax().text_range();
    acc.add(AssistId("reorder_impl", AssistKind::RefactorRewrite), "Sort methods", target, |edit| {
        let mut rewriter = algo::SyntaxRewriter::default();
        for (old, new) in methods.iter().zip(&sorted) {
            rewriter.replace(old.syntax(), new.syntax());
        }
        edit.rewrite(rewriter);
    })
}

fn compute_method_ranks(path: &ast::Path, ctx: &AssistContext) -> Option<FxHashMap<String, usize>> {
    let td = trait_definition(path, &ctx.sema)?;

    Some(
        td.items(ctx.db())
            .iter()
            .flat_map(|i| match i {
                hir::AssocItem::Function(f) => Some(f),
                _ => None,
            })
            .enumerate()
            .map(|(idx, func)| ((func.name(ctx.db()).to_string(), idx)))
            .collect(),
    )
}

fn trait_definition(path: &ast::Path, sema: &Semantics<RootDatabase>) -> Option<hir::Trait> {
    match sema.resolve_path(path)? {
        PathResolution::Def(hir::ModuleDef::Trait(trait_)) => Some(trait_),
        _ => None,
    }
}

fn get_methods(items: &ast::AssocItemList) -> Vec<ast::Fn> {
    items
        .assoc_items()
        .flat_map(|i| match i {
            ast::AssocItem::Fn(f) => Some(f),
            _ => None,
        })
        .filter(|f| f.name().is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_if_sorted() {
        cov_mark::check!(not_applicable_if_sorted);
        check_assist_not_applicable(
            reorder_impl,
            r#"
trait Bar {
    fn a() {}
    fn z() {}
    fn b() {}
}
struct Foo;
$0impl Bar for Foo {
    fn a() {}
    fn z() {}
    fn b() {}
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_empty() {
        check_assist_not_applicable(
            reorder_impl,
            r#"
trait Bar {};
struct Foo;
$0impl Bar for Foo {}
        "#,
        )
    }

    #[test]
    fn reorder_impl_trait_methods() {
        check_assist(
            reorder_impl,
            r#"
trait Bar {
    fn a() {}
    fn c() {}
    fn b() {}
    fn d() {}
}

struct Foo;
$0impl Bar for Foo {
    fn d() {}
    fn b() {}
    fn c() {}
    fn a() {}
}
        "#,
            r#"
trait Bar {
    fn a() {}
    fn c() {}
    fn b() {}
    fn d() {}
}

struct Foo;
impl Bar for Foo {
    fn a() {}
    fn c() {}
    fn b() {}
    fn d() {}
}
        "#,
        )
    }
}

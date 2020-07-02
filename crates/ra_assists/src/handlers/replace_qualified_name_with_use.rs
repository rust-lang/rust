use hir;
use ra_syntax::{algo::SyntaxRewriter, ast, match_ast, AstNode, SmolStr, SyntaxNode};

use crate::{
    utils::{find_insert_use_container, insert_use_statement},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: replace_qualified_name_with_use
//
// Adds a use statement for a given fully-qualified name.
//
// ```
// fn process(map: std::collections::<|>HashMap<String, String>) {}
// ```
// ->
// ```
// use std::collections::HashMap;
//
// fn process(map: HashMap<String, String>) {}
// ```
pub(crate) fn replace_qualified_name_with_use(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let path: ast::Path = ctx.find_node_at_offset()?;
    // We don't want to mess with use statements
    if path.syntax().ancestors().find_map(ast::UseItem::cast).is_some() {
        return None;
    }

    let hir_path = ctx.sema.lower_path(&path)?;
    let segments = collect_hir_path_segments(&hir_path)?;
    if segments.len() < 2 {
        return None;
    }

    let target = path.syntax().text_range();
    acc.add(
        AssistId("replace_qualified_name_with_use", AssistKind::RefactorRewrite),
        "Replace qualified path with use",
        target,
        |builder| {
            let path_to_import = hir_path.mod_path().clone();
            let container = match find_insert_use_container(path.syntax(), ctx) {
                Some(c) => c,
                None => return,
            };
            insert_use_statement(path.syntax(), &path_to_import, ctx, builder.text_edit_builder());

            // Now that we've brought the name into scope, re-qualify all paths that could be
            // affected (that is, all paths inside the node we added the `use` to).
            let mut rewriter = SyntaxRewriter::default();
            let syntax = container.either(|l| l.syntax().clone(), |r| r.syntax().clone());
            shorten_paths(&mut rewriter, syntax, path);
            builder.rewrite(rewriter);
        },
    )
}

fn collect_hir_path_segments(path: &hir::Path) -> Option<Vec<SmolStr>> {
    let mut ps = Vec::<SmolStr>::with_capacity(10);
    match path.kind() {
        hir::PathKind::Abs => ps.push("".into()),
        hir::PathKind::Crate => ps.push("crate".into()),
        hir::PathKind::Plain => {}
        hir::PathKind::Super(0) => ps.push("self".into()),
        hir::PathKind::Super(lvl) => {
            let mut chain = "super".to_string();
            for _ in 0..*lvl {
                chain += "::super";
            }
            ps.push(chain.into());
        }
        hir::PathKind::DollarCrate(_) => return None,
    }
    ps.extend(path.segments().iter().map(|it| it.name.to_string().into()));
    Some(ps)
}

/// Adds replacements to `re` that shorten `path` in all descendants of `node`.
fn shorten_paths(rewriter: &mut SyntaxRewriter<'static>, node: SyntaxNode, path: ast::Path) {
    for child in node.children() {
        match_ast! {
            match child {
                // Don't modify `use` items, as this can break the `use` item when injecting a new
                // import into the use tree.
                ast::UseItem(_it) => continue,
                // Don't descend into submodules, they don't have the same `use` items in scope.
                ast::Module(_it) => continue,

                ast::Path(p) => {
                    match maybe_replace_path(rewriter, p.clone(), path.clone()) {
                        Some(()) => {},
                        None => shorten_paths(rewriter, p.syntax().clone(), path.clone()),
                    }
                },
                _ => shorten_paths(rewriter, child, path.clone()),
            }
        }
    }
}

fn maybe_replace_path(
    rewriter: &mut SyntaxRewriter<'static>,
    path: ast::Path,
    target: ast::Path,
) -> Option<()> {
    if !path_eq(path.clone(), target.clone()) {
        return None;
    }

    // Shorten `path`, leaving only its last segment.
    if let Some(parent) = path.qualifier() {
        rewriter.delete(parent.syntax());
    }
    if let Some(double_colon) = path.coloncolon_token() {
        rewriter.delete(&double_colon);
    }

    Some(())
}

fn path_eq(lhs: ast::Path, rhs: ast::Path) -> bool {
    let mut lhs_curr = lhs;
    let mut rhs_curr = rhs;
    loop {
        match (lhs_curr.segment(), rhs_curr.segment()) {
            (Some(lhs), Some(rhs)) if lhs.syntax().text() == rhs.syntax().text() => (),
            _ => return false,
        }

        match (lhs_curr.qualifier(), rhs_curr.qualifier()) {
            (Some(lhs), Some(rhs)) => {
                lhs_curr = lhs;
                rhs_curr = rhs;
            }
            (None, None) => return true,
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_replace_add_use_no_anchor() {
        check_assist(
            replace_qualified_name_with_use,
            r"
std::fmt::Debug<|>
    ",
            r"
use std::fmt::Debug;

Debug
    ",
        );
    }
    #[test]
    fn test_replace_add_use_no_anchor_with_item_below() {
        check_assist(
            replace_qualified_name_with_use,
            r"
std::fmt::Debug<|>

fn main() {
}
    ",
            r"
use std::fmt::Debug;

Debug

fn main() {
}
    ",
        );
    }

    #[test]
    fn test_replace_add_use_no_anchor_with_item_above() {
        check_assist(
            replace_qualified_name_with_use,
            r"
fn main() {
}

std::fmt::Debug<|>
    ",
            r"
use std::fmt::Debug;

fn main() {
}

Debug
    ",
        );
    }

    #[test]
    fn test_replace_add_use_no_anchor_2seg() {
        check_assist(
            replace_qualified_name_with_use,
            r"
std::fmt<|>::Debug
    ",
            r"
use std::fmt;

fmt::Debug
    ",
        );
    }

    #[test]
    fn test_replace_add_use() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use stdx;

impl std::fmt::Debug<|> for Foo {
}
    ",
            r"
use stdx;
use std::fmt::Debug;

impl Debug for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_file_use_other_anchor() {
        check_assist(
            replace_qualified_name_with_use,
            r"
impl std::fmt::Debug<|> for Foo {
}
    ",
            r"
use std::fmt::Debug;

impl Debug for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_add_use_other_anchor_indent() {
        check_assist(
            replace_qualified_name_with_use,
            r"
    impl std::fmt::Debug<|> for Foo {
    }
    ",
            r"
    use std::fmt::Debug;

    impl Debug for Foo {
    }
    ",
        );
    }

    #[test]
    fn test_replace_split_different() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt;

impl std::io<|> for Foo {
}
    ",
            r"
use std::{io, fmt};

impl io for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_split_self_for_use() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt;

impl std::fmt::Debug<|> for Foo {
}
    ",
            r"
use std::fmt::{self, Debug, };

impl Debug for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_split_self_for_target() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::Debug;

impl std::fmt<|> for Foo {
}
    ",
            r"
use std::fmt::{self, Debug};

impl fmt for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_add_to_nested_self_nested() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::{Debug, nested::{Display}};

impl std::fmt::nested<|> for Foo {
}
",
            r"
use std::fmt::{Debug, nested::{Display, self}};

impl nested for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_add_to_nested_self_already_included() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::{Debug, nested::{self, Display}};

impl std::fmt::nested<|> for Foo {
}
",
            r"
use std::fmt::{Debug, nested::{self, Display}};

impl nested for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_add_to_nested_nested() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::{Debug, nested::{Display}};

impl std::fmt::nested::Debug<|> for Foo {
}
",
            r"
use std::fmt::{Debug, nested::{Display, Debug}};

impl Debug for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_split_common_target_longer() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::Debug;

impl std::fmt::nested::Display<|> for Foo {
}
",
            r"
use std::fmt::{nested::Display, Debug};

impl Display for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_split_common_use_longer() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::nested::Debug;

impl std::fmt::Display<|> for Foo {
}
",
            r"
use std::fmt::{Display, nested::Debug};

impl Display for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_use_nested_import() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use crate::{
    ty::{Substs, Ty},
    AssocItem,
};

fn foo() { crate::ty::lower<|>::trait_env() }
",
            r"
use crate::{
    ty::{Substs, Ty, lower},
    AssocItem,
};

fn foo() { lower::trait_env() }
",
        );
    }

    #[test]
    fn test_replace_alias() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt as foo;

impl foo::Debug<|> for Foo {
}
",
            r"
use std::fmt as foo;

impl Debug for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_not_applicable_one_segment() {
        check_assist_not_applicable(
            replace_qualified_name_with_use,
            r"
impl foo<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_not_applicable_in_use() {
        check_assist_not_applicable(
            replace_qualified_name_with_use,
            r"
use std::fmt<|>;
",
        );
    }

    #[test]
    fn test_replace_add_use_no_anchor_in_mod_mod() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod foo {
    mod bar {
        std::fmt::Debug<|>
    }
}
    ",
            r"
mod foo {
    mod bar {
        use std::fmt::Debug;

        Debug
    }
}
    ",
        );
    }

    #[test]
    fn inserts_imports_after_inner_attributes() {
        check_assist(
            replace_qualified_name_with_use,
            r"
#![allow(dead_code)]

fn main() {
    std::fmt::Debug<|>
}
    ",
            r"
#![allow(dead_code)]
use std::fmt::Debug;

fn main() {
    Debug
}
    ",
        );
    }

    #[test]
    fn replaces_all_affected_paths() {
        check_assist(
            replace_qualified_name_with_use,
            r"
fn main() {
    std::fmt::Debug<|>;
    let x: std::fmt::Debug = std::fmt::Debug;
}
    ",
            r"
use std::fmt::Debug;

fn main() {
    Debug;
    let x: Debug = Debug;
}
    ",
        );
    }

    #[test]
    fn replaces_all_affected_paths_mod() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod m {
    fn f() {
        std::fmt::Debug<|>;
        let x: std::fmt::Debug = std::fmt::Debug;
    }
    fn g() {
        std::fmt::Debug;
    }
}

fn f() {
    std::fmt::Debug;
}
    ",
            r"
mod m {
    use std::fmt::Debug;

    fn f() {
        Debug;
        let x: Debug = Debug;
    }
    fn g() {
        Debug;
    }
}

fn f() {
    std::fmt::Debug;
}
    ",
        );
    }

    #[test]
    fn does_not_replace_in_submodules() {
        check_assist(
            replace_qualified_name_with_use,
            r"
fn main() {
    std::fmt::Debug<|>;
}

mod sub {
    fn f() {
        std::fmt::Debug;
    }
}
    ",
            r"
use std::fmt::Debug;

fn main() {
    Debug;
}

mod sub {
    fn f() {
        std::fmt::Debug;
    }
}
    ",
        );
    }

    #[test]
    fn does_not_replace_in_use() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::Display;

fn main() {
    std::fmt<|>;
}
    ",
            r"
use std::fmt::{self, Display};

fn main() {
    fmt;
}
    ",
        );
    }
}

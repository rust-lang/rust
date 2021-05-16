use ide_db::helpers::insert_use::{insert_use, ImportScope};
use syntax::{ast, match_ast, ted, AstNode, SyntaxNode};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_qualified_name_with_use
//
// Adds a use statement for a given fully-qualified name.
//
// ```
// fn process(map: std::collections::$0HashMap<String, String>) {}
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
    if path.syntax().ancestors().find_map(ast::Use::cast).is_some() {
        return None;
    }
    if path.qualifier().is_none() {
        cov_mark::hit!(dont_import_trivial_paths);
        return None;
    }

    let target = path.syntax().text_range();
    let scope = ImportScope::find_insert_use_container_with_macros(path.syntax(), &ctx.sema)?;
    let syntax = scope.as_syntax_node();
    acc.add(
        AssistId("replace_qualified_name_with_use", AssistKind::RefactorRewrite),
        "Replace qualified path with use",
        target,
        |builder| {
            // Now that we've brought the name into scope, re-qualify all paths that could be
            // affected (that is, all paths inside the node we added the `use` to).
            let syntax = builder.make_syntax_mut(syntax.clone());
            if let Some(ref import_scope) = ImportScope::from(syntax.clone()) {
                shorten_paths(&syntax, &path.clone_for_update());
                insert_use(import_scope, path, ctx.config.insert_use);
            }
        },
    )
}

/// Adds replacements to `re` that shorten `path` in all descendants of `node`.
fn shorten_paths(node: &SyntaxNode, path: &ast::Path) {
    for child in node.children() {
        match_ast! {
            match child {
                // Don't modify `use` items, as this can break the `use` item when injecting a new
                // import into the use tree.
                ast::Use(_it) => continue,
                // Don't descend into submodules, they don't have the same `use` items in scope.
                ast::Module(_it) => continue,
                ast::Path(p) => if maybe_replace_path(p.clone(), path.clone()).is_none() {
                    shorten_paths(p.syntax(), path);
                },
                _ => shorten_paths(&child, path),
            }
        }
    }
}

fn maybe_replace_path(path: ast::Path, target: ast::Path) -> Option<()> {
    if !path_eq(path.clone(), target) {
        return None;
    }

    // Shorten `path`, leaving only its last segment.
    if let Some(parent) = path.qualifier() {
        ted::remove(parent.syntax());
    }
    if let Some(double_colon) = path.coloncolon_token() {
        ted::remove(&double_colon);
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
    fn test_replace_already_imported() {
        check_assist(
            replace_qualified_name_with_use,
            r"use std::fs;

fn main() {
    std::f$0s::Path
}",
            r"use std::fs;

fn main() {
    fs::Path
}",
        )
    }

    #[test]
    fn test_replace_add_use_no_anchor() {
        check_assist(
            replace_qualified_name_with_use,
            r"
std::fmt::Debug$0
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
std::fmt::Debug$0

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

std::fmt::Debug$0
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
std::fmt$0::Debug
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

impl std::fmt::Debug$0 for Foo {
}
    ",
            r"
use std::fmt::Debug;

use stdx;

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
impl std::fmt::Debug$0 for Foo {
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
    impl std::fmt::Debug$0 for Foo {
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

impl std::io$0 for Foo {
}
    ",
            r"
use std::{fmt, io};

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

impl std::fmt::Debug$0 for Foo {
}
    ",
            r"
use std::fmt::{self, Debug};

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

impl std::fmt$0 for Foo {
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

impl std::fmt::nested$0 for Foo {
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
    fn test_replace_add_to_nested_self_already_included() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::fmt::{Debug, nested::{self, Display}};

impl std::fmt::nested$0 for Foo {
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

impl std::fmt::nested::Debug$0 for Foo {
}
",
            r"
use std::fmt::{Debug, nested::{Debug, Display}};

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

impl std::fmt::nested::Display$0 for Foo {
}
",
            r"
use std::fmt::{Debug, nested::Display};

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

impl std::fmt::Display$0 for Foo {
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

fn foo() { crate::ty::lower$0::trait_env() }
",
            r"
use crate::{AssocItem, ty::{Substs, Ty, lower}};

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

impl foo::Debug$0 for Foo {
}
",
            r"
use std::fmt as foo;

use foo::Debug;

impl Debug for Foo {
}
",
        );
    }

    #[test]
    fn dont_import_trivial_paths() {
        cov_mark::check!(dont_import_trivial_paths);
        check_assist_not_applicable(
            replace_qualified_name_with_use,
            r"
impl foo$0 for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_not_applicable_in_use() {
        check_assist_not_applicable(
            replace_qualified_name_with_use,
            r"
use std::fmt$0;
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
        std::fmt::Debug$0
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
    std::fmt::Debug$0
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
    std::fmt::Debug$0;
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
        std::fmt::Debug$0;
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
    std::fmt::Debug$0;
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
    std::fmt$0;
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

    #[test]
    fn does_not_replace_pub_use() {
        check_assist(
            replace_qualified_name_with_use,
            r"
pub use std::fmt;

impl std::io$0 for Foo {
}
    ",
            r"
pub use std::fmt;
use std::io;

impl io for Foo {
}
    ",
        );
    }

    #[test]
    fn does_not_replace_pub_crate_use() {
        check_assist(
            replace_qualified_name_with_use,
            r"
pub(crate) use std::fmt;

impl std::io$0 for Foo {
}
    ",
            r"
pub(crate) use std::fmt;
use std::io;

impl io for Foo {
}
    ",
        );
    }
}

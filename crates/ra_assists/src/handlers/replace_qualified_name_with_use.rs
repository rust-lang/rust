use hir;
use ra_syntax::{ast, AstNode, SmolStr, TextRange};

use crate::{
    assist_ctx::{Assist, AssistCtx},
    utils::insert_use_statement,
    AssistId,
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
pub(crate) fn replace_qualified_name_with_use(ctx: AssistCtx) -> Option<Assist> {
    let path: ast::Path = ctx.find_node_at_offset()?;
    // We don't want to mess with use statements
    if path.syntax().ancestors().find_map(ast::UseItem::cast).is_some() {
        return None;
    }

    let hir_path = hir::Path::from_ast(path.clone())?;
    let segments = collect_hir_path_segments(&hir_path)?;
    if segments.len() < 2 {
        return None;
    }

    let module = path.syntax().ancestors().find_map(ast::Module::cast);
    let position = match module.and_then(|it| it.item_list()) {
        Some(item_list) => item_list.syntax().clone(),
        None => {
            let current_file = path.syntax().ancestors().find_map(ast::SourceFile::cast)?;
            current_file.syntax().clone()
        }
    };

    ctx.add_assist(
        AssistId("replace_qualified_name_with_use"),
        "Replace qualified path with use",
        |edit| {
            let path_to_import = hir_path.mod_path().clone();
            insert_use_statement(
                &position,
                &path.syntax(),
                &path_to_import,
                edit.text_edit_builder(),
            );

            if let Some(last) = path.segment() {
                // Here we are assuming the assist will provide a correct use statement
                // so we can delete the path qualifier
                edit.delete(TextRange::from_to(
                    path.syntax().text_range().start(),
                    last.syntax().text_range().start(),
                ));
            }
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

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_replace_add_use_no_anchor() {
        check_assist(
            replace_qualified_name_with_use,
            "
std::fmt::Debug<|>
    ",
            "
use std::fmt::Debug;

Debug<|>
    ",
        );
    }
    #[test]
    fn test_replace_add_use_no_anchor_with_item_below() {
        check_assist(
            replace_qualified_name_with_use,
            "
std::fmt::Debug<|>

fn main() {
}
    ",
            "
use std::fmt::Debug;

Debug<|>

fn main() {
}
    ",
        );
    }

    #[test]
    fn test_replace_add_use_no_anchor_with_item_above() {
        check_assist(
            replace_qualified_name_with_use,
            "
fn main() {
}

std::fmt::Debug<|>
    ",
            "
use std::fmt::Debug;

fn main() {
}

Debug<|>
    ",
        );
    }

    #[test]
    fn test_replace_add_use_no_anchor_2seg() {
        check_assist(
            replace_qualified_name_with_use,
            "
std::fmt<|>::Debug
    ",
            "
use std::fmt;

fmt<|>::Debug
    ",
        );
    }

    #[test]
    fn test_replace_add_use() {
        check_assist(
            replace_qualified_name_with_use,
            "
use stdx;

impl std::fmt::Debug<|> for Foo {
}
    ",
            "
use stdx;
use std::fmt::Debug;

impl Debug<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_file_use_other_anchor() {
        check_assist(
            replace_qualified_name_with_use,
            "
impl std::fmt::Debug<|> for Foo {
}
    ",
            "
use std::fmt::Debug;

impl Debug<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_add_use_other_anchor_indent() {
        check_assist(
            replace_qualified_name_with_use,
            "
    impl std::fmt::Debug<|> for Foo {
    }
    ",
            "
    use std::fmt::Debug;

    impl Debug<|> for Foo {
    }
    ",
        );
    }

    #[test]
    fn test_replace_split_different() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt;

impl std::io<|> for Foo {
}
    ",
            "
use std::{io, fmt};

impl io<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_split_self_for_use() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt;

impl std::fmt::Debug<|> for Foo {
}
    ",
            "
use std::fmt::{self, Debug, };

impl Debug<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_split_self_for_target() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt::Debug;

impl std::fmt<|> for Foo {
}
    ",
            "
use std::fmt::{self, Debug};

impl fmt<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_replace_add_to_nested_self_nested() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt::{Debug, nested::{Display}};

impl std::fmt::nested<|> for Foo {
}
",
            "
use std::fmt::{Debug, nested::{Display, self}};

impl nested<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_add_to_nested_self_already_included() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt::{Debug, nested::{self, Display}};

impl std::fmt::nested<|> for Foo {
}
",
            "
use std::fmt::{Debug, nested::{self, Display}};

impl nested<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_add_to_nested_nested() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt::{Debug, nested::{Display}};

impl std::fmt::nested::Debug<|> for Foo {
}
",
            "
use std::fmt::{Debug, nested::{Display, Debug}};

impl Debug<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_split_common_target_longer() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt::Debug;

impl std::fmt::nested::Display<|> for Foo {
}
",
            "
use std::fmt::{nested::Display, Debug};

impl Display<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_split_common_use_longer() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt::nested::Debug;

impl std::fmt::Display<|> for Foo {
}
",
            "
use std::fmt::{Display, nested::Debug};

impl Display<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_use_nested_import() {
        check_assist(
            replace_qualified_name_with_use,
            "
use crate::{
    ty::{Substs, Ty},
    AssocItem,
};

fn foo() { crate::ty::lower<|>::trait_env() }
",
            "
use crate::{
    ty::{Substs, Ty, lower},
    AssocItem,
};

fn foo() { lower<|>::trait_env() }
",
        );
    }

    #[test]
    fn test_replace_alias() {
        check_assist(
            replace_qualified_name_with_use,
            "
use std::fmt as foo;

impl foo::Debug<|> for Foo {
}
",
            "
use std::fmt as foo;

impl Debug<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_not_applicable_one_segment() {
        check_assist_not_applicable(
            replace_qualified_name_with_use,
            "
impl foo<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_replace_not_applicable_in_use() {
        check_assist_not_applicable(
            replace_qualified_name_with_use,
            "
use std::fmt<|>;
",
        );
    }

    #[test]
    fn test_replace_add_use_no_anchor_in_mod_mod() {
        check_assist(
            replace_qualified_name_with_use,
            "
mod foo {
    mod bar {
        std::fmt::Debug<|>
    }
}
    ",
            "
mod foo {
    mod bar {
        use std::fmt::Debug;

        Debug<|>
    }
}
    ",
        );
    }

    #[test]
    fn inserts_imports_after_inner_attributes() {
        check_assist(
            replace_qualified_name_with_use,
            "
#![allow(dead_code)]

fn main() {
    std::fmt::Debug<|>
}
    ",
            "
#![allow(dead_code)]
use std::fmt::Debug;

fn main() {
    Debug<|>
}
    ",
        );
    }
}

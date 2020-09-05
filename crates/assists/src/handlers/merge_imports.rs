use syntax::{
    algo::{neighbor, SyntaxRewriter},
    ast, AstNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::{
        insert_use::{try_merge_imports, try_merge_trees},
        next_prev, MergeBehaviour,
    },
    AssistId, AssistKind,
};

// Assist: merge_imports
//
// Merges two imports with a common prefix.
//
// ```
// use std::<|>fmt::Formatter;
// use std::io;
// ```
// ->
// ```
// use std::{fmt::Formatter, io};
// ```
pub(crate) fn merge_imports(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let tree: ast::UseTree = ctx.find_node_at_offset()?;
    let mut rewriter = SyntaxRewriter::default();
    let mut offset = ctx.offset();

    if let Some(use_item) = tree.syntax().parent().and_then(ast::Use::cast) {
        let (merged, to_delete) =
            next_prev().filter_map(|dir| neighbor(&use_item, dir)).find_map(|use_item2| {
                try_merge_imports(&use_item, &use_item2, MergeBehaviour::Full).zip(Some(use_item2))
            })?;

        rewriter.replace_ast(&use_item, &merged);
        rewriter += to_delete.remove();

        if to_delete.syntax().text_range().end() < offset {
            offset -= to_delete.syntax().text_range().len();
        }
    } else {
        let (merged, to_delete) =
            next_prev().filter_map(|dir| neighbor(&tree, dir)).find_map(|use_tree| {
                try_merge_trees(&tree, &use_tree, MergeBehaviour::Full).zip(Some(use_tree))
            })?;

        rewriter.replace_ast(&tree, &merged);
        rewriter += to_delete.remove();

        if to_delete.syntax().text_range().end() < offset {
            offset -= to_delete.syntax().text_range().len();
        }
    };

    let target = tree.syntax().text_range();
    acc.add(
        AssistId("merge_imports", AssistKind::RefactorRewrite),
        "Merge imports",
        target,
        |builder| {
            builder.rewrite(rewriter);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_merge_first() {
        check_assist(
            merge_imports,
            r"
use std::fmt<|>::Debug;
use std::fmt::Display;
",
            r"
use std::fmt::{Debug, Display};
",
        )
    }

    #[test]
    fn test_merge_second() {
        check_assist(
            merge_imports,
            r"
use std::fmt::Debug;
use std::fmt<|>::Display;
",
            r"
use std::fmt::{Display, Debug};
",
        );
    }

    #[test]
    fn merge_self1() {
        check_assist(
            merge_imports,
            r"
use std::fmt<|>;
use std::fmt::Display;
",
            r"
use std::fmt::{self, Display};
",
        );
    }

    #[test]
    fn merge_self2() {
        check_assist(
            merge_imports,
            r"
use std::{fmt, <|>fmt::Display};
",
            r"
use std::{fmt::{Display, self}};
",
        );
    }

    #[test]
    fn skip_pub1() {
        check_assist_not_applicable(
            merge_imports,
            r"
pub use std::fmt<|>::Debug;
use std::fmt::Display;
",
        );
    }

    #[test]
    fn skip_pub_last() {
        check_assist_not_applicable(
            merge_imports,
            r"
use std::fmt<|>::Debug;
pub use std::fmt::Display;
",
        );
    }

    #[test]
    fn skip_pub_crate_pub() {
        check_assist_not_applicable(
            merge_imports,
            r"
pub(crate) use std::fmt<|>::Debug;
pub use std::fmt::Display;
",
        );
    }

    #[test]
    fn skip_pub_pub_crate() {
        check_assist_not_applicable(
            merge_imports,
            r"
pub use std::fmt<|>::Debug;
pub(crate) use std::fmt::Display;
",
        );
    }

    #[test]
    fn merge_pub() {
        check_assist(
            merge_imports,
            r"
pub use std::fmt<|>::Debug;
pub use std::fmt::Display;
",
            r"
pub use std::fmt::{Debug, Display};
",
        )
    }

    #[test]
    fn merge_pub_crate() {
        check_assist(
            merge_imports,
            r"
pub(crate) use std::fmt<|>::Debug;
pub(crate) use std::fmt::Display;
",
            r"
pub(crate) use std::fmt::{Debug, Display};
",
        )
    }

    #[test]
    fn test_merge_nested() {
        check_assist(
            merge_imports,
            r"
use std::{fmt<|>::Debug, fmt::Display};
",
            r"
use std::{fmt::{Debug, Display}};
",
        );
        check_assist(
            merge_imports,
            r"
use std::{fmt::Debug, fmt<|>::Display};
",
            r"
use std::{fmt::{Display, Debug}};
",
        );
    }

    #[test]
    fn test_merge_single_wildcard_diff_prefixes() {
        check_assist(
            merge_imports,
            r"
use std<|>::cell::*;
use std::str;
",
            r"
use std::{cell::*, str};
",
        )
    }

    #[test]
    fn test_merge_both_wildcard_diff_prefixes() {
        check_assist(
            merge_imports,
            r"
use std<|>::cell::*;
use std::str::*;
",
            r"
use std::{cell::*, str::*};
",
        )
    }

    #[test]
    fn removes_just_enough_whitespace() {
        check_assist(
            merge_imports,
            r"
use foo<|>::bar;
use foo::baz;

/// Doc comment
",
            r"
use foo::{bar, baz};

/// Doc comment
",
        );
    }

    #[test]
    fn works_with_trailing_comma() {
        check_assist(
            merge_imports,
            r"
use {
    foo<|>::bar,
    foo::baz,
};
",
            r"
use {
    foo::{bar, baz},
};
",
        );
        check_assist(
            merge_imports,
            r"
use {
    foo::baz,
    foo<|>::bar,
};
",
            r"
use {
    foo::{bar, baz},
};
",
        );
    }

    #[test]
    fn test_double_comma() {
        check_assist(
            merge_imports,
            r"
use foo::bar::baz;
use foo::<|>{
    FooBar,
};
",
            r"
use foo::{
    FooBar,
bar::baz};
",
        )
    }

    #[test]
    fn test_empty_use() {
        check_assist_not_applicable(
            merge_imports,
            r"
use std::<|>
fn main() {}",
        );
    }
}

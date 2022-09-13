use either::Either;
use ide_db::imports::merge_imports::{try_merge_imports, try_merge_trees, MergeBehavior};
use syntax::{
    algo::neighbor,
    ast::{self, edit_in_place::Removable},
    match_ast, ted, AstNode, SyntaxElement, SyntaxNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::next_prev,
    AssistId, AssistKind,
};

use Edit::*;

// Assist: merge_imports
//
// Merges two imports with a common prefix.
//
// ```
// use std::$0fmt::Formatter;
// use std::io;
// ```
// ->
// ```
// use std::{fmt::Formatter, io};
// ```
pub(crate) fn merge_imports(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (target, edits) = if ctx.has_empty_selection() {
        // Merge a neighbor
        let tree: ast::UseTree = ctx.find_node_at_offset()?;
        let target = tree.syntax().text_range();

        let edits = if let Some(use_item) = tree.syntax().parent().and_then(ast::Use::cast) {
            let mut neighbor = next_prev().find_map(|dir| neighbor(&use_item, dir)).into_iter();
            use_item.try_merge_from(&mut neighbor)
        } else {
            let mut neighbor = next_prev().find_map(|dir| neighbor(&tree, dir)).into_iter();
            tree.try_merge_from(&mut neighbor)
        };
        (target, edits?)
    } else {
        // Merge selected
        let selection_range = ctx.selection_trimmed();
        let parent_node = match ctx.covering_element() {
            SyntaxElement::Node(n) => n,
            SyntaxElement::Token(t) => t.parent()?,
        };
        let mut selected_nodes =
            parent_node.children().filter(|it| selection_range.contains_range(it.text_range()));

        let first_selected = selected_nodes.next()?;
        let edits = match_ast! {
            match first_selected {
                ast::Use(use_item) => {
                    use_item.try_merge_from(&mut selected_nodes.filter_map(ast::Use::cast))
                },
                ast::UseTree(use_tree) => {
                    use_tree.try_merge_from(&mut selected_nodes.filter_map(ast::UseTree::cast))
                },
                _ => return None,
            }
        };
        (selection_range, edits?)
    };

    acc.add(
        AssistId("merge_imports", AssistKind::RefactorRewrite),
        "Merge imports",
        target,
        |builder| {
            let edits_mut: Vec<Edit> = edits
                .into_iter()
                .map(|it| match it {
                    Remove(Either::Left(it)) => Remove(Either::Left(builder.make_mut(it))),
                    Remove(Either::Right(it)) => Remove(Either::Right(builder.make_mut(it))),
                    Replace(old, new) => Replace(builder.make_syntax_mut(old), new),
                })
                .collect();
            for edit in edits_mut {
                match edit {
                    Remove(it) => it.as_ref().either(Removable::remove, Removable::remove),
                    Replace(old, new) => ted::replace(old, new),
                }
            }
        },
    )
}

trait Merge: AstNode + Clone {
    fn try_merge_from(self, items: &mut dyn Iterator<Item = Self>) -> Option<Vec<Edit>> {
        let mut edits = Vec::new();
        let mut merged = self.clone();
        while let Some(item) = items.next() {
            merged = merged.try_merge(&item)?;
            edits.push(Edit::Remove(item.into_either()));
        }
        if !edits.is_empty() {
            edits.push(Edit::replace(self, merged));
            Some(edits)
        } else {
            None
        }
    }
    fn try_merge(&self, other: &Self) -> Option<Self>;
    fn into_either(self) -> Either<ast::Use, ast::UseTree>;
}

impl Merge for ast::Use {
    fn try_merge(&self, other: &Self) -> Option<Self> {
        try_merge_imports(self, other, MergeBehavior::Crate)
    }
    fn into_either(self) -> Either<ast::Use, ast::UseTree> {
        Either::Left(self)
    }
}

impl Merge for ast::UseTree {
    fn try_merge(&self, other: &Self) -> Option<Self> {
        try_merge_trees(self, other, MergeBehavior::Crate)
    }
    fn into_either(self) -> Either<ast::Use, ast::UseTree> {
        Either::Right(self)
    }
}

enum Edit {
    Remove(Either<ast::Use, ast::UseTree>),
    Replace(SyntaxNode, SyntaxNode),
}

impl Edit {
    fn replace(old: impl AstNode, new: impl AstNode) -> Self {
        Edit::Replace(old.syntax().clone(), new.syntax().clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_merge_equal() {
        check_assist(
            merge_imports,
            r"
use std::fmt$0::{Display, Debug};
use std::fmt::{Display, Debug};
",
            r"
use std::fmt::{Display, Debug};
",
        )
    }

    #[test]
    fn test_merge_first() {
        check_assist(
            merge_imports,
            r"
use std::fmt$0::Debug;
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
use std::fmt$0::Display;
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
use std::fmt$0;
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
use std::{fmt, $0fmt::Display};
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
pub use std::fmt$0::Debug;
use std::fmt::Display;
",
        );
    }

    #[test]
    fn skip_pub_last() {
        check_assist_not_applicable(
            merge_imports,
            r"
use std::fmt$0::Debug;
pub use std::fmt::Display;
",
        );
    }

    #[test]
    fn skip_pub_crate_pub() {
        check_assist_not_applicable(
            merge_imports,
            r"
pub(crate) use std::fmt$0::Debug;
pub use std::fmt::Display;
",
        );
    }

    #[test]
    fn skip_pub_pub_crate() {
        check_assist_not_applicable(
            merge_imports,
            r"
pub use std::fmt$0::Debug;
pub(crate) use std::fmt::Display;
",
        );
    }

    #[test]
    fn merge_pub() {
        check_assist(
            merge_imports,
            r"
pub use std::fmt$0::Debug;
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
pub(crate) use std::fmt$0::Debug;
pub(crate) use std::fmt::Display;
",
            r"
pub(crate) use std::fmt::{Debug, Display};
",
        )
    }

    #[test]
    fn merge_pub_in_path_crate() {
        check_assist(
            merge_imports,
            r"
pub(in this::path) use std::fmt$0::Debug;
pub(in this::path) use std::fmt::Display;
",
            r"
pub(in this::path) use std::fmt::{Debug, Display};
",
        )
    }

    #[test]
    fn test_merge_nested() {
        check_assist(
            merge_imports,
            r"
use std::{fmt$0::Debug, fmt::Display};
",
            r"
use std::{fmt::{Debug, Display}};
",
        );
    }

    #[test]
    fn test_merge_nested2() {
        check_assist(
            merge_imports,
            r"
use std::{fmt::Debug, fmt$0::Display};
",
            r"
use std::{fmt::{Display, Debug}};
",
        );
    }

    #[test]
    fn test_merge_with_nested_self_item() {
        check_assist(
            merge_imports,
            r"
use std$0::{fmt::{Write, Display}};
use std::{fmt::{self, Debug}};
",
            r"
use std::{fmt::{Write, Display, self, Debug}};
",
        );
    }

    #[test]
    fn test_merge_with_nested_self_item2() {
        check_assist(
            merge_imports,
            r"
use std$0::{fmt::{self, Debug}};
use std::{fmt::{Write, Display}};
",
            r"
use std::{fmt::{self, Debug, Write, Display}};
",
        );
    }

    #[test]
    fn test_merge_self_with_nested_self_item() {
        check_assist(
            merge_imports,
            r"
use std::{fmt$0::{self, Debug}, fmt::{Write, Display}};
",
            r"
use std::{fmt::{self, Debug, Write, Display}};
",
        );
    }

    #[test]
    fn test_merge_nested_self_and_empty() {
        check_assist(
            merge_imports,
            r"
use foo::$0{bar::{self}};
use foo::{bar};
",
            r"
use foo::{bar::{self}};
",
        )
    }

    #[test]
    fn test_merge_nested_empty_and_self() {
        check_assist(
            merge_imports,
            r"
use foo::$0{bar};
use foo::{bar::{self}};
",
            r"
use foo::{bar::{self}};
",
        )
    }

    #[test]
    fn test_merge_nested_list_self_and_glob() {
        check_assist(
            merge_imports,
            r"
use std$0::{fmt::*};
use std::{fmt::{self, Display}};
",
            r"
use std::{fmt::{*, self, Display}};
",
        )
    }

    #[test]
    fn test_merge_single_wildcard_diff_prefixes() {
        check_assist(
            merge_imports,
            r"
use std$0::cell::*;
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
use std$0::cell::*;
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
use foo$0::bar;
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
    foo$0::bar,
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
    foo$0::bar,
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
use foo::$0{
    FooBar,
};
",
            r"
use foo::{
    FooBar, bar::baz,
};
",
        )
    }

    #[test]
    fn test_empty_use() {
        check_assist_not_applicable(
            merge_imports,
            r"
use std::$0
fn main() {}",
        );
    }

    #[test]
    fn split_glob() {
        check_assist(
            merge_imports,
            r"
use foo::$0*;
use foo::bar::Baz;
",
            r"
use foo::{*, bar::Baz};
",
        );
    }

    #[test]
    fn merge_selection_uses() {
        check_assist(
            merge_imports,
            r"
use std::fmt::Error;
$0use std::fmt::Display;
use std::fmt::Debug;
use std::fmt::Write;
$0use std::fmt::Result;
",
            r"
use std::fmt::Error;
use std::fmt::{Display, Debug, Write};
use std::fmt::Result;
",
        );
    }

    #[test]
    fn merge_selection_use_trees() {
        check_assist(
            merge_imports,
            r"
use std::{
    fmt::Error,
    $0fmt::Display,
    fmt::Debug,
    fmt::Write,$0
    fmt::Result,
};",
            r"
use std::{
    fmt::Error,
    fmt::{Display, Debug, Write},
    fmt::Result,
};",
        );
        // FIXME: Remove redundant braces. See also unnecessary-braces diagnostic.
        check_assist(
            merge_imports,
            r"use std::$0{fmt::Display, fmt::Debug}$0;",
            r"use std::{fmt::{Display, Debug}};",
        );
    }
}

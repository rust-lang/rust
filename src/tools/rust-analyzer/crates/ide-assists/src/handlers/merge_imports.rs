use either::Either;
use ide_db::imports::{
    insert_use::{ImportGranularity, InsertUseConfig},
    merge_imports::{MergeBehavior, try_merge_imports, try_merge_trees},
};
use syntax::{
    AstNode, SyntaxElement, SyntaxNode,
    algo::neighbor,
    ast::{self, syntax_factory::SyntaxFactory},
    match_ast,
    syntax_editor::Removable,
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::next_prev,
};

use Edit::*;

// Assist: merge_imports
//
// Merges neighbor imports with a common prefix.
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
        cov_mark::hit!(merge_with_use_item_neighbors);
        let tree = ctx.find_node_at_offset::<ast::UseTree>()?.top_use_tree();
        let target = tree.syntax().text_range();

        let use_item = tree.syntax().parent().and_then(ast::Use::cast)?;
        let mut neighbor = next_prev().find_map(|dir| neighbor(&use_item, dir)).into_iter();
        let edits = use_item.try_merge_from(&mut neighbor, &ctx.config.insert_use);
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
                    cov_mark::hit!(merge_with_selected_use_item_neighbors);
                    use_item.try_merge_from(&mut selected_nodes.filter_map(ast::Use::cast), &ctx.config.insert_use)
                },
                ast::UseTree(use_tree) => {
                    cov_mark::hit!(merge_with_selected_use_tree_neighbors);
                    use_tree.try_merge_from(&mut selected_nodes.filter_map(ast::UseTree::cast), &ctx.config.insert_use)
                },
                _ => return None,
            }
        };
        (selection_range, edits?)
    };

    let parent_node = match ctx.covering_element() {
        SyntaxElement::Node(n) => n,
        SyntaxElement::Token(t) => t.parent()?,
    };

    acc.add(AssistId::refactor_rewrite("merge_imports"), "Merge imports", target, |builder| {
        let make = SyntaxFactory::with_mappings();
        let mut editor = builder.make_editor(&parent_node);

        for edit in edits {
            match edit {
                Remove(it) => {
                    let node = it.as_ref();
                    if let Some(left) = node.left() {
                        left.remove(&mut editor);
                    } else if let Some(right) = node.right() {
                        right.remove(&mut editor);
                    }
                }
                Replace(old, new) => {
                    editor.replace(old, &new);
                }
            }
        }
        editor.add_mappings(make.finish_with_mappings());
        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}

trait Merge: AstNode + Clone {
    fn try_merge_from(
        self,
        items: &mut dyn Iterator<Item = Self>,
        cfg: &InsertUseConfig,
    ) -> Option<Vec<Edit>> {
        let mut edits = Vec::new();
        let mut merged = self.clone();
        for item in items {
            merged = merged.try_merge(&item, cfg)?;
            edits.push(Edit::Remove(item.into_either()));
        }
        if !edits.is_empty() {
            edits.push(Edit::replace(self, merged));
            Some(edits)
        } else {
            None
        }
    }
    fn try_merge(&self, other: &Self, cfg: &InsertUseConfig) -> Option<Self>;
    fn into_either(self) -> Either<ast::Use, ast::UseTree>;
}

impl Merge for ast::Use {
    fn try_merge(&self, other: &Self, cfg: &InsertUseConfig) -> Option<Self> {
        let mb = match cfg.granularity {
            ImportGranularity::One => MergeBehavior::One,
            _ => MergeBehavior::Crate,
        };
        try_merge_imports(self, other, mb)
    }
    fn into_either(self) -> Either<ast::Use, ast::UseTree> {
        Either::Left(self)
    }
}

impl Merge for ast::UseTree {
    fn try_merge(&self, other: &Self, _: &InsertUseConfig) -> Option<Self> {
        try_merge_trees(self, other, MergeBehavior::Crate)
    }
    fn into_either(self) -> Either<ast::Use, ast::UseTree> {
        Either::Right(self)
    }
}

#[derive(Debug)]
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
    use crate::tests::{
        check_assist, check_assist_import_one, check_assist_not_applicable,
        check_assist_not_applicable_for_import_one,
    };

    use super::*;

    macro_rules! check_assist_import_one_variations {
        ($first: literal, $second: literal, $expected: literal) => {
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use ", $first, ";"), concat!("use ", $second, ";")),
                $expected,
            );
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use {", $first, "};"), concat!("use ", $second, ";")),
                $expected,
            );
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use ", $first, ";"), concat!("use {", $second, "};")),
                $expected,
            );
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use {", $first, "};"), concat!("use {", $second, "};")),
                $expected,
            );
        };
    }

    #[test]
    fn test_merge_equal() {
        cov_mark::check!(merge_with_use_item_neighbors);
        check_assist(
            merge_imports,
            r"
use std::fmt$0::{Display, Debug};
use std::fmt::{Display, Debug};
",
            r"
use std::fmt::{Debug, Display};
",
        );

        // The assist macro below calls `check_assist_import_one` 4 times with different input
        // use item variations based on the first 2 input parameters.
        cov_mark::check_count!(merge_with_use_item_neighbors, 4);
        check_assist_import_one_variations!(
            "std::fmt$0::{Display, Debug}",
            "std::fmt::{Display, Debug}",
            "use {std::fmt::{Debug, Display}};"
        );
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
        );
        check_assist_import_one_variations!(
            "std::fmt$0::Debug",
            "std::fmt::Display",
            "use {std::fmt::{Debug, Display}};"
        );
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
use std::fmt::{Debug, Display};
",
        );
        check_assist_import_one_variations!(
            "std::fmt::Debug",
            "std::fmt$0::Display",
            "use {std::fmt::{Debug, Display}};"
        );
    }

    #[test]
    fn merge_self() {
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
        check_assist_import_one_variations!(
            "std::fmt$0",
            "std::fmt::Display",
            "use {std::fmt::{self, Display}};"
        );
    }

    #[test]
    fn not_applicable_to_single_import() {
        check_assist_not_applicable(merge_imports, "use std::{fmt, $0fmt::Display};");
        check_assist_not_applicable_for_import_one(
            merge_imports,
            "use {std::{fmt, $0fmt::Display}};",
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
use std::{fmt$0::Debug, fmt::Error};
use std::{fmt::Write, fmt::Display};
",
            r"
use std::fmt::{Debug, Display, Error, Write};
",
        );
    }

    #[test]
    fn test_merge_nested2() {
        check_assist(
            merge_imports,
            r"
use std::{fmt::Debug, fmt$0::Error};
use std::{fmt::Write, fmt::Display};
",
            r"
use std::fmt::{Debug, Display, Error, Write};
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
use std::fmt::{self, Debug, Display, Write};
",
        );
        check_assist_import_one_variations!(
            "std$0::{fmt::{Write, Display}}",
            "std::{fmt::{self, Debug}}",
            "use {std::fmt::{self, Debug, Display, Write}};"
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
use std::fmt::{self, Debug, Display, Write};
",
        );
        check_assist_import_one_variations!(
            "std$0::{fmt::{self, Debug}}",
            "std::{fmt::{Write, Display}}",
            "use {std::fmt::{self, Debug, Display, Write}};"
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
use foo::bar;
",
        );
        check_assist_import_one_variations!(
            "foo::$0{bar::{self}}",
            "foo::{bar}",
            "use {foo::bar};"
        );
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
use foo::bar;
",
        );
        check_assist_import_one_variations!(
            "foo::$0{bar}",
            "foo::{bar::{self}}",
            "use {foo::bar};"
        );
    }

    #[test]
    fn test_merge_nested_empty_and_self_with_other() {
        check_assist(
            merge_imports,
            r"
use foo::$0{bar};
use foo::{bar::{self, other}};
",
            r"
use foo::bar::{self, other};
",
        );
        check_assist_import_one_variations!(
            "foo::$0{bar}",
            "foo::{bar::{self, other}}",
            "use {foo::bar::{self, other}};"
        );
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
use std::fmt::{self, Display, *};
",
        );
        check_assist_import_one_variations!(
            "std$0::{fmt::*}",
            "std::{fmt::{self, Display}}",
            "use {std::fmt::{self, Display, *}};"
        );
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
        );
        check_assist_import_one_variations!(
            "std$0::cell::*",
            "std::str",
            "use {std::{cell::*, str}};"
        );
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
        );
        check_assist_import_one_variations!(
            "std$0::cell::*",
            "std::str::*",
            "use {std::{cell::*, str::*}};"
        );
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
use foo$0::{
    bar, baz,
};
use foo::qux;
",
            r"
use foo::{
    bar, baz, qux,
};
",
        );
        check_assist(
            merge_imports,
            r"
use foo::{
    baz, bar,
};
use foo$0::qux;
",
            r"
use foo::{bar, baz, qux};
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
    bar::baz, FooBar
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
use foo::{bar::Baz, *};
",
        );
        check_assist_import_one_variations!(
            "foo::$0*",
            "foo::bar::Baz",
            "use {foo::{bar::Baz, *}};"
        );
    }

    #[test]
    fn merge_selection_uses() {
        cov_mark::check!(merge_with_selected_use_item_neighbors);
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
use std::fmt::{Debug, Display, Write};
use std::fmt::Result;
",
        );

        cov_mark::check!(merge_with_selected_use_item_neighbors);
        check_assist_import_one(
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
use {std::fmt::{Debug, Display, Write}};
use std::fmt::Result;
",
        );
    }

    #[test]
    fn merge_selection_use_trees() {
        cov_mark::check!(merge_with_selected_use_tree_neighbors);
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
    fmt::{Debug, Display, Write},
    fmt::Result,
};",
        );

        cov_mark::check!(merge_with_selected_use_tree_neighbors);
        check_assist(
            merge_imports,
            r"use std::{fmt::Result, $0fmt::Display, fmt::Debug$0};",
            r"use std::{fmt::Result, fmt::{Debug, Display}};",
        );

        cov_mark::check!(merge_with_selected_use_tree_neighbors);
        check_assist(
            merge_imports,
            r"use std::$0{fmt::Display, fmt::Debug}$0;",
            r"use std::{fmt::{Debug, Display}};",
        );
    }

    #[test]
    fn test_merge_with_synonymous_imports_1() {
        check_assist(
            merge_imports,
            r"
mod top {
    pub(crate) mod a {
        pub(crate) struct A;
    }
    pub(crate) mod b {
        pub(crate) struct B;
        pub(crate) struct D;
    }
}

use top::a::A;
use $0top::b::{B, B as C};
",
            r"
mod top {
    pub(crate) mod a {
        pub(crate) struct A;
    }
    pub(crate) mod b {
        pub(crate) struct B;
        pub(crate) struct D;
    }
}

use top::{a::A, b::{B, B as C}};
",
        );
    }

    #[test]
    fn test_merge_with_synonymous_imports_2() {
        check_assist(
            merge_imports,
            r"
mod top {
    pub(crate) mod a {
        pub(crate) struct A;
    }
    pub(crate) mod b {
        pub(crate) struct B;
        pub(crate) struct D;
    }
}

use top::a::A;
use $0top::b::{B as D, B as C};
",
            r"
mod top {
    pub(crate) mod a {
        pub(crate) struct A;
    }
    pub(crate) mod b {
        pub(crate) struct B;
        pub(crate) struct D;
    }
}

use top::{a::A, b::{B as D, B as C}};
",
        );
    }
}

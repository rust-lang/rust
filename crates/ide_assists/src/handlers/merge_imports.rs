use ide_db::imports::merge_imports::{try_merge_imports, try_merge_trees, MergeBehavior};
use syntax::{algo::neighbor, ast, ted, AstNode};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::next_prev,
    AssistId, AssistKind,
};

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
pub(crate) fn merge_imports(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let tree: ast::UseTree = ctx.find_node_at_offset()?;

    let mut imports = None;
    let mut uses = None;
    if let Some(use_item) = tree.syntax().parent().and_then(ast::Use::cast) {
        let (merged, to_remove) =
            next_prev().filter_map(|dir| neighbor(&use_item, dir)).find_map(|use_item2| {
                try_merge_imports(&use_item, &use_item2, MergeBehavior::Crate).zip(Some(use_item2))
            })?;

        imports = Some((use_item, merged, to_remove));
    } else {
        let (merged, to_remove) =
            next_prev().filter_map(|dir| neighbor(&tree, dir)).find_map(|use_tree| {
                try_merge_trees(&tree, &use_tree, MergeBehavior::Crate).zip(Some(use_tree))
            })?;

        uses = Some((tree.clone(), merged, to_remove))
    };

    let target = tree.syntax().text_range();
    acc.add(
        AssistId("merge_imports", AssistKind::RefactorRewrite),
        "Merge imports",
        target,
        |builder| {
            if let Some((to_replace, replacement, to_remove)) = imports {
                let to_replace = builder.make_mut(to_replace);
                let to_remove = builder.make_mut(to_remove);

                ted::replace(to_replace.syntax(), replacement.syntax());
                to_remove.remove();
            }

            if let Some((to_replace, replacement, to_remove)) = uses {
                let to_replace = builder.make_mut(to_replace);
                let to_remove = builder.make_mut(to_remove);

                ted::replace(to_replace.syntax(), replacement.syntax());
                to_remove.remove()
            }
        },
    )
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
use std::{fmt::{self, *, Display}};
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
}

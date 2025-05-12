use hir::db::ExpandDatabase;
use ide_db::syntax_helpers::prettify_macro_expansion;
use syntax::ast::{self, AstNode};

use crate::{AssistContext, AssistId, Assists};

// Assist: inline_macro
//
// Takes a macro and inlines it one step.
//
// ```
// macro_rules! num {
//     (+$($t:tt)+) => (1 + num!($($t )+));
//     (-$($t:tt)+) => (-1 + num!($($t )+));
//     (+) => (1);
//     (-) => (-1);
// }
//
// fn main() {
//     let number = num$0!(+ + + - + +);
//     println!("{number}");
// }
// ```
// ->
// ```
// macro_rules! num {
//     (+$($t:tt)+) => (1 + num!($($t )+));
//     (-$($t:tt)+) => (-1 + num!($($t )+));
//     (+) => (1);
//     (-) => (-1);
// }
//
// fn main() {
//     let number = 1+num!(+ + - + +);
//     println!("{number}");
// }
// ```
pub(crate) fn inline_macro(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let unexpanded = ctx.find_node_at_offset::<ast::MacroCall>()?;
    let macro_call = ctx.sema.to_def(&unexpanded)?;
    let target_crate_id = ctx.sema.file_to_module_def(ctx.vfs_file_id())?.krate().into();
    let text_range = unexpanded.syntax().text_range();

    acc.add(
        AssistId::refactor_inline("inline_macro"),
        "Inline macro".to_owned(),
        text_range,
        |builder| {
            let expanded = ctx.sema.parse_or_expand(macro_call.into());
            let span_map = ctx.sema.db.expansion_span_map(macro_call);
            // Don't call `prettify_macro_expansion()` outside the actual assist action; it does some heavy rowan tree manipulation,
            // which can be very costly for big macros when it is done *even without the assist being invoked*.
            let expanded = prettify_macro_expansion(ctx.db(), expanded, &span_map, target_crate_id);
            builder.replace(text_range, expanded.to_string())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    macro_rules! simple_macro {
        () => {
            r#"
macro_rules! foo {
    (foo) => (true);
    () => (false);
}
"#
        };
    }
    macro_rules! double_macro {
        () => {
            r#"
macro_rules! bar {
    (bar) => (true);
    ($($tt:tt)?) => (false);
}
macro_rules! foo {
    (foo) => (true);
    (bar) => (bar!(bar));
    ($($tt:tt)?) => (bar!($($tt)?));
}
"#
        };
    }

    macro_rules! complex_macro {
        () => {
            r#"
macro_rules! num {
    (+$($t:tt)+) => (1 + num!($($t )+));
    (-$($t:tt)+) => (-1 + num!($($t )+));
    (+) => (1);
    (-) => (-1);
}
"#
        };
    }
    #[test]
    fn inline_macro_target() {
        check_assist_target(
            inline_macro,
            concat!(simple_macro!(), r#"fn f() { let a = foo$0!(foo); }"#),
            "foo!(foo)",
        );
    }

    #[test]
    fn inline_macro_target_start() {
        check_assist_target(
            inline_macro,
            concat!(simple_macro!(), r#"fn f() { let a = $0foo!(foo); }"#),
            "foo!(foo)",
        );
    }

    #[test]
    fn inline_macro_target_end() {
        check_assist_target(
            inline_macro,
            concat!(simple_macro!(), r#"fn f() { let a = foo!(foo$0); }"#),
            "foo!(foo)",
        );
    }

    #[test]
    fn inline_macro_simple_case1() {
        check_assist(
            inline_macro,
            concat!(simple_macro!(), r#"fn f() { let result = foo$0!(foo); }"#),
            concat!(simple_macro!(), r#"fn f() { let result = true; }"#),
        );
    }

    #[test]
    fn inline_macro_simple_case2() {
        check_assist(
            inline_macro,
            concat!(simple_macro!(), r#"fn f() { let result = foo$0!(); }"#),
            concat!(simple_macro!(), r#"fn f() { let result = false; }"#),
        );
    }

    #[test]
    fn inline_macro_simple_not_applicable() {
        check_assist_not_applicable(
            inline_macro,
            concat!(simple_macro!(), r#"fn f() { let result$0 = foo!(foo); }"#),
        );
    }

    #[test]
    fn inline_macro_simple_not_applicable_broken_macro() {
        // FIXME: This is a bug. The macro should not expand, but it's
        // the same behaviour as the "Expand Macro Recursively" command
        // so it's presumably OK for the time being.
        check_assist(
            inline_macro,
            concat!(simple_macro!(), r#"fn f() { let result = foo$0!(asdfasdf); }"#),
            concat!(simple_macro!(), r#"fn f() { let result = true; }"#),
        );
    }

    #[test]
    fn inline_macro_double_case1() {
        check_assist(
            inline_macro,
            concat!(double_macro!(), r#"fn f() { let result = foo$0!(bar); }"#),
            concat!(double_macro!(), r#"fn f() { let result = bar!(bar); }"#),
        );
    }

    #[test]
    fn inline_macro_double_case2() {
        check_assist(
            inline_macro,
            concat!(double_macro!(), r#"fn f() { let result = foo$0!(asdf); }"#),
            concat!(double_macro!(), r#"fn f() { let result = bar!(asdf); }"#),
        );
    }

    #[test]
    fn inline_macro_complex_case1() {
        check_assist(
            inline_macro,
            concat!(complex_macro!(), r#"fn f() { let result = num!(+ +$0 + - +); }"#),
            concat!(complex_macro!(), r#"fn f() { let result = 1+num!(+ + - +); }"#),
        );
    }

    #[test]
    fn inline_macro_complex_case2() {
        check_assist(
            inline_macro,
            concat!(complex_macro!(), r#"fn f() { let result = n$0um!(- + + - +); }"#),
            concat!(complex_macro!(), r#"fn f() { let result = -1+num!(+ + - +); }"#),
        );
    }

    #[test]
    fn inline_macro_recursive_macro() {
        check_assist(
            inline_macro,
            r#"
macro_rules! foo {
  () => {foo!()}
}
fn f() { let result = foo$0!(); }
"#,
            r#"
macro_rules! foo {
  () => {foo!()}
}
fn f() { let result = foo!(); }
"#,
        );
    }

    #[test]
    fn inline_macro_unknown_macro() {
        check_assist_not_applicable(
            inline_macro,
            r#"
fn f() { let result = foo$0!(); }
"#,
        );
    }

    #[test]
    fn inline_macro_function_call_not_applicable() {
        check_assist_not_applicable(
            inline_macro,
            r#"
fn f() { let result = foo$0(); }
"#,
        );
    }

    #[test]
    fn inline_macro_with_whitespace() {
        check_assist(
            inline_macro,
            r#"
macro_rules! whitespace {
    () => {
        if true {}
    };
}
fn f() { whitespace$0!(); }
"#,
            r#"
macro_rules! whitespace {
    () => {
        if true {}
    };
}
fn f() { if true{}; }
"#,
        )
    }

    #[test]
    fn whitespace_between_text_and_pound() {
        check_assist(
            inline_macro,
            r#"
macro_rules! foo {
    () => {
        cfg_if! {
            if #[cfg(test)] {
                1;
            } else {
                1;
            }
        }
    }
}
fn main() {
    $0foo!();
}
"#,
            r#"
macro_rules! foo {
    () => {
        cfg_if! {
            if #[cfg(test)] {
                1;
            } else {
                1;
            }
        }
    }
}
fn main() {
    cfg_if!{
    if #[cfg(test)]{
        1;
    }else {
        1;
    }
};
}
"#,
        );
    }

    #[test]
    fn dollar_crate() {
        check_assist(
            inline_macro,
            r#"
pub struct Foo;
#[macro_export]
macro_rules! m {
    () => { $crate::Foo };
}
fn bar() {
    m$0!();
}
"#,
            r#"
pub struct Foo;
#[macro_export]
macro_rules! m {
    () => { $crate::Foo };
}
fn bar() {
    crate::Foo;
}
"#,
        );
        check_assist(
            inline_macro,
            r#"
//- /a.rs crate:a
pub struct Foo;
#[macro_export]
macro_rules! m {
    () => { $crate::Foo };
}
//- /b.rs crate:b deps:a
fn bar() {
    a::m$0!();
}
"#,
            r#"
fn bar() {
    a::Foo;
}
"#,
        );
        check_assist(
            inline_macro,
            r#"
//- /a.rs crate:a
pub struct Foo;
#[macro_export]
macro_rules! m {
    () => { $crate::Foo };
}
//- /b.rs crate:b deps:a
pub use a::m;
//- /c.rs crate:c deps:b
fn bar() {
    b::m$0!();
}
"#,
            r#"
fn bar() {
    a::Foo;
}
"#,
        );
    }
}

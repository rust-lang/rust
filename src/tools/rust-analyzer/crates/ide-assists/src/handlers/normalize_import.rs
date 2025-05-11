use ide_db::imports::merge_imports::try_normalize_import;
use syntax::{AstNode, ast};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
};

// Assist: normalize_import
//
// Normalizes an import.
//
// ```
// use$0 std::{io, {fmt::Formatter}};
// ```
// ->
// ```
// use std::{fmt::Formatter, io};
// ```
pub(crate) fn normalize_import(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let use_item = if ctx.has_empty_selection() {
        ctx.find_node_at_offset()?
    } else {
        ctx.covering_element().ancestors().find_map(ast::Use::cast)?
    };

    let target = use_item.syntax().text_range();
    let normalized_use_item =
        try_normalize_import(&use_item, ctx.config.insert_use.granularity.into())?;

    acc.add(AssistId::refactor_rewrite("normalize_import"), "Normalize import", target, |builder| {
        builder.replace_ast(use_item, normalized_use_item);
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_assist, check_assist_import_one, check_assist_not_applicable,
        check_assist_not_applicable_for_import_one,
    };

    use super::*;

    macro_rules! check_assist_variations {
        ($fixture: literal, $expected: literal) => {
            check_assist(
                normalize_import,
                concat!("use $0", $fixture, ";"),
                concat!("use ", $expected, ";"),
            );
            check_assist(
                normalize_import,
                concat!("$0use ", $fixture, ";"),
                concat!("use ", $expected, ";"),
            );

            check_assist_import_one(
                normalize_import,
                concat!("use $0", $fixture, ";"),
                concat!("use {", $expected, "};"),
            );
            check_assist_import_one(
                normalize_import,
                concat!("$0use ", $fixture, ";"),
                concat!("use {", $expected, "};"),
            );

            check_assist_import_one(
                normalize_import,
                concat!("use $0{", $fixture, "};"),
                concat!("use {", $expected, "};"),
            );
            check_assist_import_one(
                normalize_import,
                concat!("$0use {", $fixture, "};"),
                concat!("use {", $expected, "};"),
            );

            check_assist(
                normalize_import,
                concat!("use $0", $fixture, "$0;"),
                concat!("use ", $expected, ";"),
            );
            check_assist(
                normalize_import,
                concat!("$0use ", $fixture, ";$0"),
                concat!("use ", $expected, ";"),
            );
        };
    }

    macro_rules! check_assist_not_applicable_variations {
        ($fixture: literal) => {
            check_assist_not_applicable(normalize_import, concat!("use $0", $fixture, ";"));
            check_assist_not_applicable(normalize_import, concat!("$0use ", $fixture, ";"));

            check_assist_not_applicable_for_import_one(
                normalize_import,
                concat!("use $0{", $fixture, "};"),
            );
            check_assist_not_applicable_for_import_one(
                normalize_import,
                concat!("$0use {", $fixture, "};"),
            );
        };
    }

    #[test]
    fn test_order() {
        check_assist_variations!(
            "foo::{*, Qux, bar::{Quux, Bar}, baz, FOO_BAZ, self, Baz}",
            "foo::{self, bar::{Bar, Quux}, baz, Baz, Qux, FOO_BAZ, *}"
        );
    }

    #[test]
    fn test_braces_kept() {
        check_assist_not_applicable_variations!("foo::bar::{$0self}");

        // This code compiles but transforming "bar::{self}" into "bar" causes a
        // compilation error (the name `bar` is defined multiple times).
        // Therefore, the normalize_input assist must not apply here.
        check_assist_not_applicable(
            normalize_import,
            r"
mod foo {

    pub mod bar {}

    pub const bar: i32 = 8;
}

use foo::bar::{$0self};

const bar: u32 = 99;

fn main() {
    let local_bar = bar;
}

",
        );
    }

    #[test]
    fn test_redundant_braces() {
        check_assist_variations!("foo::{bar::{baz, Qux}}", "foo::bar::{baz, Qux}");
        check_assist_variations!("foo::{bar::{self}}", "foo::bar::{self}");
        check_assist_variations!("foo::{bar::{*}}", "foo::bar::*");
        check_assist_variations!("foo::{bar::{Qux as Quux}}", "foo::bar::Qux as Quux");
        check_assist_variations!(
            "foo::bar::{{FOO_BAZ, Qux, self}, {*, baz}}",
            "foo::bar::{self, baz, Qux, FOO_BAZ, *}"
        );
        check_assist_variations!(
            "foo::bar::{{{FOO_BAZ}, {{Qux}, {self}}}, {{*}, {baz}}}",
            "foo::bar::{self, baz, Qux, FOO_BAZ, *}"
        );
    }

    #[test]
    fn test_merge() {
        check_assist_variations!(
            "foo::{*, bar, {FOO_BAZ, qux}, bar::{*, baz}, {Quux}}",
            "foo::{bar::{self, baz, *}, qux, Quux, FOO_BAZ, *}"
        );
        check_assist_variations!(
            "foo::{*, bar, {FOO_BAZ, qux}, bar::{*, baz}, {Quux, bar::{baz::Foo}}}",
            "foo::{bar::{self, baz::{self, Foo}, *}, qux, Quux, FOO_BAZ, *}"
        );
    }

    #[test]
    fn test_merge_self() {
        check_assist_variations!("std::{fmt, fmt::Display}", "std::fmt::{self, Display}");
    }

    #[test]
    fn test_merge_nested() {
        check_assist_variations!("std::{fmt::Debug, fmt::Display}", "std::fmt::{Debug, Display}");
    }

    #[test]
    fn test_merge_nested2() {
        check_assist_variations!("std::{fmt::Debug, fmt::Display}", "std::fmt::{Debug, Display}");
    }

    #[test]
    fn test_merge_self_with_nested_self_item() {
        check_assist_variations!(
            "std::{fmt::{self, Debug}, fmt::{Write, Display}}",
            "std::fmt::{self, Debug, Display, Write}"
        );
    }

    #[test]
    fn works_with_trailing_comma() {
        check_assist(
            normalize_import,
            r"
use $0{
    foo::bar,
    foo::baz,
};
        ",
            r"
use foo::{bar, baz};
        ",
        );
        check_assist_import_one(
            normalize_import,
            r"
use $0{
    foo::bar,
    foo::baz,
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
    fn not_applicable_to_normalized_import() {
        check_assist_not_applicable_variations!("foo::bar");
        check_assist_not_applicable_variations!("foo::bar::*");
        check_assist_not_applicable_variations!("foo::bar::Qux as Quux");
        check_assist_not_applicable_variations!("foo::bar::{self, baz, Qux, FOO_BAZ, *}");
        check_assist_not_applicable_variations!(
            "foo::{self, bar::{Bar, Quux}, baz, Baz, Qux, FOO_BAZ, *}"
        );
        check_assist_not_applicable_variations!(
            "foo::{bar::{self, baz, *}, qux, Quux, FOO_BAZ, *}"
        );
        check_assist_not_applicable_variations!(
            "foo::{bar::{self, baz::{self, Foo}, *}, qux, Quux, FOO_BAZ, *}"
        );
    }
}

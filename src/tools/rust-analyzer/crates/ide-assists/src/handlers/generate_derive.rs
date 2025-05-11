use syntax::{
    T,
    ast::{self, AstNode, HasAttrs, edit_in_place::AttrsOwnerEdit, make},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: generate_derive
//
// Adds a new `#[derive()]` clause to a struct or enum.
//
// ```
// struct Point {
//     x: u32,
//     y: u32,$0
// }
// ```
// ->
// ```
// #[derive($0)]
// struct Point {
//     x: u32,
//     y: u32,
// }
// ```
pub(crate) fn generate_derive(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let cap = ctx.config.snippet_cap?;
    let nominal = ctx.find_node_at_offset::<ast::Adt>()?;
    let target = nominal.syntax().text_range();
    let derive_attr = nominal
        .attrs()
        .filter_map(|x| x.as_simple_call())
        .filter(|(name, _arg)| name == "derive")
        .map(|(_name, arg)| arg)
        .next();

    let delimiter = match &derive_attr {
        None => None,
        Some(tt) => Some(tt.right_delimiter_token()?),
    };

    acc.add(AssistId::generate("generate_derive"), "Add `#[derive]`", target, |edit| {
        match derive_attr {
            None => {
                let derive = make::attr_outer(make::meta_token_tree(
                    make::ext::ident_path("derive"),
                    make::token_tree(T!['('], vec![]).clone_for_update(),
                ))
                .clone_for_update();

                let nominal = edit.make_mut(nominal);
                nominal.add_attr(derive.clone());

                let delimiter = derive
                    .meta()
                    .expect("make::attr_outer was expected to have Meta")
                    .token_tree()
                    .expect("failed to get token tree out of Meta")
                    .r_paren_token()
                    .expect("make::attr_outer was expected to have a R_PAREN");

                edit.add_tabstop_before_token(cap, delimiter);
            }
            Some(_) => {
                // Just move the cursor.
                edit.add_tabstop_before_token(
                    cap,
                    delimiter.expect("Right delim token could not be found."),
                );
            }
        };
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_target};

    use super::*;

    #[test]
    fn add_derive_new() {
        check_assist(
            generate_derive,
            "struct Foo { a: i32, $0}",
            "#[derive($0)]\nstruct Foo { a: i32, }",
        );
        check_assist(
            generate_derive,
            "struct Foo { $0 a: i32, }",
            "#[derive($0)]\nstruct Foo {  a: i32, }",
        );
        check_assist(
            generate_derive,
            "
mod m {
    struct Foo { a: i32,$0 }
}
            ",
            "
mod m {
    #[derive($0)]
    struct Foo { a: i32, }
}
            ",
        );
    }

    #[test]
    fn add_derive_existing() {
        check_assist(
            generate_derive,
            "#[derive(Clone)]\nstruct Foo { a: i32$0, }",
            "#[derive(Clone$0)]\nstruct Foo { a: i32, }",
        );
    }

    #[test]
    fn add_derive_existing_with_brackets() {
        check_assist(
            generate_derive,
            "
#[derive[Clone]]
struct Foo { a: i32$0, }
",
            "
#[derive[Clone$0]]
struct Foo { a: i32, }
",
        );
    }

    #[test]
    fn add_derive_existing_missing_delimiter() {
        // since `#[derive]` isn't a simple attr call (i.e. `#[derive()]`)
        // we don't consider it as a proper derive attr and generate a new
        // one instead
        check_assist(
            generate_derive,
            "
#[derive]
struct Foo { a: i32$0, }",
            "
#[derive]
#[derive($0)]
struct Foo { a: i32, }",
        );
    }

    #[test]
    fn add_derive_new_with_doc_comment() {
        check_assist(
            generate_derive,
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32$0, }
            ",
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
#[derive($0)]
struct Foo { a: i32, }
            ",
        );
        check_assist(
            generate_derive,
            "
mod m {
    /// `Foo` is a pretty important struct.
    /// It does stuff.
    struct Foo { a: i32,$0 }
}
            ",
            "
mod m {
    /// `Foo` is a pretty important struct.
    /// It does stuff.
    #[derive($0)]
    struct Foo { a: i32, }
}
            ",
        );
    }

    #[test]
    fn add_derive_target() {
        check_assist_target(
            generate_derive,
            "
struct SomeThingIrrelevant;
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32$0, }
struct EvenMoreIrrelevant;
            ",
            "/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32, }",
        );
    }
}

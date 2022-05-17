use ide_db::assists::{AssistId, AssistKind, GroupLabel};
use syntax::{
    ast::{self, HasAttrs},
    match_ast, AstNode, SyntaxKind, TextSize,
};

use crate::assist_context::{AssistContext, Assists};

// Assist: add_attribute
//
// Adds commonly used attributes to items.
//
// ```
// struct Point {
//     x: u32,
//     y: u32,$0
// }
// ```
// ->add_derive
// ```
// #[derive($0)]
// struct Point {
//     x: u32,
//     y: u32,
// }
// ```
pub(crate) fn add_attribute(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let cap = ctx.config.snippet_cap?;

    let (attr_owner, attrs) = ctx
        .find_node_at_offset::<ast::AnyHasAttrs>()?
        .syntax()
        .ancestors()
        .filter_map(ast::AnyHasAttrs::cast)
        .find_map(|attr_owner| {
            let node = attr_owner.syntax();
            match_ast! {
                match node {
                    ast::Adt(_) => Some((attr_owner, ADT_ATTRS)),
                    ast::Fn(_) => Some((attr_owner, FN_ATTRS)),
                    _ => None,
                }
            }
        })?;

    let offset = attr_insertion_offset(&attr_owner)?;

    for tpl in attrs {
        let existing_offset = attr_owner.attrs().find_map(|attr| {
            if attr.simple_name()? == tpl.name {
                match attr.token_tree() {
                    Some(tt) => {
                        // Attribute like `#[derive(...)]`, position cursor right before the `)`
                        return Some(tt.syntax().text_range().end() - TextSize::of(')'));
                    }
                    None => {
                        // `#[key = value]`
                        let tok = attr.syntax().last_token()?;
                        if tok.kind() == SyntaxKind::R_BRACK {
                            return Some(tok.text_range().end() - TextSize::of(']'));
                        }
                    }
                }
            }
            None
        });
        acc.add_group(
            &GroupLabel("Add attribute".into()),
            AssistId(tpl.id, AssistKind::Generate),
            format!("Add `#[{}]`", tpl.name),
            attr_owner.syntax().text_range(),
            |b| match existing_offset {
                Some(offset) => {
                    b.insert_snippet(cap, offset, "$0");
                }
                None => {
                    b.insert_snippet(cap, offset, format!("#[{}]\n", tpl.snippet));
                }
            },
        );
    }

    Some(())
}

fn attr_insertion_offset(nominal: &ast::AnyHasAttrs) -> Option<TextSize> {
    let non_ws_child = nominal
        .syntax()
        .children_with_tokens()
        .find(|it| it.kind() != SyntaxKind::COMMENT && it.kind() != SyntaxKind::WHITESPACE)?;
    Some(non_ws_child.text_range().start())
}

static ADT_ATTRS: &[AttrTemplate] = &[
    AttrTemplate { id: "add_derive", name: "derive", snippet: "derive($0)" },
    AttrTemplate { id: "add_must_use", name: "must_use", snippet: "must_use$0" },
];

static FN_ATTRS: &[AttrTemplate] = &[
    AttrTemplate { id: "add_inline", name: "inline", snippet: "inline$0" },
    AttrTemplate { id: "add_must_use", name: "must_use", snippet: "must_use$0" },
];

struct AttrTemplate {
    /// Assist ID.
    id: &'static str,
    /// User-facing attribute name.
    name: &'static str,
    /// Snippet to insert.
    snippet: &'static str,
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist_by_label, check_assist_target};

    use super::add_attribute;

    fn check_derive(ra_fixture_before: &str, ra_fixture_after: &str) {
        check_assist_by_label(
            add_attribute,
            ra_fixture_before,
            ra_fixture_after,
            "Add `#[derive]`",
        );
    }

    fn check_must_use(ra_fixture_before: &str, ra_fixture_after: &str) {
        check_assist_by_label(
            add_attribute,
            ra_fixture_before,
            ra_fixture_after,
            "Add `#[must_use]`",
        );
    }

    #[test]
    fn add_derive_new() {
        check_derive("struct Foo { a: i32, $0}", "#[derive($0)]\nstruct Foo { a: i32, }");
        check_derive("struct Foo { $0 a: i32, }", "#[derive($0)]\nstruct Foo {  a: i32, }");
    }

    #[test]
    fn add_derive_existing() {
        check_derive(
            "#[derive(Clone)]\nstruct Foo { a: i32$0, }",
            "#[derive(Clone$0)]\nstruct Foo { a: i32, }",
        );
    }

    #[test]
    fn add_derive_new_with_doc_comment() {
        check_derive(
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
    }

    #[test]
    fn add_derive_target() {
        check_assist_target(
            add_attribute,
            r#"
struct SomeThingIrrelevant;
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32$0, }
struct EvenMoreIrrelevant;
            "#,
            "/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32, }",
        );
    }

    #[test]
    fn insert_must_use() {
        check_must_use("struct S$0;", "#[must_use$0]\nstruct S;");
        check_must_use("$0fn f() {}", "#[must_use$0]\nfn f() {}");

        check_must_use(r#"#[must_use = "bla"] struct S$0;"#, r#"#[must_use = "bla"$0] struct S;"#);
        check_must_use(r#"#[must_use = ] struct S$0;"#, r#"#[must_use = $0] struct S;"#);

        check_must_use(r#"#[must_use = "bla"] $0fn f() {}"#, r#"#[must_use = "bla"$0] fn f() {}"#);
        check_must_use(r#"#[must_use = ] $0fn f() {}"#, r#"#[must_use = $0] fn f() {}"#);
    }
}

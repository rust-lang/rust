//! Macro input conditioning.

use syntax::{
    ast::{self, make, AttrsOwner},
    AstNode, SyntaxNode,
};

use crate::{
    name::{name, AsName},
    MacroCallKind,
};

pub(crate) fn process_macro_input(macro_call_kind: &MacroCallKind, node: SyntaxNode) -> SyntaxNode {
    match macro_call_kind {
        MacroCallKind::FnLike { .. } => node,
        MacroCallKind::Derive { derive_attr_index, .. } => {
            let item = match ast::Item::cast(node.clone()) {
                Some(item) => item,
                None => return node,
            };

            remove_derives_up_to(item, *derive_attr_index as usize).syntax().clone()
        }
        MacroCallKind::Attr { invoc_attr_index, .. } => {
            let item = match ast::Item::cast(node.clone()) {
                Some(item) => item,
                None => return node,
            };

            remove_attr_invoc(item, *invoc_attr_index as usize).syntax().clone()
        }
    }
}

/// Removes `#[derive]` attributes from `item`, up to `attr_index`.
fn remove_derives_up_to(item: ast::Item, attr_index: usize) -> ast::Item {
    let item = item.clone_for_update();
    for attr in item.attrs().take(attr_index + 1) {
        if let Some(name) =
            attr.path().and_then(|path| path.as_single_segment()).and_then(|seg| seg.name_ref())
        {
            if name.as_name() == name![derive] {
                replace_attr(&item, &attr);
            }
        }
    }
    item
}

/// Removes the attribute invoking an attribute macro from `item`.
fn remove_attr_invoc(item: ast::Item, attr_index: usize) -> ast::Item {
    let item = item.clone_for_update();
    let attr = item
        .attrs()
        .nth(attr_index)
        .unwrap_or_else(|| panic!("cannot find attribute #{}", attr_index));
    replace_attr(&item, &attr);
    item
}

fn replace_attr(item: &ast::Item, attr: &ast::Attr) {
    let syntax_index = attr.syntax().index();
    let ws = make::tokens::whitespace(&" ".repeat(u32::from(attr.syntax().text().len()) as usize));
    item.syntax().splice_children(syntax_index..syntax_index + 1, vec![ws.into()]);
}

#[cfg(test)]
mod tests {
    use base_db::{fixture::WithFixture, SourceDatabase};
    use expect_test::{expect, Expect};

    use crate::test_db::TestDB;

    use super::*;

    fn test_remove_derives_up_to(attr: usize, ra_fixture: &str, expect: Expect) {
        let (db, file_id) = TestDB::with_single_file(ra_fixture);
        let parsed = db.parse(file_id);

        let mut items: Vec<_> =
            parsed.syntax_node().descendants().filter_map(ast::Item::cast).collect();
        assert_eq!(items.len(), 1);

        let item = remove_derives_up_to(items.pop().unwrap(), attr);
        let res: String =
            item.syntax().children_with_tokens().map(|e| format!("{:?}\n", e)).collect();
        expect.assert_eq(&res);
    }

    #[test]
    fn remove_derive() {
        test_remove_derives_up_to(
            2,
            r#"
#[allow(unused)]
#[derive(Copy)]
#[derive(Hello)]
#[derive(Clone)]
struct A {
    bar: u32
}
        "#,
            expect![[r#"
                Node(ATTR@0..16)
                Token(WHITESPACE@16..17 "\n")
                Token(WHITESPACE@17..32 "               ")
                Token(WHITESPACE@32..33 "\n")
                Token(WHITESPACE@33..49 "                ")
                Token(WHITESPACE@49..50 "\n")
                Node(ATTR@50..66)
                Token(WHITESPACE@66..67 "\n")
                Token(STRUCT_KW@67..73 "struct")
                Token(WHITESPACE@73..74 " ")
                Node(NAME@74..75)
                Token(WHITESPACE@75..76 " ")
                Node(RECORD_FIELD_LIST@76..92)
            "#]],
        );
    }
}

//! Macro input conditioning.

use syntax::{
    ast::{self, AttrsOwner},
    AstNode, SyntaxNode,
};

use crate::{
    db::AstDatabase,
    name::{name, AsName},
    AttrId, LazyMacroId, MacroCallKind, MacroCallLoc,
};

pub(crate) fn process_macro_input(
    db: &dyn AstDatabase,
    node: SyntaxNode,
    id: LazyMacroId,
) -> SyntaxNode {
    let loc: MacroCallLoc = db.lookup_intern_macro(id);

    match loc.kind {
        MacroCallKind::FnLike { .. } => node,
        MacroCallKind::Derive { derive_attr, .. } => {
            let item = match ast::Item::cast(node.clone()) {
                Some(item) => item,
                None => return node,
            };

            remove_derives_up_to(item, derive_attr).syntax().clone()
        }
    }
}

/// Removes `#[derive]` attributes from `item`, up to `attr`.
fn remove_derives_up_to(item: ast::Item, attr: AttrId) -> ast::Item {
    let item = item.clone_for_update();
    let idx = attr.0 as usize;
    for attr in item.attrs().take(idx + 1) {
        if let Some(name) =
            attr.path().and_then(|path| path.as_single_segment()).and_then(|seg| seg.name_ref())
        {
            if name.as_name() == name![derive] {
                attr.syntax().detach();
            }
        }
    }
    item
}

#[cfg(test)]
mod tests {
    use base_db::fixture::WithFixture;
    use base_db::SourceDatabase;
    use expect_test::{expect, Expect};

    use crate::test_db::TestDB;

    use super::*;

    fn test_remove_derives_up_to(attr: AttrId, ra_fixture: &str, expect: Expect) {
        let (db, file_id) = TestDB::with_single_file(&ra_fixture);
        let parsed = db.parse(file_id);

        let mut items: Vec<_> =
            parsed.syntax_node().descendants().filter_map(ast::Item::cast).collect();
        assert_eq!(items.len(), 1);

        let item = remove_derives_up_to(items.pop().unwrap(), attr);
        expect.assert_eq(&item.to_string());
    }

    #[test]
    fn remove_derive() {
        test_remove_derives_up_to(
            AttrId(2),
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
#[allow(unused)]


#[derive(Clone)]
struct A {
    bar: u32
}"#]],
        );
    }
}

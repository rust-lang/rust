use join_to_string::join;
use ra_text_edit::TextEditBuilder;
use ra_syntax::{
    ast::{self, AstNode, NameOwner, TypeParamsOwner},
    SourceFileNode,
    TextUnit,
};

use crate::{find_node_at_offset, assists::LocalEdit};

pub fn add_impl<'a>(
    file: &'a SourceFileNode,
    offset: TextUnit,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let nominal = find_node_at_offset::<ast::NominalDef>(file.syntax(), offset)?;
    let name = nominal.name()?;

    Some(move || {
        let type_params = nominal.type_param_list();
        let mut edit = TextEditBuilder::new();
        let start_offset = nominal.syntax().range().end();
        let mut buf = String::new();
        buf.push_str("\n\nimpl");
        if let Some(type_params) = type_params {
            type_params.syntax().text().push_to(&mut buf);
        }
        buf.push_str(" ");
        buf.push_str(name.text().as_str());
        if let Some(type_params) = type_params {
            let lifetime_params = type_params
                .lifetime_params()
                .filter_map(|it| it.lifetime())
                .map(|it| it.text());
            let type_params = type_params
                .type_params()
                .filter_map(|it| it.name())
                .map(|it| it.text());
            join(lifetime_params.chain(type_params))
                .surround_with("<", ">")
                .to_buf(&mut buf);
        }
        buf.push_str(" {\n");
        let offset = start_offset + TextUnit::of_str(&buf);
        buf.push_str("\n}");
        edit.insert(start_offset, buf);
        LocalEdit {
            label: "add impl".to_string(),
            edit: edit.finish(),
            cursor_position: Some(offset),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::check_action;

    #[test]
    fn test_add_impl() {
        check_action(
            "struct Foo {<|>}\n",
            "struct Foo {}\n\nimpl Foo {\n<|>\n}\n",
            |file, off| add_impl(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo<T: Clone> {<|>}",
            "struct Foo<T: Clone> {}\n\nimpl<T: Clone> Foo<T> {\n<|>\n}",
            |file, off| add_impl(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo<'a, T: Foo<'a>> {<|>}",
            "struct Foo<'a, T: Foo<'a>> {}\n\nimpl<'a, T: Foo<'a>> Foo<'a, T> {\n<|>\n}",
            |file, off| add_impl(file, off).map(|f| f()),
        );
    }

}

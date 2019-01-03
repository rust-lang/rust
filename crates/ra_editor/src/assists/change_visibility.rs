use ra_text_edit::TextEditBuilder;
use ra_syntax::{
    SourceFileNode,
    algo::find_leaf_at_offset,
    SyntaxKind::{VISIBILITY, FN_KW, MOD_KW, STRUCT_KW, ENUM_KW, TRAIT_KW, FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF},
    TextUnit,
};

use crate::assists::LocalEdit;

pub fn change_visibility<'a>(
    file: &'a SourceFileNode,
    offset: TextUnit,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let syntax = file.syntax();

    let keyword = find_leaf_at_offset(syntax, offset).find(|leaf| match leaf.kind() {
        FN_KW | MOD_KW | STRUCT_KW | ENUM_KW | TRAIT_KW => true,
        _ => false,
    })?;
    let parent = keyword.parent()?;
    let def_kws = vec![FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF];
    let node_start = parent.range().start();
    Some(move || {
        let mut edit = TextEditBuilder::new();

        if !def_kws.iter().any(|&def_kw| def_kw == parent.kind())
            || parent.children().any(|child| child.kind() == VISIBILITY)
        {
            return LocalEdit {
                label: "make pub crate".to_string(),
                edit: edit.finish(),
                cursor_position: Some(offset),
            };
        }

        edit.insert(node_start, "pub(crate) ".to_string());
        LocalEdit {
            label: "make pub crate".to_string(),
            edit: edit.finish(),
            cursor_position: Some(node_start),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::check_action;

    #[test]
    fn test_change_visibility() {
        check_action(
            "<|>fn foo() {}",
            "<|>pub(crate) fn foo() {}",
            |file, off| change_visibility(file, off).map(|f| f()),
        );
        check_action(
            "f<|>n foo() {}",
            "<|>pub(crate) fn foo() {}",
            |file, off| change_visibility(file, off).map(|f| f()),
        );
        check_action(
            "<|>struct Foo {}",
            "<|>pub(crate) struct Foo {}",
            |file, off| change_visibility(file, off).map(|f| f()),
        );
        check_action("<|>mod foo {}", "<|>pub(crate) mod foo {}", |file, off| {
            change_visibility(file, off).map(|f| f())
        });
        check_action(
            "<|>trait Foo {}",
            "<|>pub(crate) trait Foo {}",
            |file, off| change_visibility(file, off).map(|f| f()),
        );
        check_action("m<|>od {}", "<|>pub(crate) mod {}", |file, off| {
            change_visibility(file, off).map(|f| f())
        });
        check_action(
            "pub(crate) f<|>n foo() {}",
            "pub(crate) f<|>n foo() {}",
            |file, off| change_visibility(file, off).map(|f| f()),
        );
        check_action(
            "unsafe f<|>n foo() {}",
            "<|>pub(crate) unsafe fn foo() {}",
            |file, off| change_visibility(file, off).map(|f| f()),
        );
    }
}

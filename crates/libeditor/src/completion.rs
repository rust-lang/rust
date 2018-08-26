use libsyntax2::{
    File, TextUnit,
    ast,
    algo::find_leaf_at_offset,
};

use {
    AtomEdit, find_node_at_offset,
};

#[derive(Debug)]
pub struct CompletionItem {
    name: String,
}

pub fn scope_completion(file: &File, offset: TextUnit) -> Option<Vec<CompletionItem>> {
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        // Don't bother with completion if incremental reparse fails
        file.incremental_reparse(&edit)?
    };
    let name_ref = find_node_at_offset::<ast::NameRef>(file.syntax(), offset)?;
    Some(complete(name_ref))
}

fn complete(name_ref: ast::NameRef) -> Vec<CompletionItem> {
    vec![CompletionItem {
        name: "foo".to_string()
    }]
}

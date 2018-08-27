use libsyntax2::{
    File, TextUnit, AstNode,
    ast::self,
    algo::{
        ancestors,
    },
};

use {
    AtomEdit, find_node_at_offset,
    scope::{FnScopes, compute_scopes},
};

#[derive(Debug)]
pub struct CompletionItem {
    pub name: String,
}

pub fn scope_completion(file: &File, offset: TextUnit) -> Option<Vec<CompletionItem>> {
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        // Don't bother with completion if incremental reparse fails
        file.incremental_reparse(&edit)?
    };
    let name_ref = find_node_at_offset::<ast::NameRef>(file.syntax(), offset)?;
    let fn_def = ancestors(name_ref.syntax()).filter_map(ast::FnDef::cast).next()?;
    let scopes = compute_scopes(fn_def);
    Some(complete(name_ref, &scopes))
}

fn complete(name_ref: ast::NameRef, scopes: &FnScopes) -> Vec<CompletionItem> {
    scopes.scope_chain(name_ref.syntax())
        .flat_map(|scope| scopes.entries(scope).iter())
        .map(|entry| CompletionItem {
            name: entry.name().to_string()
        })
        .collect()
}

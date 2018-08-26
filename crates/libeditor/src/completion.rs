use libsyntax2::{
    File, TextUnit, AstNode, SyntaxNodeRef,
    ast::{self, NameOwner},
    algo::{
        ancestors,
        visit::{visitor_ctx, VisitorCtx},
        walk::preorder,
    },
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
    let mut res = Vec::new();
    for node in ancestors(name_ref.syntax()) {
        process_scope(node, &mut res);
    }
    res
}

fn process_scope(node: SyntaxNodeRef, sink: &mut Vec<CompletionItem>) {
    let _ = visitor_ctx(sink)
        .visit::<ast::Block, _>(|block, sink| {
            block.let_stmts()
                .filter_map(|it| it.pat())
                .for_each(move |it| process_pat(it, sink))
        })
        .visit::<ast::FnDef, _>(|fn_def, sink| {
            fn_def.param_list().into_iter()
                .flat_map(|it| it.params())
                .filter_map(|it| it.pat())
                .for_each(move |it| process_pat(it, sink))
        })
        .accept(node);

    fn process_pat(pat: ast::Pat, sink: &mut Vec<CompletionItem>) {
        let items = preorder(pat.syntax())
            .filter_map(ast::BindPat::cast)
            .filter_map(ast::BindPat::name)
            .map(|name| CompletionItem { name: name.text().to_string() });
        sink.extend(items);
    }
}

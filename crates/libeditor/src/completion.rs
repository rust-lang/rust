use std::collections::HashMap;

use libsyntax2::{
    File, TextUnit, AstNode, SyntaxNodeRef, SyntaxNode, SmolStr,
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

// fn compute_scopes(fn_def: ast::FnDef) -> FnScopes {
//     let mut scopes = FnScopes::new();
// }

// type ScopeId = usize;

// struct FnScopes {
//     scopes: Vec<ScopeData>,
//     scope_for_expr: HashMap<SyntaxNode, ScopeId>,
// }

// impl FnScopes {
//     fn new() -> FnScopes {
//         FnScopes {
//             scopes: vec![],
//             scope_for_expr: HashMap::new(),
//         }
//     }

//     fn new_scope(&mut Self) -> ScopeId {
//         let res = self.scopes.len();
//         self.scopes.push(ScopeData { parent: None, entries: vec![] })
//     }

//     fn set_parent
// }

// struct ScopeData {
//     parent: Option<ScopeId>,
//     entries: Vec<ScopeEntry>
// }

// struct ScopeEntry {
//     syntax: SyntaxNode
// }

// impl ScopeEntry {
//     fn name(&self) -> SmolStr {
//         self.ast().name()
//             .unwrap()
//             .text()
//     }

//     fn ast(&self) -> ast::BindPat {
//         ast::BindPat::cast(self.syntax.borrowed())
//             .unwrap()
//     }
// }

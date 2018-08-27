use std::collections::HashMap;

use libsyntax2::{
    File, TextUnit, AstNode, SyntaxNodeRef, SyntaxNode, SmolStr,
    ast::{self, NameOwner},
    algo::{
        ancestors,
        walk::preorder,
        generate,
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

fn compute_scopes(fn_def: ast::FnDef) -> FnScopes {
    let mut scopes = FnScopes::new();
    let root = scopes.root_scope();
    fn_def.param_list().into_iter()
        .flat_map(|it| it.params())
        .filter_map(|it| it.pat())
        .for_each(|it| scopes.add_bindings(root, it));

    let mut scope = root;
    if let Some(body) = fn_def.body() {
        for stmt in body.statements() {
            match stmt {
                ast::Stmt::LetStmt(stmt) => {
                    scope = scopes.new_scope(scope);
                    if let Some(pat) = stmt.pat() {
                        scopes.add_bindings(scope, pat);
                    }
                    if let Some(expr) = stmt.initializer() {
                        scopes.set_scope(expr.syntax(), scope)
                    }
                }
                ast::Stmt::ExprStmt(expr) => {
                    scopes.set_scope(expr.syntax(), scope)
                }
            }
        }
        if let Some(expr) = body.expr() {
            scopes.set_scope(expr.syntax(), scope)
        }
    }
    scopes
}

type ScopeId = usize;

struct FnScopes {
    scopes: Vec<ScopeData>,
    scope_for: HashMap<SyntaxNode, ScopeId>,
}

impl FnScopes {
    fn new() -> FnScopes {
        FnScopes {
            scopes: vec![],
            scope_for: HashMap::new(),
        }
    }
    fn root_scope(&mut self) -> ScopeId {
        let res = self.scopes.len();
        self.scopes.push(ScopeData { parent: None, entries: vec![] });
        res
    }
    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        let res = self.scopes.len();
        self.scopes.push(ScopeData { parent: Some(parent), entries: vec![] });
        res
    }
    fn add_bindings(&mut self, scope: ScopeId, pat: ast::Pat) {
        let entries = preorder(pat.syntax())
            .filter_map(ast::BindPat::cast)
            .filter_map(ScopeEntry::new);
        self.scopes[scope].entries.extend(entries);
    }
    fn set_scope(&mut self, node: SyntaxNodeRef, scope: ScopeId) {
        self.scope_for.insert(node.owned(), scope);
    }
    fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scopes[scope].entries
    }
    fn scope_for(&self, node: SyntaxNodeRef) -> Option<ScopeId> {
        ancestors(node)
            .filter_map(|it| self.scope_for.get(&it.owned()).map(|&scope| scope))
            .next()
    }
    fn scope_chain<'a>(&'a self, node: SyntaxNodeRef) -> impl Iterator<Item=ScopeId> + 'a {
        generate(self.scope_for(node), move |&scope| self.scopes[scope].parent)
    }
}

struct ScopeData {
    parent: Option<ScopeId>,
    entries: Vec<ScopeEntry>
}

struct ScopeEntry {
    syntax: SyntaxNode
}

impl ScopeEntry {
    fn new(pat: ast::BindPat) -> Option<ScopeEntry> {
        if pat.name().is_some() {
            Some(ScopeEntry { syntax: pat.syntax().owned() })
        } else {
            None
        }
    }

    fn name(&self) -> SmolStr {
        self.ast().name()
            .unwrap()
            .text()
    }

    fn ast(&self) -> ast::BindPat {
        ast::BindPat::cast(self.syntax.borrowed())
            .unwrap()
    }
}

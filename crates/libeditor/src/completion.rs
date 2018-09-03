use std::collections::{HashSet, HashMap};

use libsyntax2::{
    File, TextUnit, AstNode, SyntaxNodeRef, SyntaxKind::*,
    ast::{self, LoopBodyOwner},
    algo::{
        ancestors,
        visit::{visitor, Visitor, visitor_ctx, VisitorCtx},
    },
    text_utils::is_subrange,
};

use {
    AtomEdit, find_node_at_offset,
    scope::{FnScopes, ModuleScope},
};

#[derive(Debug)]
pub struct CompletionItem {
    /// What user sees in pop-up
    pub label: String,
    /// What string is used for filtering, defaults to label
    pub lookup: Option<String>,
    /// What is inserted, defaults to label
    pub snippet: Option<String>
}

pub fn scope_completion(file: &File, offset: TextUnit) -> Option<Vec<CompletionItem>> {
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        file.reparse(&edit)
    };
    let mut has_completions = false;
    let mut res = Vec::new();
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(file.syntax(), offset) {
        has_completions = true;
        complete_name_ref(&file, name_ref, &mut res);
        // special case, `trait T { fn foo(i_am_a_name_ref) {} }`
        if is_node::<ast::Param>(name_ref.syntax()) {
            param_completions(name_ref.syntax(), &mut res);
        }
    }
    if let Some(name) = find_node_at_offset::<ast::Name>(file.syntax(), offset) {
        if is_node::<ast::Param>(name.syntax()) {
            has_completions = true;
            param_completions(name.syntax(), &mut res);
        }
    }
    if has_completions {
        Some(res)
    } else {
        None
    }
}

fn complete_name_ref(file: &File, name_ref: ast::NameRef, acc: &mut Vec<CompletionItem>) {
    if !is_node::<ast::Path>(name_ref.syntax()) {
        return;
    }
    if let Some(fn_def) = ancestors(name_ref.syntax()).filter_map(ast::FnDef::cast).next() {
        complete_expr_keywords(&file, fn_def, name_ref, acc);
        let scopes = FnScopes::new(fn_def);
        complete_fn(name_ref, &scopes, acc);
    }
    if let Some(root) = ancestors(name_ref.syntax()).filter_map(ast::Root::cast).next() {
        let scope = ModuleScope::new(root);
        acc.extend(
            scope.entries().iter()
                .filter(|entry| entry.syntax() != name_ref.syntax())
                .map(|entry| CompletionItem {
                    label: entry.name().to_string(),
                    lookup: None,
                    snippet: None,
                })
        );
    }
}

fn param_completions(ctx: SyntaxNodeRef, acc: &mut Vec<CompletionItem>) {
    let mut params = HashMap::new();
    for node in ancestors(ctx) {
        let _ = visitor_ctx(&mut params)
            .visit::<ast::Root, _>(process)
            .visit::<ast::ItemList, _>(process)
            .accept(node);
    }
    params.into_iter()
        .filter_map(|(label, (count, param))| {
            let lookup = param.pat()?.syntax().text().to_string();
            if count < 2 { None } else { Some((label, lookup)) }
        })
        .for_each(|(label, lookup)| {
            acc.push(CompletionItem {
                label, lookup: Some(lookup), snippet: None
            })
        });

    fn process<'a, N: ast::FnDefOwner<'a>>(node: N, params: &mut HashMap<String, (u32, ast::Param<'a>)>) {
        node.functions()
            .filter_map(|it| it.param_list())
            .flat_map(|it| it.params())
            .for_each(|param| {
                let text = param.syntax().text().to_string();
                params.entry(text)
                      .or_insert((0, param))
                      .0 += 1;
            })
    }
}

fn is_node<'a, N: AstNode<'a>>(node: SyntaxNodeRef<'a>) -> bool {
    match ancestors(node).filter_map(N::cast).next() {
        None => false,
        Some(n) => n.syntax().range() == node.range(),
    }
}


fn complete_expr_keywords(file: &File, fn_def: ast::FnDef, name_ref: ast::NameRef, acc: &mut Vec<CompletionItem>) {
    acc.push(keyword("if", "if $0 {}"));
    acc.push(keyword("match", "match $0 {}"));
    acc.push(keyword("while", "while $0 {}"));
    acc.push(keyword("loop", "loop {$0}"));

    if let Some(off) = name_ref.syntax().range().start().checked_sub(2.into()) {
        if let Some(if_expr) = find_node_at_offset::<ast::IfExpr>(file.syntax(), off) {
            if if_expr.syntax().range().end() < name_ref.syntax().range().start() {
                acc.push(keyword("else", "else {$0}"));
                acc.push(keyword("else if", "else if $0 {}"));
            }
        }
    }
    if is_in_loop_body(name_ref) {
        acc.push(keyword("continue", "continue"));
        acc.push(keyword("break", "break"));
    }
    acc.extend(complete_return(fn_def, name_ref));
}

fn is_in_loop_body(name_ref: ast::NameRef) -> bool {
    for node in ancestors(name_ref.syntax()) {
        if node.kind() == FN_DEF || node.kind() == LAMBDA_EXPR {
            break;
        }
        let loop_body = visitor()
            .visit::<ast::ForExpr, _>(LoopBodyOwner::loop_body)
            .visit::<ast::WhileExpr, _>(LoopBodyOwner::loop_body)
            .visit::<ast::LoopExpr, _>(LoopBodyOwner::loop_body)
            .accept(node);
        if let Some(Some(body)) = loop_body {
            if is_subrange(body.syntax().range(), name_ref.syntax().range()) {
                return true;
            }
        }
    }
    false
}

fn complete_return(fn_def: ast::FnDef, name_ref: ast::NameRef) -> Option<CompletionItem> {
    // let is_last_in_block = ancestors(name_ref.syntax()).filter_map(ast::Expr::cast)
    //     .next()
    //     .and_then(|it| it.syntax().parent())
    //     .and_then(ast::Block::cast)
    //     .is_some();

    // if is_last_in_block {
    //     return None;
    // }

    let is_stmt = match ancestors(name_ref.syntax()).filter_map(ast::ExprStmt::cast).next() {
        None => false,
        Some(expr_stmt) => expr_stmt.syntax().range() == name_ref.syntax().range()
    };
    let snip = match (is_stmt, fn_def.ret_type().is_some()) {
        (true, true) => "return $0;",
        (true, false) => "return;",
        (false, true) => "return $0",
        (false, false) => "return",
    };
    Some(keyword("return", snip))
}

fn keyword(kw: &str, snip: &str) -> CompletionItem {
    CompletionItem {
        label: kw.to_string(),
        lookup: None,
        snippet: Some(snip.to_string()),
    }
}

fn complete_fn(name_ref: ast::NameRef, scopes: &FnScopes, acc: &mut Vec<CompletionItem>) {
    let mut shadowed = HashSet::new();
    acc.extend(
        scopes.scope_chain(name_ref.syntax())
            .flat_map(|scope| scopes.entries(scope).iter())
            .filter(|entry| shadowed.insert(entry.name()))
            .map(|entry| CompletionItem {
                label: entry.name().to_string(),
                lookup: None,
                snippet: None,
            })
    );
    if scopes.self_param.is_some() {
        acc.push(CompletionItem {
            label: "self".to_string(),
            lookup: None,
            snippet: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::{assert_eq_dbg, extract_offset};

    fn check_scope_completion(code: &str, expected_completions: &str) {
        let (off, code) = extract_offset(&code);
        let file = File::parse(&code);
        let completions = scope_completion(&file, off)
            .unwrap()
            .into_iter()
            .filter(|c| c.snippet.is_none())
            .collect::<Vec<_>>();
        assert_eq_dbg(expected_completions, &completions);
    }

    fn check_snippet_completion(code: &str, expected_completions: &str) {
        let (off, code) = extract_offset(&code);
        let file = File::parse(&code);
        let completions = scope_completion(&file, off)
            .unwrap()
            .into_iter()
            .filter(|c| c.snippet.is_some())
            .collect::<Vec<_>>();
        assert_eq_dbg(expected_completions, &completions);
    }

    #[test]
    fn test_completion_let_scope() {
        check_scope_completion(r"
            fn quux(x: i32) {
                let y = 92;
                1 + <|>;
                let z = ();
            }
            ", r#"[CompletionItem { label: "y", lookup: None, snippet: None },
                   CompletionItem { label: "x", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#);
    }

    #[test]
    fn test_completion_if_let_scope() {
        check_scope_completion(r"
            fn quux() {
                if let Some(x) = foo() {
                    let y = 92;
                };
                if let Some(a) = bar() {
                    let b = 62;
                    1 + <|>
                }
            }
            ", r#"[CompletionItem { label: "b", lookup: None, snippet: None },
                   CompletionItem { label: "a", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#);
    }

    #[test]
    fn test_completion_for_scope() {
        check_scope_completion(r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ", r#"[CompletionItem { label: "x", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#);
    }

    #[test]
    fn test_completion_mod_scope() {
        check_scope_completion(r"
            struct Foo;
            enum Baz {}
            fn quux() {
                <|>
            }
            ", r#"[CompletionItem { label: "Foo", lookup: None, snippet: None },
                   CompletionItem { label: "Baz", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#);
    }

    #[test]
    fn test_completion_mod_scope_no_self_use() {
        check_scope_completion(r"
            use foo<|>;
            ", r#"[]"#);
    }

    #[test]
    fn test_complete_type() {
        check_scope_completion(r"
            struct Foo;
            fn x() -> <|>
        ", r#"[CompletionItem { label: "Foo", lookup: None, snippet: None },
               CompletionItem { label: "x", lookup: None, snippet: None }]"#)
    }

    #[test]
    fn test_complete_shadowing() {
        check_scope_completion(r"
            fn foo() -> {
                let bar = 92;
                {
                    let bar = 62;
                    <|>
                }
            }
        ", r#"[CompletionItem { label: "bar", lookup: None, snippet: None },
               CompletionItem { label: "foo", lookup: None, snippet: None }]"#)
    }

    #[test]
    fn test_complete_self() {
        check_scope_completion(r"
            impl S { fn foo(&self) { <|> } }
        ", r#"[CompletionItem { label: "self", lookup: None, snippet: None }]"#)
    }

    #[test]
    fn test_completion_kewords() {
        check_snippet_completion(r"
            fn quux() {
                <|>
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return") }]"#);
    }

    #[test]
    fn test_completion_else() {
        check_snippet_completion(r"
            fn quux() {
                if true {
                    ()
                } <|>
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "else", lookup: None, snippet: Some("else {$0}") },
                   CompletionItem { label: "else if", lookup: None, snippet: Some("else if $0 {}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return") }]"#);
    }

    #[test]
    fn test_completion_return_value() {
        check_snippet_completion(r"
            fn quux() -> i32 {
                <|>
                92
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0;") }]"#);
        check_snippet_completion(r"
            fn quux() {
                <|>
                92
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return;") }]"#);
    }

    #[test]
    fn test_completion_return_no_stmt() {
        check_snippet_completion(r"
            fn quux() -> i32 {
                match () {
                    () => <|>
                }
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0") }]"#);
    }

    #[test]
    fn test_continue_break_completion() {
        check_snippet_completion(r"
            fn quux() -> i32 {
                loop { <|> }
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "continue", lookup: None, snippet: Some("continue") },
                   CompletionItem { label: "break", lookup: None, snippet: Some("break") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0") }]"#);
        check_snippet_completion(r"
            fn quux() -> i32 {
                loop { || { <|> } }
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0") }]"#);
    }

    #[test]
    fn test_param_completion_last_param() {
        check_scope_completion(r"
            fn foo(file_id: FileId) {}
            fn bar(file_id: FileId) {}
            fn baz(file<|>) {}
        ", r#"[CompletionItem { label: "file_id: FileId", lookup: Some("file_id"), snippet: None }]"#);
    }

    #[test]
    fn test_param_completion_nth_param() {
        check_scope_completion(r"
            fn foo(file_id: FileId) {}
            fn bar(file_id: FileId) {}
            fn baz(file<|>, x: i32) {}
        ", r#"[CompletionItem { label: "file_id: FileId", lookup: Some("file_id"), snippet: None }]"#);
    }

    #[test]
    fn test_param_completion_trait_param() {
        check_scope_completion(r"
            pub(crate) trait SourceRoot {
                pub fn contains(&self, file_id: FileId) -> bool;
                pub fn module_map(&self) -> &ModuleMap;
                pub fn lines(&self, file_id: FileId) -> &LineIndex;
                pub fn syntax(&self, file<|>)
            }
        ", r#"[CompletionItem { label: "self", lookup: None, snippet: None },
               CompletionItem { label: "SourceRoot", lookup: None, snippet: None },
               CompletionItem { label: "file_id: FileId", lookup: Some("file_id"), snippet: None }]"#);
    }
}

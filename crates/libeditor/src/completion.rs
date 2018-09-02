use std::collections::HashSet;

use libsyntax2::{
    File, TextUnit, AstNode, SyntaxKind::*,
    ast::{self, LoopBodyOwner},
    algo::{
        ancestors,
        visit::{visitor, Visitor},
    },
    text_utils::is_subrange,
};

use {
    AtomEdit, find_node_at_offset,
    scope::{FnScopes, ModuleScope},
};

#[derive(Debug)]
pub struct CompletionItem {
    pub name: String,
    pub snippet: Option<String>
}

pub fn scope_completion(file: &File, offset: TextUnit) -> Option<Vec<CompletionItem>> {
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        file.reparse(&edit)
    };
    let name_ref = find_node_at_offset::<ast::NameRef>(file.syntax(), offset)?;
    if !is_single_segment(name_ref) {
        return None;
    }

    let mut res = Vec::new();
    if let Some(fn_def) = ancestors(name_ref.syntax()).filter_map(ast::FnDef::cast).next() {
        complete_expr_keywords(&file, fn_def, name_ref, &mut res);
        let scopes = FnScopes::new(fn_def);
        complete_fn(name_ref, &scopes, &mut res);
    }
    if let Some(root) = ancestors(name_ref.syntax()).filter_map(ast::Root::cast).next() {
        let scope = ModuleScope::new(root);
        res.extend(
            scope.entries().iter()
                .filter(|entry| entry.syntax() != name_ref.syntax())
                .map(|entry| CompletionItem {
                    name: entry.name().to_string(),
                    snippet: None,
                })
        );
    }
    Some(res)
}

fn is_single_segment(name_ref: ast::NameRef) -> bool {
    match ancestors(name_ref.syntax()).filter_map(ast::Path::cast).next() {
        None => false,
        Some(path) => {
            path.syntax().range() == name_ref.syntax().range()
        }
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
        name: kw.to_string(),
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
                name: entry.name().to_string(),
                snippet: None,
            })
    );
    if scopes.self_param.is_some() {
        acc.push(CompletionItem {
            name: "self".to_string(),
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
            ", r#"[CompletionItem { name: "y", snippet: None },
                   CompletionItem { name: "x", snippet: None },
                   CompletionItem { name: "quux", snippet: None }]"#);
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
            ", r#"[CompletionItem { name: "b", snippet: None },
                   CompletionItem { name: "a", snippet: None },
                   CompletionItem { name: "quux", snippet: None }]"#);
    }

    #[test]
    fn test_completion_for_scope() {
        check_scope_completion(r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ", r#"[CompletionItem { name: "x", snippet: None },
                   CompletionItem { name: "quux", snippet: None }]"#);
    }

    #[test]
    fn test_completion_mod_scope() {
        check_scope_completion(r"
            struct Foo;
            enum Baz {}
            fn quux() {
                <|>
            }
            ", r#"[CompletionItem { name: "Foo", snippet: None },
                   CompletionItem { name: "Baz", snippet: None },
                   CompletionItem { name: "quux", snippet: None }]"#);
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
        ", r#"[CompletionItem { name: "Foo", snippet: None },
               CompletionItem { name: "x", snippet: None }]"#)
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
        ", r#"[CompletionItem { name: "bar", snippet: None },
               CompletionItem { name: "foo", snippet: None }]"#)
    }

    #[test]
    fn test_complete_self() {
        check_scope_completion(r"
            impl S { fn foo(&self) { <|> } }
        ", r#"[CompletionItem { name: "self", snippet: None }]"#)
    }

    #[test]
    fn test_completion_kewords() {
        check_snippet_completion(r"
            fn quux() {
                <|>
            }
            ", r#"[CompletionItem { name: "if", snippet: Some("if $0 {}") },
                   CompletionItem { name: "match", snippet: Some("match $0 {}") },
                   CompletionItem { name: "while", snippet: Some("while $0 {}") },
                   CompletionItem { name: "loop", snippet: Some("loop {$0}") },
                   CompletionItem { name: "return", snippet: Some("return") }]"#);
    }

    #[test]
    fn test_completion_else() {
        check_snippet_completion(r"
            fn quux() {
                if true {
                    ()
                } <|>
            }
            ", r#"[CompletionItem { name: "if", snippet: Some("if $0 {}") },
                   CompletionItem { name: "match", snippet: Some("match $0 {}") },
                   CompletionItem { name: "while", snippet: Some("while $0 {}") },
                   CompletionItem { name: "loop", snippet: Some("loop {$0}") },
                   CompletionItem { name: "else", snippet: Some("else {$0}") },
                   CompletionItem { name: "else if", snippet: Some("else if $0 {}") },
                   CompletionItem { name: "return", snippet: Some("return") }]"#);
    }

    #[test]
    fn test_completion_return_value() {
        check_snippet_completion(r"
            fn quux() -> i32 {
                <|>
                92
            }
            ", r#"[CompletionItem { name: "if", snippet: Some("if $0 {}") },
                   CompletionItem { name: "match", snippet: Some("match $0 {}") },
                   CompletionItem { name: "while", snippet: Some("while $0 {}") },
                   CompletionItem { name: "loop", snippet: Some("loop {$0}") },
                   CompletionItem { name: "return", snippet: Some("return $0;") }]"#);
        check_snippet_completion(r"
            fn quux() {
                <|>
                92
            }
            ", r#"[CompletionItem { name: "if", snippet: Some("if $0 {}") },
                   CompletionItem { name: "match", snippet: Some("match $0 {}") },
                   CompletionItem { name: "while", snippet: Some("while $0 {}") },
                   CompletionItem { name: "loop", snippet: Some("loop {$0}") },
                   CompletionItem { name: "return", snippet: Some("return;") }]"#);
    }

    #[test]
    fn test_completion_return_no_stmt() {
        check_snippet_completion(r"
            fn quux() -> i32 {
                match () {
                    () => <|>
                }
            }
            ", r#"[CompletionItem { name: "if", snippet: Some("if $0 {}") },
                   CompletionItem { name: "match", snippet: Some("match $0 {}") },
                   CompletionItem { name: "while", snippet: Some("while $0 {}") },
                   CompletionItem { name: "loop", snippet: Some("loop {$0}") },
                   CompletionItem { name: "return", snippet: Some("return $0") }]"#);
    }

    #[test]
    fn test_continue_break_completion() {
        check_snippet_completion(r"
            fn quux() -> i32 {
                loop { <|> }
            }
            ", r#"[CompletionItem { name: "if", snippet: Some("if $0 {}") },
                   CompletionItem { name: "match", snippet: Some("match $0 {}") },
                   CompletionItem { name: "while", snippet: Some("while $0 {}") },
                   CompletionItem { name: "loop", snippet: Some("loop {$0}") },
                   CompletionItem { name: "continue", snippet: Some("continue") },
                   CompletionItem { name: "break", snippet: Some("break") },
                   CompletionItem { name: "return", snippet: Some("return $0") }]"#);
        check_snippet_completion(r"
            fn quux() -> i32 {
                loop { || { <|> } }
            }
            ", r#"[CompletionItem { name: "if", snippet: Some("if $0 {}") },
                   CompletionItem { name: "match", snippet: Some("match $0 {}") },
                   CompletionItem { name: "while", snippet: Some("while $0 {}") },
                   CompletionItem { name: "loop", snippet: Some("loop {$0}") },
                   CompletionItem { name: "return", snippet: Some("return $0") }]"#);
    }
}

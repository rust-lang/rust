mod reference_completion;

use ra_editor::find_node_at_offset;
use ra_syntax::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast,
    AstNode, AtomEdit,
    SyntaxNodeRef,
};
use rustc_hash::{FxHashMap};

use crate::{
    db::{self, SyntaxDatabase},
    hir::
        ModuleDescriptor
    ,
    Cancelable, FilePosition
};

#[derive(Debug)]
pub struct CompletionItem {
    /// What user sees in pop-up
    pub label: String,
    /// What string is used for filtering, defaults to label
    pub lookup: Option<String>,
    /// What is inserted, defaults to label
    pub snippet: Option<String>,
}

pub(crate) fn completions(
    db: &db::RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<Vec<CompletionItem>>> {
    let original_file = db.file_syntax(position.file_id);
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomEdit::insert(position.offset, "intellijRulezz".to_string());
        original_file.reparse(&edit)
    };

    let module = ctry!(ModuleDescriptor::guess_from_position(db, position)?);

    let mut res = Vec::new();
    let mut has_completions = false;
    // First, let's try to complete a reference to some declaration.
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(file.syntax(), position.offset) {
        has_completions = true;
        reference_completion::completions(&mut res, db, &module, &file, name_ref)?;
        // special case, `trait T { fn foo(i_am_a_name_ref) {} }`
        if is_node::<ast::Param>(name_ref.syntax()) {
            param_completions(name_ref.syntax(), &mut res);
        }
    }

    // Otherwise, if this is a declaration, use heuristics to suggest a name.
    if let Some(name) = find_node_at_offset::<ast::Name>(file.syntax(), position.offset) {
        if is_node::<ast::Param>(name.syntax()) {
            has_completions = true;
            param_completions(name.syntax(), &mut res);
        }
    }
    let res = if has_completions { Some(res) } else { None };
    Ok(res)
}

fn param_completions(ctx: SyntaxNodeRef, acc: &mut Vec<CompletionItem>) {
    let mut params = FxHashMap::default();
    for node in ctx.ancestors() {
        let _ = visitor_ctx(&mut params)
            .visit::<ast::SourceFile, _>(process)
            .visit::<ast::ItemList, _>(process)
            .accept(node);
    }
    params
        .into_iter()
        .filter_map(|(label, (count, param))| {
            let lookup = param.pat()?.syntax().text().to_string();
            if count < 2 {
                None
            } else {
                Some((label, lookup))
            }
        })
        .for_each(|(label, lookup)| {
            acc.push(CompletionItem {
                label,
                lookup: Some(lookup),
                snippet: None,
            })
        });

    fn process<'a, N: ast::FnDefOwner<'a>>(
        node: N,
        params: &mut FxHashMap<String, (u32, ast::Param<'a>)>,
    ) {
        node.functions()
            .filter_map(|it| it.param_list())
            .flat_map(|it| it.params())
            .for_each(|param| {
                let text = param.syntax().text().to_string();
                params.entry(text).or_insert((0, param)).0 += 1;
            })
    }
}

fn is_node<'a, N: AstNode<'a>>(node: SyntaxNodeRef<'a>) -> bool {
    match node.ancestors().filter_map(N::cast).next() {
        None => false,
        Some(n) => n.syntax().range() == node.range(),
    }
}

#[cfg(test)]
mod tests {
    use test_utils::assert_eq_dbg;

    use crate::mock_analysis::single_file_with_position;

    use super::*;

    fn check_scope_completion(code: &str, expected_completions: &str) {
        let (analysis, position) = single_file_with_position(code);
        let completions = completions(&analysis.imp.db, position)
            .unwrap()
            .unwrap()
            .into_iter()
            .filter(|c| c.snippet.is_none())
            .collect::<Vec<_>>();
        assert_eq_dbg(expected_completions, &completions);
    }

    fn check_snippet_completion(code: &str, expected_completions: &str) {
        let (analysis, position) = single_file_with_position(code);
        let completions = completions(&analysis.imp.db, position)
            .unwrap()
            .unwrap()
            .into_iter()
            .filter(|c| c.snippet.is_some())
            .collect::<Vec<_>>();
        assert_eq_dbg(expected_completions, &completions);
    }

    #[test]
    fn test_completion_let_scope() {
        check_scope_completion(
            r"
            fn quux(x: i32) {
                let y = 92;
                1 + <|>;
                let z = ();
            }
            ",
            r#"[CompletionItem { label: "y", lookup: None, snippet: None },
                   CompletionItem { label: "x", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#,
        );
    }

    #[test]
    fn test_completion_if_let_scope() {
        check_scope_completion(
            r"
            fn quux() {
                if let Some(x) = foo() {
                    let y = 92;
                };
                if let Some(a) = bar() {
                    let b = 62;
                    1 + <|>
                }
            }
            ",
            r#"[CompletionItem { label: "b", lookup: None, snippet: None },
                   CompletionItem { label: "a", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#,
        );
    }

    #[test]
    fn test_completion_for_scope() {
        check_scope_completion(
            r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ",
            r#"[CompletionItem { label: "x", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#,
        );
    }

    #[test]
    fn test_completion_mod_scope() {
        check_scope_completion(
            r"
            struct Foo;
            enum Baz {}
            fn quux() {
                <|>
            }
            ",
            r#"[CompletionItem { label: "quux", lookup: None, snippet: None },
                CompletionItem { label: "Foo", lookup: None, snippet: None },
                CompletionItem { label: "Baz", lookup: None, snippet: None }]"#,
        );
    }

    #[test]
    fn test_completion_mod_scope_no_self_use() {
        check_scope_completion(
            r"
            use foo<|>;
            ",
            r#"[]"#,
        );
    }

    #[test]
    fn test_completion_self_path() {
        check_scope_completion(
            r"
            use self::m::<|>;

            mod m {
                struct Bar;
            }
            ",
            r#"[CompletionItem { label: "Bar", lookup: None, snippet: None }]"#,
        );
    }

    #[test]
    fn test_completion_mod_scope_nested() {
        check_scope_completion(
            r"
            struct Foo;
            mod m {
                struct Bar;
                fn quux() { <|> }
            }
            ",
            r#"[CompletionItem { label: "quux", lookup: None, snippet: None },
                CompletionItem { label: "Bar", lookup: None, snippet: None }]"#,
        );
    }

    #[test]
    fn test_complete_type() {
        check_scope_completion(
            r"
            struct Foo;
            fn x() -> <|>
        ",
            r#"[CompletionItem { label: "Foo", lookup: None, snippet: None },
               CompletionItem { label: "x", lookup: None, snippet: None }]"#,
        )
    }

    #[test]
    fn test_complete_shadowing() {
        check_scope_completion(
            r"
            fn foo() -> {
                let bar = 92;
                {
                    let bar = 62;
                    <|>
                }
            }
        ",
            r#"[CompletionItem { label: "bar", lookup: None, snippet: None },
               CompletionItem { label: "foo", lookup: None, snippet: None }]"#,
        )
    }

    #[test]
    fn test_complete_self() {
        check_scope_completion(
            r"
            impl S { fn foo(&self) { <|> } }
        ",
            r#"[CompletionItem { label: "self", lookup: None, snippet: None }]"#,
        )
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
                   CompletionItem { label: "return", lookup: None, snippet: Some("return") },
                   CompletionItem { label: "pd", lookup: None, snippet: Some("eprintln!(\"$0 = {:?}\", $0);") },
                   CompletionItem { label: "ppd", lookup: None, snippet: Some("eprintln!(\"$0 = {:#?}\", $0);") }]"#);
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
                   CompletionItem { label: "return", lookup: None, snippet: Some("return") },
                   CompletionItem { label: "pd", lookup: None, snippet: Some("eprintln!(\"$0 = {:?}\", $0);") },
                   CompletionItem { label: "ppd", lookup: None, snippet: Some("eprintln!(\"$0 = {:#?}\", $0);") }]"#);
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
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0;") },
                   CompletionItem { label: "pd", lookup: None, snippet: Some("eprintln!(\"$0 = {:?}\", $0);") },
                   CompletionItem { label: "ppd", lookup: None, snippet: Some("eprintln!(\"$0 = {:#?}\", $0);") }]"#);
        check_snippet_completion(r"
            fn quux() {
                <|>
                92
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return;") },
                   CompletionItem { label: "pd", lookup: None, snippet: Some("eprintln!(\"$0 = {:?}\", $0);") },
                   CompletionItem { label: "ppd", lookup: None, snippet: Some("eprintln!(\"$0 = {:#?}\", $0);") }]"#);
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
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0") },
                   CompletionItem { label: "pd", lookup: None, snippet: Some("eprintln!(\"$0 = {:?}\", $0);") },
                   CompletionItem { label: "ppd", lookup: None, snippet: Some("eprintln!(\"$0 = {:#?}\", $0);") }]"#);
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
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0") },
                   CompletionItem { label: "pd", lookup: None, snippet: Some("eprintln!(\"$0 = {:?}\", $0);") },
                   CompletionItem { label: "ppd", lookup: None, snippet: Some("eprintln!(\"$0 = {:#?}\", $0);") }]"#);
        check_snippet_completion(r"
            fn quux() -> i32 {
                loop { || { <|> } }
            }
            ", r#"[CompletionItem { label: "if", lookup: None, snippet: Some("if $0 {}") },
                   CompletionItem { label: "match", lookup: None, snippet: Some("match $0 {}") },
                   CompletionItem { label: "while", lookup: None, snippet: Some("while $0 {}") },
                   CompletionItem { label: "loop", lookup: None, snippet: Some("loop {$0}") },
                   CompletionItem { label: "return", lookup: None, snippet: Some("return $0") },
                   CompletionItem { label: "pd", lookup: None, snippet: Some("eprintln!(\"$0 = {:?}\", $0);") },
                   CompletionItem { label: "ppd", lookup: None, snippet: Some("eprintln!(\"$0 = {:#?}\", $0);") }]"#);
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

    #[test]
    fn test_item_snippets() {
        // check_snippet_completion(r"
        //     <|>
        //     ",
        //     r##"[CompletionItem { label: "tfn", lookup: None, snippet: Some("#[test]\nfn $1() {\n    $0\n}") }]"##,
        // );
        check_snippet_completion(r"
            #[cfg(test)]
            mod tests {
                <|>
            }
            ",
            r##"[CompletionItem { label: "tfn", lookup: None, snippet: Some("#[test]\nfn $1() {\n    $0\n}") },
                 CompletionItem { label: "pub(crate)", lookup: None, snippet: Some("pub(crate) $0") }]"##,
        );
    }
}

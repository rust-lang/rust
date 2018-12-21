use rustc_hash::{FxHashSet};
use ra_editor::find_node_at_offset;
use ra_syntax::{
    algo::visit::{visitor, Visitor},
    SourceFileNode, AstNode,
    ast::{self, LoopBodyOwner},
    SyntaxKind::*,
};
use hir::{
    self,
    FnScopes, Def, Path
};

use crate::{
    db::RootDatabase,
    completion::{CompletionItem, Completions, CompletionKind::*},
    Cancelable
};

pub(super) fn completions(
    acc: &mut Completions,
    db: &RootDatabase,
    module: &hir::Module,
    file: &SourceFileNode,
    name_ref: ast::NameRef,
) -> Cancelable<()> {
    let kind = match classify_name_ref(name_ref) {
        Some(it) => it,
        None => return Ok(()),
    };

    match kind {
        NameRefKind::LocalRef { enclosing_fn } => {
            if let Some(fn_def) = enclosing_fn {
                let scopes = FnScopes::new(fn_def);
                complete_fn(name_ref, &scopes, acc);
                complete_expr_keywords(&file, fn_def, name_ref, acc);
                complete_expr_snippets(acc);
            }

            let module_scope = module.scope(db)?;
            module_scope
                .entries()
                .filter(|(_name, res)| {
                    // Don't expose this item
                    match res.import {
                        None => true,
                        Some(import) => {
                            let range = import.range(db, module.source().file_id());
                            !range.is_subrange(&name_ref.syntax().range())
                        }
                    }
                })
                .for_each(|(name, _res)| {
                    CompletionItem::new(name.to_string())
                        .kind(Reference)
                        .add_to(acc)
                });
        }
        NameRefKind::Path(path) => complete_path(acc, db, module, path)?,
        NameRefKind::BareIdentInMod => {
            let name_range = name_ref.syntax().range();
            let top_node = name_ref
                .syntax()
                .ancestors()
                .take_while(|it| it.range() == name_range)
                .last()
                .unwrap();
            match top_node.parent().map(|it| it.kind()) {
                Some(SOURCE_FILE) | Some(ITEM_LIST) => complete_mod_item_snippets(acc),
                _ => (),
            }
        }
    }
    Ok(())
}

enum NameRefKind<'a> {
    /// NameRef is a part of single-segment path, for example, a refernece to a
    /// local variable.
    LocalRef {
        enclosing_fn: Option<ast::FnDef<'a>>,
    },
    /// NameRef is the last segment in some path
    Path(Path),
    /// NameRef is bare identifier at the module's root.
    /// Used for keyword completion
    BareIdentInMod,
}

fn classify_name_ref(name_ref: ast::NameRef) -> Option<NameRefKind> {
    let name_range = name_ref.syntax().range();
    let top_node = name_ref
        .syntax()
        .ancestors()
        .take_while(|it| it.range() == name_range)
        .last()
        .unwrap();
    match top_node.parent().map(|it| it.kind()) {
        Some(SOURCE_FILE) | Some(ITEM_LIST) => return Some(NameRefKind::BareIdentInMod),
        _ => (),
    }

    let parent = name_ref.syntax().parent()?;
    if let Some(segment) = ast::PathSegment::cast(parent) {
        let path = segment.parent_path();
        if let Some(path) = Path::from_ast(path) {
            if !path.is_ident() {
                return Some(NameRefKind::Path(path));
            }
        }
        if path.qualifier().is_none() {
            let enclosing_fn = name_ref
                .syntax()
                .ancestors()
                .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
                .find_map(ast::FnDef::cast);
            return Some(NameRefKind::LocalRef { enclosing_fn });
        }
    }
    None
}

fn complete_fn(name_ref: ast::NameRef, scopes: &FnScopes, acc: &mut Completions) {
    let mut shadowed = FxHashSet::default();
    scopes
        .scope_chain(name_ref.syntax())
        .flat_map(|scope| scopes.entries(scope).iter())
        .filter(|entry| shadowed.insert(entry.name()))
        .for_each(|entry| {
            CompletionItem::new(entry.name().to_string())
                .kind(Reference)
                .add_to(acc)
        });
    if scopes.self_param.is_some() {
        CompletionItem::new("self").kind(Reference).add_to(acc);
    }
}

fn complete_path(
    acc: &mut Completions,
    db: &RootDatabase,
    module: &hir::Module,
    mut path: Path,
) -> Cancelable<()> {
    if path.segments.is_empty() {
        return Ok(());
    }
    path.segments.pop();
    let def_id = match module.resolve_path(db, path)? {
        None => return Ok(()),
        Some(it) => it,
    };
    let target_module = match def_id.resolve(db)? {
        Def::Module(it) => it,
        _ => return Ok(()),
    };
    let module_scope = target_module.scope(db)?;
    module_scope.entries().for_each(|(name, _res)| {
        CompletionItem::new(name.to_string())
            .kind(Reference)
            .add_to(acc)
    });
    Ok(())
}

fn complete_mod_item_snippets(acc: &mut Completions) {
    CompletionItem::new("Test function")
        .lookup_by("tfn")
        .snippet(
            "\
#[test]
fn ${1:feature}() {
    $0
}",
        )
        .kind(Snippet)
        .add_to(acc);
    CompletionItem::new("pub(crate)")
        .snippet("pub(crate) $0")
        .kind(Snippet)
        .add_to(acc);
}

fn complete_expr_keywords(
    file: &SourceFileNode,
    fn_def: ast::FnDef,
    name_ref: ast::NameRef,
    acc: &mut Completions,
) {
    acc.add(keyword("if", "if $0 {}"));
    acc.add(keyword("match", "match $0 {}"));
    acc.add(keyword("while", "while $0 {}"));
    acc.add(keyword("loop", "loop {$0}"));

    if let Some(off) = name_ref.syntax().range().start().checked_sub(2.into()) {
        if let Some(if_expr) = find_node_at_offset::<ast::IfExpr>(file.syntax(), off) {
            if if_expr.syntax().range().end() < name_ref.syntax().range().start() {
                acc.add(keyword("else", "else {$0}"));
                acc.add(keyword("else if", "else if $0 {}"));
            }
        }
    }
    if is_in_loop_body(name_ref) {
        acc.add(keyword("continue", "continue"));
        acc.add(keyword("break", "break"));
    }
    acc.add_all(complete_return(fn_def, name_ref));
}

fn is_in_loop_body(name_ref: ast::NameRef) -> bool {
    for node in name_ref.syntax().ancestors() {
        if node.kind() == FN_DEF || node.kind() == LAMBDA_EXPR {
            break;
        }
        let loop_body = visitor()
            .visit::<ast::ForExpr, _>(LoopBodyOwner::loop_body)
            .visit::<ast::WhileExpr, _>(LoopBodyOwner::loop_body)
            .visit::<ast::LoopExpr, _>(LoopBodyOwner::loop_body)
            .accept(node);
        if let Some(Some(body)) = loop_body {
            if name_ref
                .syntax()
                .range()
                .is_subrange(&body.syntax().range())
            {
                return true;
            }
        }
    }
    false
}

fn complete_return(fn_def: ast::FnDef, name_ref: ast::NameRef) -> Option<CompletionItem> {
    // let is_last_in_block = name_ref.syntax().ancestors().filter_map(ast::Expr::cast)
    //     .next()
    //     .and_then(|it| it.syntax().parent())
    //     .and_then(ast::Block::cast)
    //     .is_some();

    // if is_last_in_block {
    //     return None;
    // }

    let is_stmt = match name_ref
        .syntax()
        .ancestors()
        .filter_map(ast::ExprStmt::cast)
        .next()
    {
        None => false,
        Some(expr_stmt) => expr_stmt.syntax().range() == name_ref.syntax().range(),
    };
    let snip = match (is_stmt, fn_def.ret_type().is_some()) {
        (true, true) => "return $0;",
        (true, false) => "return;",
        (false, true) => "return $0",
        (false, false) => "return",
    };
    Some(keyword("return", snip))
}

fn keyword(kw: &str, snippet: &str) -> CompletionItem {
    CompletionItem::new(kw)
        .kind(Keyword)
        .snippet(snippet)
        .build()
}

fn complete_expr_snippets(acc: &mut Completions) {
    CompletionItem::new("pd")
        .snippet("eprintln!(\"$0 = {:?}\", $0);")
        .kind(Snippet)
        .add_to(acc);
    CompletionItem::new("ppd")
        .snippet("eprintln!(\"$0 = {:#?}\", $0);")
        .kind(Snippet)
        .add_to(acc);
}

#[cfg(test)]
mod tests {
    use crate::completion::{CompletionKind, check_completion};

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    fn check_keyword_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Keyword);
    }

    fn check_snippet_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Snippet);
    }

    #[test]
    fn test_completion_let_scope() {
        check_reference_completion(
            r"
            fn quux(x: i32) {
                let y = 92;
                1 + <|>;
                let z = ();
            }
            ",
            "y;x;quux",
        );
    }

    #[test]
    fn test_completion_if_let_scope() {
        check_reference_completion(
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
            "b;a;quux",
        );
    }

    #[test]
    fn test_completion_for_scope() {
        check_reference_completion(
            r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ",
            "x;quux",
        );
    }

    #[test]
    fn test_completion_mod_scope() {
        check_reference_completion(
            r"
            struct Foo;
            enum Baz {}
            fn quux() {
                <|>
            }
            ",
            "quux;Foo;Baz",
        );
    }

    #[test]
    fn test_completion_mod_scope_no_self_use() {
        check_reference_completion(
            r"
            use foo<|>;
            ",
            "",
        );
    }

    #[test]
    fn test_completion_self_path() {
        check_reference_completion(
            r"
            use self::m::<|>;

            mod m {
                struct Bar;
            }
            ",
            "Bar",
        );
    }

    #[test]
    fn test_completion_mod_scope_nested() {
        check_reference_completion(
            r"
            struct Foo;
            mod m {
                struct Bar;
                fn quux() { <|> }
            }
            ",
            "quux;Bar",
        );
    }

    #[test]
    fn test_complete_type() {
        check_reference_completion(
            r"
            struct Foo;
            fn x() -> <|>
            ",
            "Foo;x",
        )
    }

    #[test]
    fn test_complete_shadowing() {
        check_reference_completion(
            r"
            fn foo() -> {
                let bar = 92;
                {
                    let bar = 62;
                    <|>
                }
            }
            ",
            "bar;foo",
        )
    }

    #[test]
    fn test_complete_self() {
        check_reference_completion(r"impl S { fn foo(&self) { <|> } }", "self")
    }

    #[test]
    fn test_complete_crate_path() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::Sp<|>
            ",
            "Spam;foo",
        );
    }

    #[test]
    fn test_complete_crate_path_with_braces() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::{Sp<|>};
            ",
            "Spam;foo",
        );
    }

    #[test]
    fn test_complete_crate_path_in_nested_tree() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            pub mod bar {
                pub mod baz {
                    pub struct Spam;
                }
            }
            //- /foo.rs
            use crate::{bar::{baz::Sp<|>}};
            ",
            "Spam",
        );
    }

    #[test]
    fn test_completion_kewords() {
        check_keyword_completion(
            r"
            fn quux() {
                <|>
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return"
            "#,
        );
    }

    #[test]
    fn test_completion_else() {
        check_keyword_completion(
            r"
            fn quux() {
                if true {
                    ()
                } <|>
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            else "else {$0}"
            else if "else if $0 {}"
            return "return"
            "#,
        );
    }

    #[test]
    fn test_completion_return_value() {
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                <|>
                92
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return $0;"
            "#,
        );
        check_keyword_completion(
            r"
            fn quux() {
                <|>
                92
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return;"
            "#,
        );
    }

    #[test]
    fn test_completion_return_no_stmt() {
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                match () {
                    () => <|>
                }
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return $0"
            "#,
        );
    }

    #[test]
    fn test_continue_break_completion() {
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                loop { <|> }
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            continue "continue"
            break "break"
            return "return $0"
            "#,
        );
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                loop { || { <|> } }
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return $0"
            "#,
        );
    }

    #[test]
    fn test_item_snippets() {
        // check_snippet_completion(r"
        //     <|>
        //     ",
        //     r##"[CompletionItem { label: "Test function", lookup: None, snippet: Some("#[test]\nfn test_${1:feature}() {\n$0\n}"##,
        // );
        check_snippet_completion(
            r"
            #[cfg(test)]
            mod tests {
                <|>
            }
            ",
            r##"
            tfn "Test function" "#[test]\nfn ${1:feature}() {\n    $0\n}"
            pub(crate) "pub(crate) $0"
        "##,
        );
    }

}

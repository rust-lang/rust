use ra_editor::find_node_at_offset;
use ra_syntax::{
    algo::visit::{visitor, visitor_ctx, Visitor, VisitorCtx},
    ast::{self, AstChildren, LoopBodyOwner, ModuleItemOwner},
    AstNode, AtomEdit, File,
    SyntaxKind::*,
    SyntaxNodeRef, TextUnit,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    db::{self, SyntaxDatabase},
    descriptors::function::FnScopes,
    descriptors::module::{ModuleId, ModuleScope, ModuleTree, ModuleSource},
    descriptors::DescriptorDatabase,
    input::FilesDatabase,
    Cancelable, FileId,
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

pub(crate) fn resolve_based_completion(
    db: &db::RootDatabase,
    file_id: FileId,
    offset: TextUnit,
) -> Cancelable<Option<Vec<CompletionItem>>> {
    let source_root_id = db.file_source_root(file_id);
    let file = db.file_syntax(file_id);
    let module_tree = db.module_tree(source_root_id)?;
    let module_id = match module_tree.any_module_for_source(ModuleSource::File(file_id)) {
        None => return Ok(None),
        Some(it) => it,
    };
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        file.reparse(&edit)
    };
    let target_module_id = match find_target_module(&module_tree, module_id, &file, offset) {
        None => return Ok(None),
        Some(it) => it,
    };
    let module_scope = db.module_scope(source_root_id, target_module_id)?;
    let res: Vec<_> = module_scope
        .entries()
        .iter()
        .map(|entry| CompletionItem {
            label: entry.name().to_string(),
            lookup: None,
            snippet: None,
        })
        .collect();
    Ok(Some(res))
}

pub(crate) fn find_target_module(
    module_tree: &ModuleTree,
    module_id: ModuleId,
    file: &File,
    offset: TextUnit,
) -> Option<ModuleId> {
    let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), offset)?;
    let mut crate_path = crate_path(name_ref)?;

    crate_path.pop();
    let mut target_module = module_id.root(&module_tree);
    for name in crate_path {
        target_module = target_module.child(module_tree, name.text().as_str())?;
    }
    Some(target_module)
}

fn crate_path(name_ref: ast::NameRef) -> Option<Vec<ast::NameRef>> {
    let mut path = name_ref
        .syntax()
        .parent()
        .and_then(ast::PathSegment::cast)?
        .parent_path();
    let mut res = Vec::new();
    loop {
        let segment = path.segment()?;
        match segment.kind()? {
            ast::PathSegmentKind::Name(name) => res.push(name),
            ast::PathSegmentKind::CrateKw => break,
            ast::PathSegmentKind::SelfKw | ast::PathSegmentKind::SuperKw => return None,
        }
        path = path.qualifier()?;
    }
    res.reverse();
    Some(res)
}

pub(crate) fn scope_completion(
    db: &db::RootDatabase,
    file_id: FileId,
    offset: TextUnit,
) -> Option<Vec<CompletionItem>> {
    let original_file = db.file_syntax(file_id);
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        original_file.reparse(&edit)
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
        let name_range = name_ref.syntax().range();
        let top_node = name_ref
            .syntax()
            .ancestors()
            .take_while(|it| it.range() == name_range)
            .last()
            .unwrap();
        match top_node.parent().map(|it| it.kind()) {
            Some(ROOT) | Some(ITEM_LIST) => complete_mod_item_snippets(&mut res),
            _ => (),
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

fn complete_module_items(
    file: &File,
    items: AstChildren<ast::ModuleItem>,
    this_item: Option<ast::NameRef>,
    acc: &mut Vec<CompletionItem>,
) {
    let scope = ModuleScope::new(items); // FIXME
    acc.extend(
        scope
            .entries()
            .iter()
            .filter(|entry| {
                let syntax = entry.ptr().resolve(file);
                Some(syntax.borrowed()) != this_item.map(|it| it.syntax())
            })
            .map(|entry| CompletionItem {
                label: entry.name().to_string(),
                lookup: None,
                snippet: None,
            }),
    );
}

fn complete_name_ref(file: &File, name_ref: ast::NameRef, acc: &mut Vec<CompletionItem>) {
    if !is_node::<ast::Path>(name_ref.syntax()) {
        return;
    }
    let mut visited_fn = false;
    for node in name_ref.syntax().ancestors() {
        if let Some(items) = visitor()
            .visit::<ast::Root, _>(|it| Some(it.items()))
            .visit::<ast::Module, _>(|it| Some(it.item_list()?.items()))
            .accept(node)
        {
            if let Some(items) = items {
                complete_module_items(file, items, Some(name_ref), acc);
            }
            break;
        } else if !visited_fn {
            if let Some(fn_def) = ast::FnDef::cast(node) {
                visited_fn = true;
                complete_expr_keywords(&file, fn_def, name_ref, acc);
                complete_expr_snippets(acc);
                let scopes = FnScopes::new(fn_def);
                complete_fn(name_ref, &scopes, acc);
            }
        }
    }
}

fn param_completions(ctx: SyntaxNodeRef, acc: &mut Vec<CompletionItem>) {
    let mut params = FxHashMap::default();
    for node in ctx.ancestors() {
        let _ = visitor_ctx(&mut params)
            .visit::<ast::Root, _>(process)
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

fn complete_expr_keywords(
    file: &File,
    fn_def: ast::FnDef,
    name_ref: ast::NameRef,
    acc: &mut Vec<CompletionItem>,
) {
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

fn keyword(kw: &str, snip: &str) -> CompletionItem {
    CompletionItem {
        label: kw.to_string(),
        lookup: None,
        snippet: Some(snip.to_string()),
    }
}

fn complete_expr_snippets(acc: &mut Vec<CompletionItem>) {
    acc.push(CompletionItem {
        label: "pd".to_string(),
        lookup: None,
        snippet: Some("eprintln!(\"$0 = {:?}\", $0);".to_string()),
    });
    acc.push(CompletionItem {
        label: "ppd".to_string(),
        lookup: None,
        snippet: Some("eprintln!(\"$0 = {:#?}\", $0);".to_string()),
    });
}

fn complete_mod_item_snippets(acc: &mut Vec<CompletionItem>) {
    acc.push(CompletionItem {
        label: "tfn".to_string(),
        lookup: None,
        snippet: Some("#[test]\nfn $1() {\n    $0\n}".to_string()),
    });
    acc.push(CompletionItem {
        label: "pub(crate)".to_string(),
        lookup: None,
        snippet: Some("pub(crate) $0".to_string()),
    })
}

fn complete_fn(name_ref: ast::NameRef, scopes: &FnScopes, acc: &mut Vec<CompletionItem>) {
    let mut shadowed = FxHashSet::default();
    acc.extend(
        scopes
            .scope_chain(name_ref.syntax())
            .flat_map(|scope| scopes.entries(scope).iter())
            .filter(|entry| shadowed.insert(entry.name()))
            .map(|entry| CompletionItem {
                label: entry.name().to_string(),
                lookup: None,
                snippet: None,
            }),
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
    use test_utils::assert_eq_dbg;

    use crate::mock_analysis::single_file_with_position;

    use super::*;

    fn check_scope_completion(code: &str, expected_completions: &str) {
        let (analysis, position) = single_file_with_position(code);
        let completions = scope_completion(&analysis.imp.db, position.file_id, position.offset)
            .unwrap()
            .into_iter()
            .filter(|c| c.snippet.is_none())
            .collect::<Vec<_>>();
        assert_eq_dbg(expected_completions, &completions);
    }

    fn check_snippet_completion(code: &str, expected_completions: &str) {
        let (analysis, position) = single_file_with_position(code);
        let completions = scope_completion(&analysis.imp.db, position.file_id, position.offset)
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
            r#"[CompletionItem { label: "Foo", lookup: None, snippet: None },
                   CompletionItem { label: "Baz", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#,
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
    fn test_completion_mod_scope_nested() {
        check_scope_completion(
            r"
            struct Foo;
            mod m {
                struct Bar;
                fn quux() { <|> }
            }
            ",
            r#"[CompletionItem { label: "Bar", lookup: None, snippet: None },
                   CompletionItem { label: "quux", lookup: None, snippet: None }]"#,
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

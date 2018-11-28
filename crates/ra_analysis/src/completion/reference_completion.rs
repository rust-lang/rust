use rustc_hash::{FxHashSet};
use ra_editor::find_node_at_offset;
use ra_syntax::{
    algo::visit::{visitor, Visitor},
    SourceFileNode, AstNode,
    ast::{self, LoopBodyOwner},
    SyntaxKind::*,
};

use crate::{
    db::RootDatabase,
    completion::CompletionItem,
    descriptors::{
        module::{ModuleDescriptor},
        function::FnScopes,
        Path,
    },
    Cancelable
};

pub(super) fn completions(
    acc: &mut Vec<CompletionItem>,
    db: &RootDatabase,
    module: &ModuleDescriptor,
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
            acc.extend(
                module_scope
                    .items
                    .iter()
                    .filter(|(_name, res)| {
                        // Don't expose this item
                        match res.import_name {
                            None => true,
                            Some(ptr) => !ptr.range().is_subrange(&name_ref.syntax().range()),
                        }
                    })
                    .map(|(name, _res)| CompletionItem {
                        label: name.to_string(),
                        lookup: None,
                        snippet: None,
                    }),
            );
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

fn complete_path(
    acc: &mut Vec<CompletionItem>,
    db: &RootDatabase,
    module: &ModuleDescriptor,
    mut path: Path,
) -> Cancelable<()> {
    if path.segments.is_empty() {
        return Ok(());
    }
    path.segments.pop();
    let target_module = match module.resolve_path(path) {
        None => return Ok(()),
        Some(it) => it,
    };
    let module_scope = target_module.scope(db)?;
    let completions = module_scope
        .items
        .iter()
        .map(|(name, _res)| CompletionItem {
            label: name.to_string(),
            lookup: None,
            snippet: None,
        });
    acc.extend(completions);
    Ok(())
}

fn complete_mod_item_snippets(acc: &mut Vec<CompletionItem>) {
    acc.push(CompletionItem {
        label: "Test function".to_string(),
        lookup: None,
        snippet: Some("#[test]\n\
            fn test_${1:feature}() {\n\
               $0\n\
            }".to_string()),
    });
    acc.push(CompletionItem {
        label: "pub(crate)".to_string(),
        lookup: None,
        snippet: Some("pub(crate) $0".to_string()),
    })
}

fn complete_expr_keywords(
    file: &SourceFileNode,
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

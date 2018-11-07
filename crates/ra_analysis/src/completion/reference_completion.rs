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
    input::{SourceRootId},
    completion::CompletionItem,
    descriptors::module::{ModuleId, ModuleScope, ModuleTree},
    descriptors::function::FnScopes,
    descriptors::DescriptorDatabase,
    Cancelable
};

pub(super) fn completions(
    acc: &mut Vec<CompletionItem>,
    db: &RootDatabase,
    source_root_id: SourceRootId,
    module_tree: &ModuleTree,
    module_id: ModuleId,
    file: &SourceFileNode,
    name_ref: ast::NameRef,
) -> Cancelable<()> {
    let kind = match classify_name_ref(name_ref) {
        Some(it) => it,
        None => return Ok(()),
    };

    match kind {
        NameRefKind::LocalRef => {
            let module_scope = db.module_scope(source_root_id, module_id)?;
            if let Some(fn_def) = complete_local_name(acc, &module_scope, name_ref) {
                complete_expr_keywords(&file, fn_def, name_ref, acc);
                complete_expr_snippets(acc);
            }
        }
        NameRefKind::CratePath(path) => {
            complete_path(acc, db, source_root_id, module_tree, module_id, path)?
        }
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
    LocalRef,
    /// NameRef is the last segment in crate:: path
    CratePath(Vec<ast::NameRef<'a>>),
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
        if path.qualifier().is_none() {
            return Some(NameRefKind::LocalRef);
        }
        if let Some(crate_path) = crate_path(path) {
            return Some(NameRefKind::CratePath(crate_path));
        }
    }
    None
}

fn crate_path(mut path: ast::Path) -> Option<Vec<ast::NameRef>> {
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

fn complete_local_name<'a>(
    acc: &mut Vec<CompletionItem>,
    module_scope: &ModuleScope,
    name_ref: ast::NameRef<'a>,
) -> Option<ast::FnDef<'a>> {
    let enclosing_fn = name_ref
        .syntax()
        .ancestors()
        .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
        .find_map(ast::FnDef::cast);
    if let Some(fn_def) = enclosing_fn {
        let scopes = FnScopes::new(fn_def);
        complete_fn(name_ref, &scopes, acc);
    }

    acc.extend(
        module_scope
            .entries()
            .iter()
            .filter(|entry| {
                // Don't expose this item
                !entry.ptr().range().is_subrange(&name_ref.syntax().range())
            })
            .map(|entry| CompletionItem {
                label: entry.name().to_string(),
                lookup: None,
                snippet: None,
            }),
    );
    enclosing_fn
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
    source_root_id: SourceRootId,
    module_tree: &ModuleTree,
    module_id: ModuleId,
    crate_path: Vec<ast::NameRef>,
) -> Cancelable<()> {
    let target_module_id = match find_target_module(module_tree, module_id, crate_path) {
        None => return Ok(()),
        Some(it) => it,
    };
    let module_scope = db.module_scope(source_root_id, target_module_id)?;
    let completions = module_scope.entries().iter().map(|entry| CompletionItem {
        label: entry.name().to_string(),
        lookup: None,
        snippet: None,
    });
    acc.extend(completions);
    Ok(())
}

fn find_target_module(
    module_tree: &ModuleTree,
    module_id: ModuleId,
    mut crate_path: Vec<ast::NameRef>,
) -> Option<ModuleId> {
    crate_path.pop();
    let mut target_module = module_id.root(&module_tree);
    for name in crate_path {
        target_module = target_module.child(module_tree, name.text().as_str())?;
    }
    Some(target_module)
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

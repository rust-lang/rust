//! A module with ide helpers for high-level ide features.
pub mod famous_defs;
pub mod generated_lints;
pub mod import_assets;
pub mod insert_use;
pub mod merge_imports;
pub mod insert_whitespace_into_node;
pub mod node_ext;
pub mod rust_doc;

use std::{collections::VecDeque, iter};

use base_db::FileId;
use hir::{ItemInNs, MacroDef, ModuleDef, Name, PathResolution, Semantics};
use itertools::Itertools;
use syntax::{
    ast::{self, make, HasLoopBody},
    AstNode, AstToken, Direction, SyntaxElement, SyntaxKind, SyntaxToken, TokenAtOffset, WalkEvent,
    T,
};

use crate::{defs::Definition, RootDatabase};

pub use self::famous_defs::FamousDefs;

pub fn item_name(db: &RootDatabase, item: ItemInNs) -> Option<Name> {
    match item {
        ItemInNs::Types(module_def_id) => ModuleDef::from(module_def_id).name(db),
        ItemInNs::Values(module_def_id) => ModuleDef::from(module_def_id).name(db),
        ItemInNs::Macros(macro_def_id) => MacroDef::from(macro_def_id).name(db),
    }
}

/// Parses and returns the derive path at the cursor position in the given attribute, if it is a derive.
/// This special case is required because the derive macro is a compiler builtin that discards the input derives.
///
/// The returned path is synthesized from TokenTree tokens and as such cannot be used with the [`Semantics`].
pub fn get_path_in_derive_attr(
    sema: &hir::Semantics<RootDatabase>,
    attr: &ast::Attr,
    cursor: &ast::Ident,
) -> Option<ast::Path> {
    let path = attr.path()?;
    let tt = attr.token_tree()?;
    if !tt.syntax().text_range().contains_range(cursor.syntax().text_range()) {
        return None;
    }
    let scope = sema.scope(attr.syntax());
    let resolved_attr = sema.resolve_path(&path)?;
    let derive = FamousDefs(sema, scope.krate()).core_macros_builtin_derive()?;
    if PathResolution::Macro(derive) != resolved_attr {
        return None;
    }
    get_path_at_cursor_in_tt(cursor)
}

/// Parses the path the identifier is part of inside a token tree.
pub fn get_path_at_cursor_in_tt(cursor: &ast::Ident) -> Option<ast::Path> {
    let cursor = cursor.syntax();
    let first = cursor
        .siblings_with_tokens(Direction::Prev)
        .filter_map(SyntaxElement::into_token)
        .take_while(|tok| tok.kind() != T!['('] && tok.kind() != T![,])
        .last()?;
    let path_tokens = first
        .siblings_with_tokens(Direction::Next)
        .filter_map(SyntaxElement::into_token)
        .take_while(|tok| tok != cursor);

    syntax::hacks::parse_expr_from_str(&path_tokens.chain(iter::once(cursor.clone())).join(""))
        .and_then(|expr| match expr {
            ast::Expr::PathExpr(it) => it.path(),
            _ => None,
        })
}

/// Picks the token with the highest rank returned by the passed in function.
pub fn pick_best_token(
    tokens: TokenAtOffset<SyntaxToken>,
    f: impl Fn(SyntaxKind) -> usize,
) -> Option<SyntaxToken> {
    tokens.max_by_key(move |t| f(t.kind()))
}

/// Converts the mod path struct into its ast representation.
pub fn mod_path_to_ast(path: &hir::ModPath) -> ast::Path {
    let _p = profile::span("mod_path_to_ast");

    let mut segments = Vec::new();
    let mut is_abs = false;
    match path.kind {
        hir::PathKind::Plain => {}
        hir::PathKind::Super(0) => segments.push(make::path_segment_self()),
        hir::PathKind::Super(n) => segments.extend((0..n).map(|_| make::path_segment_super())),
        hir::PathKind::DollarCrate(_) | hir::PathKind::Crate => {
            segments.push(make::path_segment_crate())
        }
        hir::PathKind::Abs => is_abs = true,
    }

    segments.extend(
        path.segments()
            .iter()
            .map(|segment| make::path_segment(make::name_ref(&segment.to_smol_str()))),
    );
    make::path_from_segments(segments, is_abs)
}

/// Iterates all `ModuleDef`s and `Impl` blocks of the given file.
pub fn visit_file_defs(
    sema: &Semantics<RootDatabase>,
    file_id: FileId,
    cb: &mut dyn FnMut(Definition),
) {
    let db = sema.db;
    let module = match sema.to_module_def(file_id) {
        Some(it) => it,
        None => return,
    };
    let mut defs: VecDeque<_> = module.declarations(db).into();
    while let Some(def) = defs.pop_front() {
        if let ModuleDef::Module(submodule) = def {
            if let hir::ModuleSource::Module(_) = submodule.definition_source(db).value {
                defs.extend(submodule.declarations(db));
                submodule.impl_defs(db).into_iter().for_each(|impl_| cb(impl_.into()));
            }
        }
        cb(def.into());
    }
    module.impl_defs(db).into_iter().for_each(|impl_| cb(impl_.into()));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnippetCap {
    _private: (),
}

impl SnippetCap {
    pub const fn new(allow_snippets: bool) -> Option<SnippetCap> {
        if allow_snippets {
            Some(SnippetCap { _private: () })
        } else {
            None
        }
    }
}

/// Calls `cb` on each expression inside `expr` that is at "tail position".
/// Does not walk into `break` or `return` expressions.
/// Note that modifying the tree while iterating it will cause undefined iteration which might
/// potentially results in an out of bounds panic.
pub fn for_each_tail_expr(expr: &ast::Expr, cb: &mut dyn FnMut(&ast::Expr)) {
    match expr {
        ast::Expr::BlockExpr(b) => {
            match b.modifier() {
                Some(
                    ast::BlockModifier::Async(_)
                    | ast::BlockModifier::Try(_)
                    | ast::BlockModifier::Const(_),
                ) => return cb(expr),

                Some(ast::BlockModifier::Label(label)) => {
                    for_each_break_expr(Some(label), b.stmt_list(), &mut |b| {
                        cb(&ast::Expr::BreakExpr(b))
                    });
                }
                Some(ast::BlockModifier::Unsafe(_)) => (),
                None => (),
            }
            if let Some(stmt_list) = b.stmt_list() {
                if let Some(e) = stmt_list.tail_expr() {
                    for_each_tail_expr(&e, cb);
                }
            }
        }
        ast::Expr::IfExpr(if_) => {
            let mut if_ = if_.clone();
            loop {
                if let Some(block) = if_.then_branch() {
                    for_each_tail_expr(&ast::Expr::BlockExpr(block), cb);
                }
                match if_.else_branch() {
                    Some(ast::ElseBranch::IfExpr(it)) => if_ = it,
                    Some(ast::ElseBranch::Block(block)) => {
                        for_each_tail_expr(&ast::Expr::BlockExpr(block), cb);
                        break;
                    }
                    None => break,
                }
            }
        }
        ast::Expr::LoopExpr(l) => {
            for_each_break_expr(l.label(), l.loop_body().and_then(|it| it.stmt_list()), &mut |b| {
                cb(&ast::Expr::BreakExpr(b))
            })
        }
        ast::Expr::MatchExpr(m) => {
            if let Some(arms) = m.match_arm_list() {
                arms.arms().filter_map(|arm| arm.expr()).for_each(|e| for_each_tail_expr(&e, cb));
            }
        }
        ast::Expr::ArrayExpr(_)
        | ast::Expr::AwaitExpr(_)
        | ast::Expr::BinExpr(_)
        | ast::Expr::BoxExpr(_)
        | ast::Expr::BreakExpr(_)
        | ast::Expr::CallExpr(_)
        | ast::Expr::CastExpr(_)
        | ast::Expr::ClosureExpr(_)
        | ast::Expr::ContinueExpr(_)
        | ast::Expr::FieldExpr(_)
        | ast::Expr::ForExpr(_)
        | ast::Expr::IndexExpr(_)
        | ast::Expr::Literal(_)
        | ast::Expr::MacroCall(_)
        | ast::Expr::MacroStmts(_)
        | ast::Expr::MethodCallExpr(_)
        | ast::Expr::ParenExpr(_)
        | ast::Expr::PathExpr(_)
        | ast::Expr::PrefixExpr(_)
        | ast::Expr::RangeExpr(_)
        | ast::Expr::RecordExpr(_)
        | ast::Expr::RefExpr(_)
        | ast::Expr::ReturnExpr(_)
        | ast::Expr::TryExpr(_)
        | ast::Expr::TupleExpr(_)
        | ast::Expr::WhileExpr(_)
        | ast::Expr::YieldExpr(_) => cb(expr),
    }
}

/// Calls `cb` on each break expr inside of `body` that is applicable for the given label.
pub fn for_each_break_expr(
    label: Option<ast::Label>,
    body: Option<ast::StmtList>,
    cb: &mut dyn FnMut(ast::BreakExpr),
) {
    let label = label.and_then(|lbl| lbl.lifetime());
    let mut depth = 0;
    if let Some(b) = body {
        let preorder = &mut b.syntax().preorder();
        let ev_as_expr = |ev| match ev {
            WalkEvent::Enter(it) => Some(WalkEvent::Enter(ast::Expr::cast(it)?)),
            WalkEvent::Leave(it) => Some(WalkEvent::Leave(ast::Expr::cast(it)?)),
        };
        let eq_label = |lt: Option<ast::Lifetime>| {
            lt.zip(label.as_ref()).map_or(false, |(lt, lbl)| lt.text() == lbl.text())
        };
        while let Some(node) = preorder.find_map(ev_as_expr) {
            match node {
                WalkEvent::Enter(expr) => match expr {
                    ast::Expr::LoopExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::ForExpr(_) => {
                        depth += 1
                    }
                    ast::Expr::BlockExpr(e) if e.label().is_some() => depth += 1,
                    ast::Expr::BreakExpr(b)
                        if (depth == 0 && b.lifetime().is_none()) || eq_label(b.lifetime()) =>
                    {
                        cb(b);
                    }
                    _ => (),
                },
                WalkEvent::Leave(expr) => match expr {
                    ast::Expr::LoopExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::ForExpr(_) => {
                        depth -= 1
                    }
                    ast::Expr::BlockExpr(e) if e.label().is_some() => depth -= 1,
                    _ => (),
                },
            }
        }
    }
}

/// Checks if the given lint is equal or is contained by the other lint which may or may not be a group.
pub fn lint_eq_or_in_group(lint: &str, lint_is: &str) -> bool {
    if lint == lint_is {
        return true;
    }

    if let Some(group) = generated_lints::DEFAULT_LINT_GROUPS
        .iter()
        .chain(generated_lints::CLIPPY_LINT_GROUPS.iter())
        .chain(generated_lints::RUSTDOC_LINT_GROUPS.iter())
        .find(|&check| check.lint.label == lint_is)
    {
        group.children.contains(&lint)
    } else {
        false
    }
}

/// Parses the input token tree as comma separated plain paths.
pub fn parse_tt_as_comma_sep_paths(input: ast::TokenTree) -> Option<Vec<ast::Path>> {
    let r_paren = input.r_paren_token();
    let tokens =
        input.syntax().children_with_tokens().skip(1).map_while(|it| match it.into_token() {
            // seeing a keyword means the attribute is unclosed so stop parsing here
            Some(tok) if tok.kind().is_keyword() => None,
            // don't include the right token tree parenthesis if it exists
            tok @ Some(_) if tok == r_paren => None,
            // only nodes that we can find are other TokenTrees, those are unexpected in this parse though
            None => None,
            Some(tok) => Some(tok),
        });
    let input_expressions = tokens.into_iter().group_by(|tok| tok.kind() == T![,]);
    let paths = input_expressions
        .into_iter()
        .filter_map(|(is_sep, group)| (!is_sep).then(|| group))
        .filter_map(|mut tokens| {
            syntax::hacks::parse_expr_from_str(&tokens.join("")).and_then(|expr| match expr {
                ast::Expr::PathExpr(it) => it.path(),
                _ => None,
            })
        })
        .collect();
    Some(paths)
}

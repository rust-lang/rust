//! A module with ide helpers for high-level ide features.
pub mod import_assets;
pub mod insert_use;
pub mod merge_imports;
pub mod rust_doc;
pub mod generated_lints;

use std::collections::VecDeque;

use base_db::FileId;
use either::Either;
use hir::{Crate, Enum, ItemInNs, MacroDef, Module, ModuleDef, Name, ScopeDef, Semantics, Trait};
use syntax::{
    ast::{self, make, LoopBodyOwner},
    AstNode, Direction, SyntaxElement, SyntaxKind, SyntaxToken, TokenAtOffset, WalkEvent, T,
};

use crate::RootDatabase;

pub fn item_name(db: &RootDatabase, item: ItemInNs) -> Option<Name> {
    match item {
        ItemInNs::Types(module_def_id) => ModuleDef::from(module_def_id).name(db),
        ItemInNs::Values(module_def_id) => ModuleDef::from(module_def_id).name(db),
        ItemInNs::Macros(macro_def_id) => MacroDef::from(macro_def_id).name(db),
    }
}

/// Resolves the path at the cursor token as a derive macro if it inside a token tree of a derive attribute.
pub fn try_resolve_derive_input_at(
    sema: &Semantics<RootDatabase>,
    derive_attr: &ast::Attr,
    cursor: &SyntaxToken,
) -> Option<MacroDef> {
    use itertools::Itertools;
    if cursor.kind() != T![ident] {
        return None;
    }
    let tt = match derive_attr.as_simple_call() {
        Some((name, tt))
            if name == "derive" && tt.syntax().text_range().contains_range(cursor.text_range()) =>
        {
            tt
        }
        _ => return None,
    };
    let tokens: Vec<_> = cursor
        .siblings_with_tokens(Direction::Prev)
        .flat_map(SyntaxElement::into_token)
        .take_while(|tok| tok.kind() != T!['('] && tok.kind() != T![,])
        .collect();
    let path = ast::Path::parse(&tokens.into_iter().rev().join("")).ok()?;
    match sema.scope(tt.syntax()).speculative_resolve(&path) {
        Some(hir::PathResolution::Macro(makro)) if makro.kind() == hir::MacroKind::Derive => {
            Some(makro)
        }
        _ => None,
    }
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
            .map(|segment| make::path_segment(make::name_ref(&segment.to_string()))),
    );
    make::path_from_segments(segments, is_abs)
}

/// Iterates all `ModuleDef`s and `Impl` blocks of the given file.
pub fn visit_file_defs(
    sema: &Semantics<RootDatabase>,
    file_id: FileId,
    cb: &mut dyn FnMut(Either<hir::ModuleDef, hir::Impl>),
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
                submodule.impl_defs(db).into_iter().for_each(|impl_| cb(Either::Right(impl_)));
            }
        }
        cb(Either::Left(def));
    }
    module.impl_defs(db).into_iter().for_each(|impl_| cb(Either::Right(impl_)));
}

/// Helps with finding well-know things inside the standard library. This is
/// somewhat similar to the known paths infra inside hir, but it different; We
/// want to make sure that IDE specific paths don't become interesting inside
/// the compiler itself as well.
///
/// Note that, by default, rust-analyzer tests **do not** include core or std
/// libraries. If you are writing tests for functionality using [`FamousDefs`],
/// you'd want to include minicore (see `test_utils::MiniCore`) declaration at
/// the start of your tests:
///
/// ```
/// //- minicore: iterator, ord, derive
/// ```
pub struct FamousDefs<'a, 'b>(pub &'a Semantics<'b, RootDatabase>, pub Option<Crate>);

#[allow(non_snake_case)]
impl FamousDefs<'_, '_> {
    pub fn std(&self) -> Option<Crate> {
        self.find_crate("std")
    }

    pub fn core(&self) -> Option<Crate> {
        self.find_crate("core")
    }

    pub fn core_cmp_Ord(&self) -> Option<Trait> {
        self.find_trait("core:cmp:Ord")
    }

    pub fn core_convert_From(&self) -> Option<Trait> {
        self.find_trait("core:convert:From")
    }

    pub fn core_convert_Into(&self) -> Option<Trait> {
        self.find_trait("core:convert:Into")
    }

    pub fn core_option_Option(&self) -> Option<Enum> {
        self.find_enum("core:option:Option")
    }

    pub fn core_result_Result(&self) -> Option<Enum> {
        self.find_enum("core:result:Result")
    }

    pub fn core_default_Default(&self) -> Option<Trait> {
        self.find_trait("core:default:Default")
    }

    pub fn core_iter_Iterator(&self) -> Option<Trait> {
        self.find_trait("core:iter:traits:iterator:Iterator")
    }

    pub fn core_iter_IntoIterator(&self) -> Option<Trait> {
        self.find_trait("core:iter:traits:collect:IntoIterator")
    }

    pub fn core_iter(&self) -> Option<Module> {
        self.find_module("core:iter")
    }

    pub fn core_ops_Deref(&self) -> Option<Trait> {
        self.find_trait("core:ops:Deref")
    }

    fn find_trait(&self, path: &str) -> Option<Trait> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Trait(it)) => Some(it),
            _ => None,
        }
    }

    fn find_enum(&self, path: &str) -> Option<Enum> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Adt(hir::Adt::Enum(it))) => Some(it),
            _ => None,
        }
    }

    fn find_module(&self, path: &str) -> Option<Module> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(it)) => Some(it),
            _ => None,
        }
    }

    fn find_crate(&self, name: &str) -> Option<Crate> {
        let krate = self.1?;
        let db = self.0.db;
        let res =
            krate.dependencies(db).into_iter().find(|dep| dep.name.to_string() == name)?.krate;
        Some(res)
    }

    fn find_def(&self, path: &str) -> Option<ScopeDef> {
        let db = self.0.db;
        let mut path = path.split(':');
        let trait_ = path.next_back()?;
        let std_crate = path.next()?;
        let std_crate = self.find_crate(std_crate)?;
        let mut module = std_crate.root_module(db);
        for segment in path {
            module = module.children(db).find_map(|child| {
                let name = child.name(db)?;
                if name.to_string() == segment {
                    Some(child)
                } else {
                    None
                }
            })?;
        }
        let def =
            module.scope(db, None).into_iter().find(|(name, _def)| name.to_string() == trait_)?.1;
        Some(def)
    }
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
            if let Some(e) = b.tail_expr() {
                for_each_tail_expr(&e, cb);
            }
        }
        ast::Expr::EffectExpr(e) => match e.effect() {
            ast::Effect::Label(label) => {
                for_each_break_expr(Some(label), e.block_expr(), &mut |b| {
                    cb(&ast::Expr::BreakExpr(b))
                });
                if let Some(b) = e.block_expr() {
                    for_each_tail_expr(&ast::Expr::BlockExpr(b), cb);
                }
            }
            ast::Effect::Unsafe(_) => {
                if let Some(e) = e.block_expr().and_then(|b| b.tail_expr()) {
                    for_each_tail_expr(&e, cb);
                }
            }
            ast::Effect::Async(_) | ast::Effect::Try(_) | ast::Effect::Const(_) => cb(expr),
        },
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
            for_each_break_expr(l.label(), l.loop_body(), &mut |b| cb(&ast::Expr::BreakExpr(b)))
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
    body: Option<ast::BlockExpr>,
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
                    ast::Expr::EffectExpr(e) if e.label().is_some() => depth += 1,
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
                    ast::Expr::EffectExpr(e) if e.label().is_some() => depth -= 1,
                    _ => (),
                },
            }
        }
    }
}

use std::iter;

use ast::make;
use either::Either;
use hir::{HirDisplay, Local};
use ide_db::{
    defs::{Definition, NameRefClass},
    search::{FileReference, ReferenceAccess, SearchScope},
};
use itertools::Itertools;
use stdx::format_to;
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        AstNode,
    },
    ted,
    SyntaxKind::{self, BLOCK_EXPR, BREAK_EXPR, COMMENT, PATH_EXPR, RETURN_EXPR},
    SyntaxNode, SyntaxToken, TextRange, TextSize, TokenAtOffset, WalkEvent, T,
};

use crate::{
    assist_context::{AssistContext, Assists, TreeMutator},
    AssistId,
};

// Assist: extract_function
//
// Extracts selected statements into new function.
//
// ```
// fn main() {
//     let n = 1;
//     $0let m = n + 2;
//     let k = m + n;$0
//     let g = 3;
// }
// ```
// ->
// ```
// fn main() {
//     let n = 1;
//     fun_name(n);
//     let g = 3;
// }
//
// fn $0fun_name(n: i32) {
//     let m = n + 2;
//     let k = m + n;
// }
// ```
pub(crate) fn extract_function(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if ctx.frange.range.is_empty() {
        return None;
    }

    let node = ctx.covering_element();
    if node.kind() == COMMENT {
        cov_mark::hit!(extract_function_in_comment_is_not_applicable);
        return None;
    }

    let node = match node {
        syntax::NodeOrToken::Node(n) => n,
        syntax::NodeOrToken::Token(t) => t.parent()?,
    };

    let body = extraction_target(&node, ctx.frange.range)?;

    let vars_used_in_body = vars_used_in_body(ctx, &body);
    let self_param = self_param_from_usages(ctx, &body, &vars_used_in_body);

    let anchor = if self_param.is_some() { Anchor::Method } else { Anchor::Freestanding };
    let insert_after = scope_for_fn_insertion(&body, anchor)?;
    let module = ctx.sema.scope(&insert_after).module()?;

    let vars_defined_in_body_and_outlive =
        vars_defined_in_body_and_outlive(ctx, &body, node.parent().as_ref().unwrap_or(&node));
    let ret_ty = body_return_ty(ctx, &body)?;

    // FIXME: we compute variables that outlive here just to check `never!` condition
    //        this requires traversing whole `body` (cheap) and finding all references (expensive)
    //        maybe we can move this check to `edit` closure somehow?
    if stdx::never!(!vars_defined_in_body_and_outlive.is_empty() && !ret_ty.is_unit()) {
        // We should not have variables that outlive body if we have expression block
        return None;
    }
    let control_flow = external_control_flow(ctx, &body)?;

    let target_range = body.text_range();

    acc.add(
        AssistId("extract_function", crate::AssistKind::RefactorExtract),
        "Extract into function",
        target_range,
        move |builder| {
            let params = extracted_function_params(ctx, &body, &vars_used_in_body);

            let fun = Function {
                name: "fun_name".to_string(),
                self_param: self_param.map(|(_, pat)| pat),
                params,
                control_flow,
                ret_ty,
                body,
                vars_defined_in_body_and_outlive,
            };

            let new_indent = IndentLevel::from_node(&insert_after);
            let old_indent = fun.body.indent_level();
            let body_contains_await = body_contains_await(&fun.body);

            builder.replace(
                target_range,
                format_replacement(ctx, &fun, old_indent, body_contains_await),
            );

            let fn_def =
                format_function(ctx, module, &fun, old_indent, new_indent, body_contains_await);
            let insert_offset = insert_after.text_range().end();
            match ctx.config.snippet_cap {
                Some(cap) => builder.insert_snippet(cap, insert_offset, fn_def),
                None => builder.insert(insert_offset, fn_def),
            }
        },
    )
}

fn external_control_flow(ctx: &AssistContext, body: &FunctionBody) -> Option<ControlFlow> {
    let mut ret_expr = None;
    let mut try_expr = None;
    let mut break_expr = None;
    let mut continue_expr = None;
    let (syntax, text_range) = match body {
        FunctionBody::Expr(expr) => (expr.syntax(), expr.syntax().text_range()),
        FunctionBody::Span { parent, text_range } => (parent.syntax(), *text_range),
    };

    let mut nested_loop = None;
    let mut nested_scope = None;

    for e in syntax.preorder() {
        let e = match e {
            WalkEvent::Enter(e) => e,
            WalkEvent::Leave(e) => {
                if nested_loop.as_ref() == Some(&e) {
                    nested_loop = None;
                }
                if nested_scope.as_ref() == Some(&e) {
                    nested_scope = None;
                }
                continue;
            }
        };
        if nested_scope.is_some() {
            continue;
        }
        if !text_range.contains_range(e.text_range()) {
            continue;
        }
        match e.kind() {
            SyntaxKind::LOOP_EXPR | SyntaxKind::WHILE_EXPR | SyntaxKind::FOR_EXPR => {
                if nested_loop.is_none() {
                    nested_loop = Some(e);
                }
            }
            SyntaxKind::FN
            | SyntaxKind::CONST
            | SyntaxKind::STATIC
            | SyntaxKind::IMPL
            | SyntaxKind::MODULE => {
                if nested_scope.is_none() {
                    nested_scope = Some(e);
                }
            }
            SyntaxKind::RETURN_EXPR => {
                ret_expr = Some(ast::ReturnExpr::cast(e).unwrap());
            }
            SyntaxKind::TRY_EXPR => {
                try_expr = Some(ast::TryExpr::cast(e).unwrap());
            }
            SyntaxKind::BREAK_EXPR if nested_loop.is_none() => {
                break_expr = Some(ast::BreakExpr::cast(e).unwrap());
            }
            SyntaxKind::CONTINUE_EXPR if nested_loop.is_none() => {
                continue_expr = Some(ast::ContinueExpr::cast(e).unwrap());
            }
            _ => {}
        }
    }

    let kind = match (try_expr, ret_expr, break_expr, continue_expr) {
        (Some(e), None, None, None) => {
            let func = e.syntax().ancestors().find_map(ast::Fn::cast)?;
            let def = ctx.sema.to_def(&func)?;
            let ret_ty = def.ret_type(ctx.db());
            let kind = try_kind_of_ty(ret_ty, ctx)?;

            Some(FlowKind::Try { kind })
        }
        (Some(_), Some(r), None, None) => match r.expr() {
            Some(expr) => {
                if let Some(kind) = expr_err_kind(&expr, ctx) {
                    Some(FlowKind::TryReturn { expr, kind })
                } else {
                    cov_mark::hit!(external_control_flow_try_and_return_non_err);
                    return None;
                }
            }
            None => return None,
        },
        (Some(_), _, _, _) => {
            cov_mark::hit!(external_control_flow_try_and_bc);
            return None;
        }
        (None, Some(r), None, None) => match r.expr() {
            Some(expr) => Some(FlowKind::ReturnValue(expr)),
            None => Some(FlowKind::Return),
        },
        (None, Some(_), _, _) => {
            cov_mark::hit!(external_control_flow_return_and_bc);
            return None;
        }
        (None, None, Some(_), Some(_)) => {
            cov_mark::hit!(external_control_flow_break_and_continue);
            return None;
        }
        (None, None, Some(b), None) => match b.expr() {
            Some(expr) => Some(FlowKind::BreakValue(expr)),
            None => Some(FlowKind::Break),
        },
        (None, None, None, Some(_)) => Some(FlowKind::Continue),
        (None, None, None, None) => None,
    };

    Some(ControlFlow { kind })
}

/// Checks is expr is `Err(_)` or `None`
fn expr_err_kind(expr: &ast::Expr, ctx: &AssistContext) -> Option<TryKind> {
    let func_name = match expr {
        ast::Expr::CallExpr(call_expr) => call_expr.expr()?,
        ast::Expr::PathExpr(_) => expr.clone(),
        _ => return None,
    };
    let text = func_name.syntax().text();

    if text == "Err" {
        Some(TryKind::Result { ty: ctx.sema.type_of_expr(expr)? })
    } else if text == "None" {
        Some(TryKind::Option)
    } else {
        None
    }
}

#[derive(Debug)]
struct Function {
    name: String,
    self_param: Option<ast::SelfParam>,
    params: Vec<Param>,
    control_flow: ControlFlow,
    ret_ty: RetType,
    body: FunctionBody,
    vars_defined_in_body_and_outlive: Vec<OutlivedLocal>,
}

#[derive(Debug)]
struct Param {
    var: Local,
    ty: hir::Type,
    has_usages_afterwards: bool,
    has_mut_inside_body: bool,
    is_copy: bool,
}

#[derive(Debug)]
struct ControlFlow {
    kind: Option<FlowKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    Value,
    MutValue,
    SharedRef,
    MutRef,
}

#[derive(Debug, Eq, PartialEq)]
enum FunType {
    Unit,
    Single(hir::Type),
    Tuple(Vec<hir::Type>),
}

impl Function {
    fn return_type(&self, ctx: &AssistContext) -> FunType {
        match &self.ret_ty {
            RetType::Expr(ty) if ty.is_unit() => FunType::Unit,
            RetType::Expr(ty) => FunType::Single(ty.clone()),
            RetType::Stmt => match self.vars_defined_in_body_and_outlive.as_slice() {
                [] => FunType::Unit,
                [var] => FunType::Single(var.local.ty(ctx.db())),
                vars => {
                    let types = vars.iter().map(|v| v.local.ty(ctx.db())).collect();
                    FunType::Tuple(types)
                }
            },
        }
    }
}

impl ParamKind {
    fn is_ref(&self) -> bool {
        matches!(self, ParamKind::SharedRef | ParamKind::MutRef)
    }
}

impl Param {
    fn kind(&self) -> ParamKind {
        match (self.has_usages_afterwards, self.has_mut_inside_body, self.is_copy) {
            (true, true, _) => ParamKind::MutRef,
            (true, false, false) => ParamKind::SharedRef,
            (false, true, _) => ParamKind::MutValue,
            (true, false, true) | (false, false, _) => ParamKind::Value,
        }
    }

    fn to_arg(&self, ctx: &AssistContext) -> ast::Expr {
        let var = path_expr_from_local(ctx, self.var);
        match self.kind() {
            ParamKind::Value | ParamKind::MutValue => var,
            ParamKind::SharedRef => make::expr_ref(var, false),
            ParamKind::MutRef => make::expr_ref(var, true),
        }
    }

    fn to_param(&self, ctx: &AssistContext, module: hir::Module) -> ast::Param {
        let var = self.var.name(ctx.db()).unwrap().to_string();
        let var_name = make::name(&var);
        let pat = match self.kind() {
            ParamKind::MutValue => make::ident_pat(false, true, var_name),
            ParamKind::Value | ParamKind::SharedRef | ParamKind::MutRef => {
                make::ext::simple_ident_pat(var_name)
            }
        };

        let ty = make_ty(&self.ty, ctx, module);
        let ty = match self.kind() {
            ParamKind::Value | ParamKind::MutValue => ty,
            ParamKind::SharedRef => make::ty_ref(ty, false),
            ParamKind::MutRef => make::ty_ref(ty, true),
        };

        make::param(pat.into(), ty)
    }
}

/// Control flow that is exported from extracted function
///
/// E.g.:
/// ```rust,no_run
/// loop {
///     $0
///     if 42 == 42 {
///         break;
///     }
///     $0
/// }
/// ```
#[derive(Debug, Clone)]
enum FlowKind {
    /// Return without value (`return;`)
    Return,
    /// Return with value (`return $expr;`)
    ReturnValue(ast::Expr),
    Try {
        kind: TryKind,
    },
    TryReturn {
        expr: ast::Expr,
        kind: TryKind,
    },
    /// Break without value (`return;`)
    Break,
    /// Break with value (`break $expr;`)
    BreakValue(ast::Expr),
    /// Continue
    Continue,
}

#[derive(Debug, Clone)]
enum TryKind {
    Option,
    Result { ty: hir::Type },
}

impl FlowKind {
    fn make_result_handler(&self, expr: Option<ast::Expr>) -> ast::Expr {
        match self {
            FlowKind::Return | FlowKind::ReturnValue(_) => make::expr_return(expr),
            FlowKind::Break | FlowKind::BreakValue(_) => make::expr_break(expr),
            FlowKind::Try { .. } | FlowKind::TryReturn { .. } => {
                stdx::never!("cannot have result handler with try");
                expr.unwrap_or_else(|| make::expr_return(None))
            }
            FlowKind::Continue => {
                stdx::always!(expr.is_none(), "continue with value is not possible");
                make::expr_continue()
            }
        }
    }

    fn expr_ty(&self, ctx: &AssistContext) -> Option<hir::Type> {
        match self {
            FlowKind::ReturnValue(expr)
            | FlowKind::BreakValue(expr)
            | FlowKind::TryReturn { expr, .. } => ctx.sema.type_of_expr(expr),
            FlowKind::Try { .. } => {
                stdx::never!("try does not have defined expr_ty");
                None
            }
            FlowKind::Return | FlowKind::Break | FlowKind::Continue => None,
        }
    }
}

fn try_kind_of_ty(ty: hir::Type, ctx: &AssistContext) -> Option<TryKind> {
    if ty.is_unknown() {
        // We favour Result for `expr?`
        return Some(TryKind::Result { ty });
    }
    let adt = ty.as_adt()?;
    let name = adt.name(ctx.db());
    // FIXME: use lang items to determine if it is std type or user defined
    //        E.g. if user happens to define type named `Option`, we would have false positive
    match name.to_string().as_str() {
        "Option" => Some(TryKind::Option),
        "Result" => Some(TryKind::Result { ty }),
        _ => None,
    }
}

#[derive(Debug)]
enum RetType {
    Expr(hir::Type),
    Stmt,
}

impl RetType {
    fn is_unit(&self) -> bool {
        match self {
            RetType::Expr(ty) => ty.is_unit(),
            RetType::Stmt => true,
        }
    }
}

/// Semantically same as `ast::Expr`, but preserves identity when using only part of the Block
#[derive(Debug)]
enum FunctionBody {
    Expr(ast::Expr),
    Span { parent: ast::BlockExpr, text_range: TextRange },
}

impl FunctionBody {
    fn from_whole_node(node: SyntaxNode) -> Option<Self> {
        match node.kind() {
            PATH_EXPR => None,
            BREAK_EXPR => ast::BreakExpr::cast(node).and_then(|e| e.expr()).map(Self::Expr),
            RETURN_EXPR => ast::ReturnExpr::cast(node).and_then(|e| e.expr()).map(Self::Expr),
            BLOCK_EXPR => ast::BlockExpr::cast(node)
                .filter(|it| it.is_standalone())
                .map(Into::into)
                .map(Self::Expr),
            _ => ast::Expr::cast(node).map(Self::Expr),
        }
    }

    fn from_range(node: SyntaxNode, text_range: TextRange) -> Option<FunctionBody> {
        let block = ast::BlockExpr::cast(node)?;
        Some(Self::Span { parent: block, text_range })
    }

    fn indent_level(&self) -> IndentLevel {
        match &self {
            FunctionBody::Expr(expr) => IndentLevel::from_node(expr.syntax()),
            FunctionBody::Span { parent, .. } => IndentLevel::from_node(parent.syntax()) + 1,
        }
    }

    fn tail_expr(&self) -> Option<ast::Expr> {
        match &self {
            FunctionBody::Expr(expr) => Some(expr.clone()),
            FunctionBody::Span { parent, text_range } => {
                let tail_expr = parent.tail_expr()?;
                if text_range.contains_range(tail_expr.syntax().text_range()) {
                    Some(tail_expr)
                } else {
                    None
                }
            }
        }
    }

    fn descendants(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        match self {
            FunctionBody::Expr(expr) => Either::Right(expr.syntax().descendants()),
            FunctionBody::Span { parent, text_range } => Either::Left(
                parent
                    .syntax()
                    .descendants()
                    .filter(move |it| text_range.contains_range(it.text_range())),
            ),
        }
    }

    fn text_range(&self) -> TextRange {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().text_range(),
            FunctionBody::Span { parent: _, text_range } => *text_range,
        }
    }

    fn contains_range(&self, range: TextRange) -> bool {
        self.text_range().contains_range(range)
    }

    fn preceedes_range(&self, range: TextRange) -> bool {
        self.text_range().end() <= range.start()
    }

    fn contains_node(&self, node: &SyntaxNode) -> bool {
        self.contains_range(node.text_range())
    }
}

impl HasTokenAtOffset for FunctionBody {
    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().token_at_offset(offset),
            FunctionBody::Span { parent, text_range } => {
                match parent.syntax().token_at_offset(offset) {
                    TokenAtOffset::None => TokenAtOffset::None,
                    TokenAtOffset::Single(t) => {
                        if text_range.contains_range(t.text_range()) {
                            TokenAtOffset::Single(t)
                        } else {
                            TokenAtOffset::None
                        }
                    }
                    TokenAtOffset::Between(a, b) => {
                        match (
                            text_range.contains_range(a.text_range()),
                            text_range.contains_range(b.text_range()),
                        ) {
                            (true, true) => TokenAtOffset::Between(a, b),
                            (true, false) => TokenAtOffset::Single(a),
                            (false, true) => TokenAtOffset::Single(b),
                            (false, false) => TokenAtOffset::None,
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct OutlivedLocal {
    local: Local,
    mut_usage_outside_body: bool,
}

/// Try to guess what user wants to extract
///
/// We have basically have two cases:
/// * We want whole node, like `loop {}`, `2 + 2`, `{ let n = 1; }` exprs.
///   Then we can use `ast::Expr`
/// * We want a few statements for a block. E.g.
///   ```rust,no_run
///   fn foo() -> i32 {
///     let m = 1;
///     $0
///     let n = 2;
///     let k = 3;
///     k + n
///     $0
///   }
///   ```
///
fn extraction_target(node: &SyntaxNode, selection_range: TextRange) -> Option<FunctionBody> {
    // we have selected exactly the expr node
    // wrap it before anything else
    if node.text_range() == selection_range {
        let body = FunctionBody::from_whole_node(node.clone());
        if body.is_some() {
            return body;
        }
    }

    // we have selected a few statements in a block
    // so covering_element returns the whole block
    if node.kind() == BLOCK_EXPR {
        // Extract the full statements.
        let statements_range = node
            .children()
            .filter(|c| selection_range.intersect(c.text_range()).is_some())
            .fold(selection_range, |acc, c| acc.cover(c.text_range()));
        let body = FunctionBody::from_range(node.clone(), statements_range);
        if body.is_some() {
            return body;
        }
    }

    // we have selected single statement
    // `from_whole_node` failed because (let) statement is not and expression
    // so we try to expand covering_element to parent and repeat the previous
    if let Some(parent) = node.parent() {
        if parent.kind() == BLOCK_EXPR {
            // Extract the full statement.
            let body = FunctionBody::from_range(parent, node.text_range());
            if body.is_some() {
                return body;
            }
        }
    }

    // select the closest containing expr (both ifs are used)
    std::iter::once(node.clone()).chain(node.ancestors()).find_map(FunctionBody::from_whole_node)
}

/// list local variables that are referenced in `body`
fn vars_used_in_body(ctx: &AssistContext, body: &FunctionBody) -> Vec<Local> {
    // FIXME: currently usages inside macros are not found
    body.descendants()
        .filter_map(ast::NameRef::cast)
        .filter_map(|name_ref| NameRefClass::classify(&ctx.sema, &name_ref))
        .map(|name_kind| name_kind.referenced(ctx.db()))
        .filter_map(|definition| match definition {
            Definition::Local(local) => Some(local),
            _ => None,
        })
        .unique()
        .collect()
}

fn body_contains_await(body: &FunctionBody) -> bool {
    body.descendants().any(|d| matches!(d.kind(), SyntaxKind::AWAIT_EXPR))
}

/// find `self` param, that was not defined inside `body`
///
/// It should skip `self` params from impls inside `body`
fn self_param_from_usages(
    ctx: &AssistContext,
    body: &FunctionBody,
    vars_used_in_body: &[Local],
) -> Option<(Local, ast::SelfParam)> {
    let mut iter = vars_used_in_body
        .iter()
        .filter(|var| var.is_self(ctx.db()))
        .map(|var| (var, var.source(ctx.db())))
        .filter(|(_, src)| is_defined_before(ctx, body, src))
        .filter_map(|(&node, src)| match src.value {
            Either::Right(it) => Some((node, it)),
            Either::Left(_) => {
                stdx::never!(false, "Local::is_self returned true, but source is IdentPat");
                None
            }
        });

    let self_param = iter.next();
    stdx::always!(
        iter.next().is_none(),
        "body references two different self params, both defined outside"
    );

    self_param
}

/// find variables that should be extracted as params
///
/// Computes additional info that affects param type and mutability
fn extracted_function_params(
    ctx: &AssistContext,
    body: &FunctionBody,
    vars_used_in_body: &[Local],
) -> Vec<Param> {
    vars_used_in_body
        .iter()
        .filter(|var| !var.is_self(ctx.db()))
        .map(|node| (node, node.source(ctx.db())))
        .filter(|(_, src)| is_defined_before(ctx, body, src))
        .filter_map(|(&node, src)| {
            if src.value.is_left() {
                Some(node)
            } else {
                stdx::never!(false, "Local::is_self returned false, but source is SelfParam");
                None
            }
        })
        .map(|var| {
            let usages = LocalUsages::find(ctx, var);
            let ty = var.ty(ctx.db());
            let is_copy = ty.is_copy(ctx.db());
            Param {
                var,
                ty,
                has_usages_afterwards: has_usages_after_body(&usages, body),
                has_mut_inside_body: has_exclusive_usages(ctx, &usages, body),
                is_copy,
            }
        })
        .collect()
}

fn has_usages_after_body(usages: &LocalUsages, body: &FunctionBody) -> bool {
    usages.iter().any(|reference| body.preceedes_range(reference.range))
}

/// checks if relevant var is used with `&mut` access inside body
fn has_exclusive_usages(ctx: &AssistContext, usages: &LocalUsages, body: &FunctionBody) -> bool {
    usages
        .iter()
        .filter(|reference| body.contains_range(reference.range))
        .any(|reference| reference_is_exclusive(reference, body, ctx))
}

/// checks if this reference requires `&mut` access inside node
fn reference_is_exclusive(
    reference: &FileReference,
    node: &dyn HasTokenAtOffset,
    ctx: &AssistContext,
) -> bool {
    // we directly modify variable with set: `n = 0`, `n += 1`
    if reference.access == Some(ReferenceAccess::Write) {
        return true;
    }

    // we take `&mut` reference to variable: `&mut v`
    let path = match path_element_of_reference(node, reference) {
        Some(path) => path,
        None => return false,
    };

    expr_require_exclusive_access(ctx, &path).unwrap_or(false)
}

/// checks if this expr requires `&mut` access, recurses on field access
fn expr_require_exclusive_access(ctx: &AssistContext, expr: &ast::Expr) -> Option<bool> {
    match expr {
        ast::Expr::MacroCall(_) => {
            // FIXME: expand macro and check output for mutable usages of the variable?
            return None;
        }
        _ => (),
    }

    let parent = expr.syntax().parent()?;

    if let Some(bin_expr) = ast::BinExpr::cast(parent.clone()) {
        if bin_expr.op_kind()?.is_assignment() {
            return Some(bin_expr.lhs()?.syntax() == expr.syntax());
        }
        return Some(false);
    }

    if let Some(ref_expr) = ast::RefExpr::cast(parent.clone()) {
        return Some(ref_expr.mut_token().is_some());
    }

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        let func = ctx.sema.resolve_method_call(&method_call)?;
        let self_param = func.self_param(ctx.db())?;
        let access = self_param.access(ctx.db());

        return Some(matches!(access, hir::Access::Exclusive));
    }

    if let Some(field) = ast::FieldExpr::cast(parent) {
        return expr_require_exclusive_access(ctx, &field.into());
    }

    Some(false)
}

/// Container of local variable usages
///
/// Semanticall same as `UsageSearchResult`, but provides more convenient interface
struct LocalUsages(ide_db::search::UsageSearchResult);

impl LocalUsages {
    fn find(ctx: &AssistContext, var: Local) -> Self {
        Self(
            Definition::Local(var)
                .usages(&ctx.sema)
                .in_scope(SearchScope::single_file(ctx.frange.file_id))
                .all(),
        )
    }

    fn iter(&self) -> impl Iterator<Item = &FileReference> + '_ {
        self.0.iter().flat_map(|(_, rs)| rs.iter())
    }
}

trait HasTokenAtOffset {
    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken>;
}

impl HasTokenAtOffset for SyntaxNode {
    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        SyntaxNode::token_at_offset(self, offset)
    }
}

/// find relevant `ast::Expr` for reference
///
/// # Preconditions
///
/// `node` must cover `reference`, that is `node.text_range().contains_range(reference.range)`
fn path_element_of_reference(
    node: &dyn HasTokenAtOffset,
    reference: &FileReference,
) -> Option<ast::Expr> {
    let token = node.token_at_offset(reference.range.start()).right_biased().or_else(|| {
        stdx::never!(false, "cannot find token at variable usage: {:?}", reference);
        None
    })?;
    let path = token.ancestors().find_map(ast::Expr::cast).or_else(|| {
        stdx::never!(false, "cannot find path parent of variable usage: {:?}", token);
        None
    })?;
    stdx::always!(
        matches!(path, ast::Expr::PathExpr(_) | ast::Expr::MacroCall(_)),
        "unexpected expression type for variable usage: {:?}",
        path
    );
    Some(path)
}

/// list local variables defined inside `body`
fn vars_defined_in_body(body: &FunctionBody, ctx: &AssistContext) -> Vec<Local> {
    // FIXME: this doesn't work well with macros
    //        see https://github.com/rust-analyzer/rust-analyzer/pull/7535#discussion_r570048550
    body.descendants()
        .filter_map(ast::IdentPat::cast)
        .filter_map(|let_stmt| ctx.sema.to_def(&let_stmt))
        .unique()
        .collect()
}

/// list local variables defined inside `body` that should be returned from extracted function
fn vars_defined_in_body_and_outlive(
    ctx: &AssistContext,
    body: &FunctionBody,
    parent: &SyntaxNode,
) -> Vec<OutlivedLocal> {
    let vars_defined_in_body = vars_defined_in_body(body, ctx);
    vars_defined_in_body
        .into_iter()
        .filter_map(|var| var_outlives_body(ctx, body, var, parent))
        .collect()
}

/// checks if the relevant local was defined before(outside of) body
fn is_defined_before(
    ctx: &AssistContext,
    body: &FunctionBody,
    src: &hir::InFile<Either<ast::IdentPat, ast::SelfParam>>,
) -> bool {
    src.file_id.original_file(ctx.db()) == ctx.frange.file_id
        && !body.contains_node(either_syntax(&src.value))
}

fn either_syntax(value: &Either<ast::IdentPat, ast::SelfParam>) -> &SyntaxNode {
    match value {
        Either::Left(pat) => pat.syntax(),
        Either::Right(it) => it.syntax(),
    }
}

/// returns usage details if local variable is used after(outside of) body
fn var_outlives_body(
    ctx: &AssistContext,
    body: &FunctionBody,
    var: Local,
    parent: &SyntaxNode,
) -> Option<OutlivedLocal> {
    let usages = LocalUsages::find(ctx, var);
    let has_usages = usages.iter().any(|reference| body.preceedes_range(reference.range));
    if !has_usages {
        return None;
    }
    let has_mut_usages = usages
        .iter()
        .filter(|reference| body.preceedes_range(reference.range))
        .any(|reference| reference_is_exclusive(reference, parent, ctx));
    Some(OutlivedLocal { local: var, mut_usage_outside_body: has_mut_usages })
}

fn body_return_ty(ctx: &AssistContext, body: &FunctionBody) -> Option<RetType> {
    match body.tail_expr() {
        Some(expr) => {
            let ty = ctx.sema.type_of_expr(&expr)?;
            Some(RetType::Expr(ty))
        }
        None => Some(RetType::Stmt),
    }
}
/// Where to put extracted function definition
#[derive(Debug)]
enum Anchor {
    /// Extract free function and put right after current top-level function
    Freestanding,
    /// Extract method and put right after current function in the impl-block
    Method,
}

/// find where to put extracted function definition
///
/// Function should be put right after returned node
fn scope_for_fn_insertion(body: &FunctionBody, anchor: Anchor) -> Option<SyntaxNode> {
    match body {
        FunctionBody::Expr(e) => scope_for_fn_insertion_node(e.syntax(), anchor),
        FunctionBody::Span { parent, .. } => scope_for_fn_insertion_node(parent.syntax(), anchor),
    }
}

fn scope_for_fn_insertion_node(node: &SyntaxNode, anchor: Anchor) -> Option<SyntaxNode> {
    let mut ancestors = node.ancestors().peekable();
    let mut last_ancestor = None;
    while let Some(next_ancestor) = ancestors.next() {
        match next_ancestor.kind() {
            SyntaxKind::SOURCE_FILE => break,
            SyntaxKind::ITEM_LIST => {
                if !matches!(anchor, Anchor::Freestanding) {
                    continue;
                }
                if ancestors.peek().map(SyntaxNode::kind) == Some(SyntaxKind::MODULE) {
                    break;
                }
            }
            SyntaxKind::ASSOC_ITEM_LIST => {
                if !matches!(anchor, Anchor::Method) {
                    continue;
                }
                if ancestors.peek().map(SyntaxNode::kind) == Some(SyntaxKind::IMPL) {
                    break;
                }
            }
            _ => {}
        }
        last_ancestor = Some(next_ancestor);
    }
    last_ancestor
}

fn format_replacement(
    ctx: &AssistContext,
    fun: &Function,
    indent: IndentLevel,
    body_contains_await: bool,
) -> String {
    let ret_ty = fun.return_type(ctx);

    let args = fun.params.iter().map(|param| param.to_arg(ctx));
    let args = make::arg_list(args);
    let call_expr = if fun.self_param.is_some() {
        let self_arg = make::expr_path(make::ext::ident_path("self"));
        make::expr_method_call(self_arg, &fun.name, args)
    } else {
        let func = make::expr_path(make::ext::ident_path(&fun.name));
        make::expr_call(func, args)
    };

    let handler = FlowHandler::from_ret_ty(fun, &ret_ty);

    let expr = handler.make_call_expr(call_expr).indent(indent);

    let mut buf = String::new();
    match fun.vars_defined_in_body_and_outlive.as_slice() {
        [] => {}
        [var] => {
            format_to!(buf, "let {}{} = ", mut_modifier(var), var.local.name(ctx.db()).unwrap())
        }
        [v0, vs @ ..] => {
            buf.push_str("let (");
            format_to!(buf, "{}{}", mut_modifier(v0), v0.local.name(ctx.db()).unwrap());
            for var in vs {
                format_to!(buf, ", {}{}", mut_modifier(var), var.local.name(ctx.db()).unwrap());
            }
            buf.push_str(") = ");
        }
    }
    fn mut_modifier(var: &OutlivedLocal) -> &'static str {
        if var.mut_usage_outside_body {
            "mut "
        } else {
            ""
        }
    }
    format_to!(buf, "{}", expr);
    if body_contains_await {
        buf.push_str(".await");
    }
    if fun.ret_ty.is_unit()
        && (!fun.vars_defined_in_body_and_outlive.is_empty() || !expr.is_block_like())
    {
        buf.push(';');
    }
    buf
}

enum FlowHandler {
    None,
    Try { kind: TryKind },
    If { action: FlowKind },
    IfOption { action: FlowKind },
    MatchOption { none: FlowKind },
    MatchResult { err: FlowKind },
}

impl FlowHandler {
    fn from_ret_ty(fun: &Function, ret_ty: &FunType) -> FlowHandler {
        match &fun.control_flow.kind {
            None => FlowHandler::None,
            Some(flow_kind) => {
                let action = flow_kind.clone();
                if *ret_ty == FunType::Unit {
                    match flow_kind {
                        FlowKind::Return | FlowKind::Break | FlowKind::Continue => {
                            FlowHandler::If { action }
                        }
                        FlowKind::ReturnValue(_) | FlowKind::BreakValue(_) => {
                            FlowHandler::IfOption { action }
                        }
                        FlowKind::Try { kind } | FlowKind::TryReturn { kind, .. } => {
                            FlowHandler::Try { kind: kind.clone() }
                        }
                    }
                } else {
                    match flow_kind {
                        FlowKind::Return | FlowKind::Break | FlowKind::Continue => {
                            FlowHandler::MatchOption { none: action }
                        }
                        FlowKind::ReturnValue(_) | FlowKind::BreakValue(_) => {
                            FlowHandler::MatchResult { err: action }
                        }
                        FlowKind::Try { kind } | FlowKind::TryReturn { kind, .. } => {
                            FlowHandler::Try { kind: kind.clone() }
                        }
                    }
                }
            }
        }
    }

    fn make_call_expr(&self, call_expr: ast::Expr) -> ast::Expr {
        match self {
            FlowHandler::None => call_expr,
            FlowHandler::Try { kind: _ } => make::expr_try(call_expr),
            FlowHandler::If { action } => {
                let action = action.make_result_handler(None);
                let stmt = make::expr_stmt(action);
                let block = make::block_expr(iter::once(stmt.into()), None);
                let condition = make::condition(call_expr, None);
                make::expr_if(condition, block, None)
            }
            FlowHandler::IfOption { action } => {
                let path = make::ext::ident_path("Some");
                let value_pat = make::ext::simple_ident_pat(make::name("value"));
                let pattern = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                let cond = make::condition(call_expr, Some(pattern.into()));
                let value = make::expr_path(make::ext::ident_path("value"));
                let action_expr = action.make_result_handler(Some(value));
                let action_stmt = make::expr_stmt(action_expr);
                let then = make::block_expr(iter::once(action_stmt.into()), None);
                make::expr_if(cond, then, None)
            }
            FlowHandler::MatchOption { none } => {
                let some_name = "value";

                let some_arm = {
                    let path = make::ext::ident_path("Some");
                    let value_pat = make::ext::simple_ident_pat(make::name(some_name));
                    let pat = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                    let value = make::expr_path(make::ext::ident_path(some_name));
                    make::match_arm(iter::once(pat.into()), None, value)
                };
                let none_arm = {
                    let path = make::ext::ident_path("None");
                    let pat = make::path_pat(path);
                    make::match_arm(iter::once(pat), None, none.make_result_handler(None))
                };
                let arms = make::match_arm_list(vec![some_arm, none_arm]);
                make::expr_match(call_expr, arms)
            }
            FlowHandler::MatchResult { err } => {
                let ok_name = "value";
                let err_name = "value";

                let ok_arm = {
                    let path = make::ext::ident_path("Ok");
                    let value_pat = make::ext::simple_ident_pat(make::name(ok_name));
                    let pat = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                    let value = make::expr_path(make::ext::ident_path(ok_name));
                    make::match_arm(iter::once(pat.into()), None, value)
                };
                let err_arm = {
                    let path = make::ext::ident_path("Err");
                    let value_pat = make::ext::simple_ident_pat(make::name(err_name));
                    let pat = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                    let value = make::expr_path(make::ext::ident_path(err_name));
                    make::match_arm(
                        iter::once(pat.into()),
                        None,
                        err.make_result_handler(Some(value)),
                    )
                };
                let arms = make::match_arm_list(vec![ok_arm, err_arm]);
                make::expr_match(call_expr, arms)
            }
        }
    }
}

fn path_expr_from_local(ctx: &AssistContext, var: Local) -> ast::Expr {
    let name = var.name(ctx.db()).unwrap().to_string();
    make::expr_path(make::ext::ident_path(&name))
}

fn format_function(
    ctx: &AssistContext,
    module: hir::Module,
    fun: &Function,
    old_indent: IndentLevel,
    new_indent: IndentLevel,
    body_contains_await: bool,
) -> String {
    let mut fn_def = String::new();
    let params = make_param_list(ctx, module, fun);
    let ret_ty = make_ret_ty(ctx, module, fun);
    let body = make_body(ctx, old_indent, new_indent, fun);
    let async_kw = if body_contains_await { "async " } else { "" };
    match ctx.config.snippet_cap {
        Some(_) => format_to!(fn_def, "\n\n{}{}fn $0{}{}", new_indent, async_kw, fun.name, params),
        None => format_to!(fn_def, "\n\n{}{}fn {}{}", new_indent, async_kw, fun.name, params),
    }
    if let Some(ret_ty) = ret_ty {
        format_to!(fn_def, " {}", ret_ty);
    }
    format_to!(fn_def, " {}", body);

    fn_def
}

fn make_param_list(ctx: &AssistContext, module: hir::Module, fun: &Function) -> ast::ParamList {
    let self_param = fun.self_param.clone();
    let params = fun.params.iter().map(|param| param.to_param(ctx, module));
    make::param_list(self_param, params)
}

impl FunType {
    fn make_ty(&self, ctx: &AssistContext, module: hir::Module) -> ast::Type {
        match self {
            FunType::Unit => make::ty_unit(),
            FunType::Single(ty) => make_ty(ty, ctx, module),
            FunType::Tuple(types) => match types.as_slice() {
                [] => {
                    stdx::never!("tuple type with 0 elements");
                    make::ty_unit()
                }
                [ty] => {
                    stdx::never!("tuple type with 1 element");
                    make_ty(ty, ctx, module)
                }
                types => {
                    let types = types.iter().map(|ty| make_ty(ty, ctx, module));
                    make::ty_tuple(types)
                }
            },
        }
    }
}

fn make_ret_ty(ctx: &AssistContext, module: hir::Module, fun: &Function) -> Option<ast::RetType> {
    let fun_ty = fun.return_type(ctx);
    let handler = FlowHandler::from_ret_ty(fun, &fun_ty);
    let ret_ty = match &handler {
        FlowHandler::None => {
            if matches!(fun_ty, FunType::Unit) {
                return None;
            }
            fun_ty.make_ty(ctx, module)
        }
        FlowHandler::Try { kind: TryKind::Option } => {
            make::ext::ty_option(fun_ty.make_ty(ctx, module))
        }
        FlowHandler::Try { kind: TryKind::Result { ty: parent_ret_ty } } => {
            let handler_ty = parent_ret_ty
                .type_arguments()
                .nth(1)
                .map(|ty| make_ty(&ty, ctx, module))
                .unwrap_or_else(make::ty_unit);
            make::ext::ty_result(fun_ty.make_ty(ctx, module), handler_ty)
        }
        FlowHandler::If { .. } => make::ext::ty_bool(),
        FlowHandler::IfOption { action } => {
            let handler_ty = action
                .expr_ty(ctx)
                .map(|ty| make_ty(&ty, ctx, module))
                .unwrap_or_else(make::ty_unit);
            make::ext::ty_option(handler_ty)
        }
        FlowHandler::MatchOption { .. } => make::ext::ty_option(fun_ty.make_ty(ctx, module)),
        FlowHandler::MatchResult { err } => {
            let handler_ty =
                err.expr_ty(ctx).map(|ty| make_ty(&ty, ctx, module)).unwrap_or_else(make::ty_unit);
            make::ext::ty_result(fun_ty.make_ty(ctx, module), handler_ty)
        }
    };
    Some(make::ret_type(ret_ty))
}

fn make_body(
    ctx: &AssistContext,
    old_indent: IndentLevel,
    new_indent: IndentLevel,
    fun: &Function,
) -> ast::BlockExpr {
    let ret_ty = fun.return_type(ctx);
    let handler = FlowHandler::from_ret_ty(fun, &ret_ty);
    let block = match &fun.body {
        FunctionBody::Expr(expr) => {
            let expr = rewrite_body_segment(ctx, &fun.params, &handler, expr.syntax());
            let expr = ast::Expr::cast(expr).unwrap();
            match expr {
                ast::Expr::BlockExpr(block) => {
                    // If the extracted expression is itself a block, there is no need to wrap it inside another block.
                    let block = block.dedent(old_indent);
                    // Recreate the block for formatting consistency with other extracted functions.
                    make::block_expr(block.statements(), block.tail_expr())
                }
                _ => {
                    let expr = expr.dedent(old_indent).indent(IndentLevel(1));

                    make::block_expr(Vec::new(), Some(expr))
                }
            }
        }
        FunctionBody::Span { parent, text_range } => {
            let mut elements: Vec<_> = parent
                .syntax()
                .children()
                .filter(|it| text_range.contains_range(it.text_range()))
                .map(|it| rewrite_body_segment(ctx, &fun.params, &handler, &it))
                .collect();

            let mut tail_expr = match elements.pop() {
                Some(node) => ast::Expr::cast(node.clone()).or_else(|| {
                    elements.push(node);
                    None
                }),
                None => None,
            };

            if tail_expr.is_none() {
                match fun.vars_defined_in_body_and_outlive.as_slice() {
                    [] => {}
                    [var] => {
                        tail_expr = Some(path_expr_from_local(ctx, var.local));
                    }
                    vars => {
                        let exprs = vars.iter().map(|var| path_expr_from_local(ctx, var.local));
                        let expr = make::expr_tuple(exprs);
                        tail_expr = Some(expr);
                    }
                }
            }

            let elements = elements.into_iter().filter_map(|node| match ast::Stmt::cast(node) {
                Some(stmt) => Some(stmt),
                None => {
                    stdx::never!("block contains non-statement");
                    None
                }
            });

            let body_indent = IndentLevel(1);
            let elements = elements.map(|stmt| stmt.dedent(old_indent).indent(body_indent));
            let tail_expr = tail_expr.map(|expr| expr.dedent(old_indent).indent(body_indent));

            make::block_expr(elements, tail_expr)
        }
    };

    let block = match &handler {
        FlowHandler::None => block,
        FlowHandler::Try { kind } => {
            let block = with_default_tail_expr(block, make::expr_unit());
            map_tail_expr(block, |tail_expr| {
                let constructor = match kind {
                    TryKind::Option => "Some",
                    TryKind::Result { .. } => "Ok",
                };
                let func = make::expr_path(make::ext::ident_path(constructor));
                let args = make::arg_list(iter::once(tail_expr));
                make::expr_call(func, args)
            })
        }
        FlowHandler::If { .. } => {
            let lit_false = make::expr_literal("false");
            with_tail_expr(block, lit_false.into())
        }
        FlowHandler::IfOption { .. } => {
            let none = make::expr_path(make::ext::ident_path("None"));
            with_tail_expr(block, none)
        }
        FlowHandler::MatchOption { .. } => map_tail_expr(block, |tail_expr| {
            let some = make::expr_path(make::ext::ident_path("Some"));
            let args = make::arg_list(iter::once(tail_expr));
            make::expr_call(some, args)
        }),
        FlowHandler::MatchResult { .. } => map_tail_expr(block, |tail_expr| {
            let ok = make::expr_path(make::ext::ident_path("Ok"));
            let args = make::arg_list(iter::once(tail_expr));
            make::expr_call(ok, args)
        }),
    };

    block.indent(new_indent)
}

fn map_tail_expr(block: ast::BlockExpr, f: impl FnOnce(ast::Expr) -> ast::Expr) -> ast::BlockExpr {
    let tail_expr = match block.tail_expr() {
        Some(tail_expr) => tail_expr,
        None => return block,
    };
    make::block_expr(block.statements(), Some(f(tail_expr)))
}

fn with_default_tail_expr(block: ast::BlockExpr, tail_expr: ast::Expr) -> ast::BlockExpr {
    match block.tail_expr() {
        Some(_) => block,
        None => make::block_expr(block.statements(), Some(tail_expr)),
    }
}

fn with_tail_expr(block: ast::BlockExpr, tail_expr: ast::Expr) -> ast::BlockExpr {
    let stmt_tail = block.tail_expr().map(|expr| make::expr_stmt(expr).into());
    let stmts = block.statements().chain(stmt_tail);
    make::block_expr(stmts, Some(tail_expr))
}

fn format_type(ty: &hir::Type, ctx: &AssistContext, module: hir::Module) -> String {
    ty.display_source_code(ctx.db(), module.into()).ok().unwrap_or_else(|| "()".to_string())
}

fn make_ty(ty: &hir::Type, ctx: &AssistContext, module: hir::Module) -> ast::Type {
    let ty_str = format_type(ty, ctx, module);
    make::ty(&ty_str)
}

fn rewrite_body_segment(
    ctx: &AssistContext,
    params: &[Param],
    handler: &FlowHandler,
    syntax: &SyntaxNode,
) -> SyntaxNode {
    let syntax = fix_param_usages(ctx, params, syntax);
    update_external_control_flow(handler, &syntax);
    syntax
}

/// change all usages to account for added `&`/`&mut` for some params
fn fix_param_usages(ctx: &AssistContext, params: &[Param], syntax: &SyntaxNode) -> SyntaxNode {
    let mut usages_for_param: Vec<(&Param, Vec<ast::Expr>)> = Vec::new();

    let tm = TreeMutator::new(syntax);

    for param in params {
        if !param.kind().is_ref() {
            continue;
        }

        let usages = LocalUsages::find(ctx, param.var);
        let usages = usages
            .iter()
            .filter(|reference| syntax.text_range().contains_range(reference.range))
            .filter_map(|reference| path_element_of_reference(syntax, reference))
            .map(|expr| tm.make_mut(&expr));

        usages_for_param.push((param, usages.collect()));
    }

    let res = tm.make_syntax_mut(syntax);

    for (param, usages) in usages_for_param {
        for usage in usages {
            match usage.syntax().ancestors().skip(1).find_map(ast::Expr::cast) {
                Some(ast::Expr::MethodCallExpr(_) | ast::Expr::FieldExpr(_)) => {
                    // do nothing
                }
                Some(ast::Expr::RefExpr(node))
                    if param.kind() == ParamKind::MutRef && node.mut_token().is_some() =>
                {
                    ted::replace(node.syntax(), node.expr().unwrap().syntax());
                }
                Some(ast::Expr::RefExpr(node))
                    if param.kind() == ParamKind::SharedRef && node.mut_token().is_none() =>
                {
                    ted::replace(node.syntax(), node.expr().unwrap().syntax());
                }
                Some(_) | None => {
                    let p = &make::expr_prefix(T![*], usage.clone()).clone_for_update();
                    ted::replace(usage.syntax(), p.syntax())
                }
            }
        }
    }

    res
}

fn update_external_control_flow(handler: &FlowHandler, syntax: &SyntaxNode) {
    let mut nested_loop = None;
    let mut nested_scope = None;
    for event in syntax.preorder() {
        match event {
            WalkEvent::Enter(e) => match e.kind() {
                SyntaxKind::LOOP_EXPR | SyntaxKind::WHILE_EXPR | SyntaxKind::FOR_EXPR => {
                    if nested_loop.is_none() {
                        nested_loop = Some(e.clone());
                    }
                }
                SyntaxKind::FN
                | SyntaxKind::CONST
                | SyntaxKind::STATIC
                | SyntaxKind::IMPL
                | SyntaxKind::MODULE => {
                    if nested_scope.is_none() {
                        nested_scope = Some(e.clone());
                    }
                }
                _ => {}
            },
            WalkEvent::Leave(e) => {
                if nested_scope.is_none() {
                    if let Some(expr) = ast::Expr::cast(e.clone()) {
                        match expr {
                            ast::Expr::ReturnExpr(return_expr) if nested_scope.is_none() => {
                                let expr = return_expr.expr();
                                if let Some(replacement) = make_rewritten_flow(handler, expr) {
                                    ted::replace(return_expr.syntax(), replacement.syntax())
                                }
                            }
                            ast::Expr::BreakExpr(break_expr) if nested_loop.is_none() => {
                                let expr = break_expr.expr();
                                if let Some(replacement) = make_rewritten_flow(handler, expr) {
                                    ted::replace(break_expr.syntax(), replacement.syntax())
                                }
                            }
                            ast::Expr::ContinueExpr(continue_expr) if nested_loop.is_none() => {
                                if let Some(replacement) = make_rewritten_flow(handler, None) {
                                    ted::replace(continue_expr.syntax(), replacement.syntax())
                                }
                            }
                            _ => {
                                // do nothing
                            }
                        }
                    }
                }

                if nested_loop.as_ref() == Some(&e) {
                    nested_loop = None;
                }
                if nested_scope.as_ref() == Some(&e) {
                    nested_scope = None;
                }
            }
        };
    }
}

fn make_rewritten_flow(handler: &FlowHandler, arg_expr: Option<ast::Expr>) -> Option<ast::Expr> {
    let value = match handler {
        FlowHandler::None | FlowHandler::Try { .. } => return None,
        FlowHandler::If { .. } => make::expr_literal("true").into(),
        FlowHandler::IfOption { .. } => {
            let expr = arg_expr.unwrap_or_else(|| make::expr_tuple(Vec::new()));
            let args = make::arg_list(iter::once(expr));
            make::expr_call(make::expr_path(make::ext::ident_path("Some")), args)
        }
        FlowHandler::MatchOption { .. } => make::expr_path(make::ext::ident_path("None")),
        FlowHandler::MatchResult { .. } => {
            let expr = arg_expr.unwrap_or_else(|| make::expr_tuple(Vec::new()));
            let args = make::arg_list(iter::once(expr));
            make::expr_call(make::expr_path(make::ext::ident_path("Err")), args)
        }
    };
    Some(make::expr_return(Some(value)).clone_for_update())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn no_args_from_binary_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    foo($01 + 1$0);
}
"#,
            r#"
fn foo() {
    foo(fun_name());
}

fn $0fun_name() -> i32 {
    1 + 1
}
"#,
        );
    }

    #[test]
    fn no_args_from_binary_expr_in_module() {
        check_assist(
            extract_function,
            r#"
mod bar {
    fn foo() {
        foo($01 + 1$0);
    }
}
"#,
            r#"
mod bar {
    fn foo() {
        foo(fun_name());
    }

    fn $0fun_name() -> i32 {
        1 + 1
    }
}
"#,
        );
    }

    #[test]
    fn no_args_from_binary_expr_indented() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0{ 1 + 1 }$0;
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() -> i32 {
    1 + 1
}
"#,
        );
    }

    #[test]
    fn no_args_from_stmt_with_last_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    let k = 1;
    $0let m = 1;
    m + 1$0
}
"#,
            r#"
fn foo() -> i32 {
    let k = 1;
    fun_name()
}

fn $0fun_name() -> i32 {
    let m = 1;
    m + 1
}
"#,
        );
    }

    #[test]
    fn no_args_from_stmt_unit() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let k = 3;
    $0let m = 1;
    let n = m + 1;$0
    let g = 5;
}
"#,
            r#"
fn foo() {
    let k = 3;
    fun_name();
    let g = 5;
}

fn $0fun_name() {
    let m = 1;
    let n = m + 1;
}
"#,
        );
    }

    #[test]
    fn no_args_if() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0if true { }$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    if true { }
}
"#,
        );
    }

    #[test]
    fn no_args_if_else() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    $0if true { 1 } else { 2 }$0
}
"#,
            r#"
fn foo() -> i32 {
    fun_name()
}

fn $0fun_name() -> i32 {
    if true { 1 } else { 2 }
}
"#,
        );
    }

    #[test]
    fn no_args_if_let_else() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    $0if let true = false { 1 } else { 2 }$0
}
"#,
            r#"
fn foo() -> i32 {
    fun_name()
}

fn $0fun_name() -> i32 {
    if let true = false { 1 } else { 2 }
}
"#,
        );
    }

    #[test]
    fn no_args_match() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    $0match true {
        true => 1,
        false => 2,
    }$0
}
"#,
            r#"
fn foo() -> i32 {
    fun_name()
}

fn $0fun_name() -> i32 {
    match true {
        true => 1,
        false => 2,
    }
}
"#,
        );
    }

    #[test]
    fn no_args_while() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0while true { }$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    while true { }
}
"#,
        );
    }

    #[test]
    fn no_args_for() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0for v in &[0, 1] { }$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    for v in &[0, 1] { }
}
"#,
        );
    }

    #[test]
    fn no_args_from_loop_unit() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0loop {
        let m = 1;
    }$0
}
"#,
            r#"
fn foo() {
    fun_name()
}

fn $0fun_name() -> ! {
    loop {
        let m = 1;
    }
}
"#,
        );
    }

    #[test]
    fn no_args_from_loop_with_return() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let v = $0loop {
        let m = 1;
        break m;
    }$0;
}
"#,
            r#"
fn foo() {
    let v = fun_name();
}

fn $0fun_name() -> i32 {
    loop {
        let m = 1;
        break m;
    }
}
"#,
        );
    }

    #[test]
    fn no_args_from_match() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let v: i32 = $0match Some(1) {
        Some(x) => x,
        None => 0,
    }$0;
}
"#,
            r#"
fn foo() {
    let v: i32 = fun_name();
}

fn $0fun_name() -> i32 {
    match Some(1) {
        Some(x) => x,
        None => 0,
    }
}
"#,
        );
    }

    #[test]
    fn extract_partial_block_single_line() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    let mut v = $0n * n;$0
    v += 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let mut v = fun_name(n);
    v += 1;
}

fn $0fun_name(n: i32) -> i32 {
    let mut v = n * n;
    v
}
"#,
        );
    }

    #[test]
    fn extract_partial_block() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let m = 2;
    let n = 1;
    let mut v = m $0* n;
    let mut w = 3;$0
    v += 1;
    w += 1;
}
"#,
            r#"
fn foo() {
    let m = 2;
    let n = 1;
    let (mut v, mut w) = fun_name(m, n);
    v += 1;
    w += 1;
}

fn $0fun_name(m: i32, n: i32) -> (i32, i32) {
    let mut v = m * n;
    let mut w = 3;
    (v, w)
}
"#,
        );
    }

    #[test]
    fn argument_form_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    $0n+2$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    n+2
}
"#,
        )
    }

    #[test]
    fn argument_used_twice_form_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    $0n+n$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    n+n
}
"#,
        )
    }

    #[test]
    fn two_arguments_form_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    let m = 3;
    $0n+n*m$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    let m = 3;
    fun_name(n, m)
}

fn $0fun_name(n: u32, m: u32) -> u32 {
    n+n*m
}
"#,
        )
    }

    #[test]
    fn argument_and_locals() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    $0let m = 1;
    n + m$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    let m = 1;
    n + m
}
"#,
        )
    }

    #[test]
    fn in_comment_is_not_applicable() {
        cov_mark::check!(extract_function_in_comment_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn main() { 1 + /* $0comment$0 */ 1; }");
    }

    #[test]
    fn part_of_expr_stmt() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $01$0 + 1;
}
"#,
            r#"
fn foo() {
    fun_name() + 1;
}

fn $0fun_name() -> i32 {
    1
}
"#,
        );
    }

    #[test]
    fn function_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0bar(1 + 1)$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    bar(1 + 1)
}
"#,
        )
    }

    #[test]
    fn extract_from_nested() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let x = true;
    let tuple = match x {
        true => ($02 + 2$0, true)
        _ => (0, false)
    };
}
"#,
            r#"
fn main() {
    let x = true;
    let tuple = match x {
        true => (fun_name(), true)
        _ => (0, false)
    };
}

fn $0fun_name() -> i32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn param_from_closure() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let lambda = |x: u32| $0x * 2$0;
}
"#,
            r#"
fn main() {
    let lambda = |x: u32| fun_name(x);
}

fn $0fun_name(x: u32) -> u32 {
    x * 2
}
"#,
        );
    }

    #[test]
    fn extract_return_stmt() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    $0return 2 + 2$0;
}
"#,
            r#"
fn foo() -> u32 {
    return fun_name();
}

fn $0fun_name() -> u32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn does_not_add_extra_whitespace() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {


    $0return 2 + 2$0;
}
"#,
            r#"
fn foo() -> u32 {


    return fun_name();
}

fn $0fun_name() -> u32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn break_stmt() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let result = loop {
        $0break 2 + 2$0;
    };
}
"#,
            r#"
fn main() {
    let result = loop {
        break fun_name();
    };
}

fn $0fun_name() -> i32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn extract_cast() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let v = $00f32 as u32$0;
}
"#,
            r#"
fn main() {
    let v = fun_name();
}

fn $0fun_name() -> u32 {
    0f32 as u32
}
"#,
        );
    }

    #[test]
    fn return_not_applicable() {
        check_assist_not_applicable(extract_function, r"fn foo() { $0return$0; } ");
    }

    #[test]
    fn method_to_freestanding() {
        check_assist(
            extract_function,
            r#"
struct S;

impl S {
    fn foo(&self) -> i32 {
        $01+1$0
    }
}
"#,
            r#"
struct S;

impl S {
    fn foo(&self) -> i32 {
        fun_name()
    }
}

fn $0fun_name() -> i32 {
    1+1
}
"#,
        );
    }

    #[test]
    fn method_with_reference() {
        check_assist(
            extract_function,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&self) -> i32 {
        $01+self.f$0
    }
}
"#,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&self) -> i32 {
        self.fun_name()
    }

    fn $0fun_name(&self) -> i32 {
        1+self.f
    }
}
"#,
        );
    }

    #[test]
    fn method_with_mut() {
        check_assist(
            extract_function,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&mut self) {
        $0self.f += 1;$0
    }
}
"#,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&mut self) {
        self.fun_name();
    }

    fn $0fun_name(&mut self) {
        self.f += 1;
    }
}
"#,
        );
    }

    #[test]
    fn variable_defined_inside_and_used_after_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let k = n * n;$0
    let m = k + 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let k = fun_name(n);
    let m = k + 1;
}

fn $0fun_name(n: i32) -> i32 {
    let k = n * n;
    k
}
"#,
        );
    }

    #[test]
    fn variable_defined_inside_and_used_after_mutably_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let mut k = n * n;$0
    k += 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let mut k = fun_name(n);
    k += 1;
}

fn $0fun_name(n: i32) -> i32 {
    let mut k = n * n;
    k
}
"#,
        );
    }

    #[test]
    fn two_variables_defined_inside_and_used_after_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let k = n * n;
    let m = k + 2;$0
    let h = k + m;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let (k, m) = fun_name(n);
    let h = k + m;
}

fn $0fun_name(n: i32) -> (i32, i32) {
    let k = n * n;
    let m = k + 2;
    (k, m)
}
"#,
        );
    }

    #[test]
    fn multi_variables_defined_inside_and_used_after_mutably_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let mut k = n * n;
    let mut m = k + 2;
    let mut o = m + 3;
    o += 1;$0
    k += o;
    m = 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let (mut k, mut m, o) = fun_name(n);
    k += o;
    m = 1;
}

fn $0fun_name(n: i32) -> (i32, i32, i32) {
    let mut k = n * n;
    let mut m = k + 2;
    let mut o = m + 3;
    o += 1;
    (k, m, o)
}
"#,
        );
    }

    #[test]
    fn nontrivial_patterns_define_variables() {
        check_assist(
            extract_function,
            r#"
struct Counter(i32);
fn foo() {
    $0let Counter(n) = Counter(0);$0
    let m = n;
}
"#,
            r#"
struct Counter(i32);
fn foo() {
    let n = fun_name();
    let m = n;
}

fn $0fun_name() -> i32 {
    let Counter(n) = Counter(0);
    n
}
"#,
        );
    }

    #[test]
    fn struct_with_two_fields_pattern_define_variables() {
        check_assist(
            extract_function,
            r#"
struct Counter { n: i32, m: i32 };
fn foo() {
    $0let Counter { n, m: k } = Counter { n: 1, m: 2 };$0
    let h = n + k;
}
"#,
            r#"
struct Counter { n: i32, m: i32 };
fn foo() {
    let (n, k) = fun_name();
    let h = n + k;
}

fn $0fun_name() -> (i32, i32) {
    let Counter { n, m: k } = Counter { n: 1, m: 2 };
    (n, k)
}
"#,
        );
    }

    #[test]
    fn mut_var_from_outer_scope() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let mut n = 1;
    $0n += 1;$0
    let m = n + 1;
}
"#,
            r#"
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let m = n + 1;
}

fn $0fun_name(n: &mut i32) {
    *n += 1;
}
"#,
        );
    }

    #[test]
    fn mut_field_from_outer_scope() {
        check_assist(
            extract_function,
            r#"
struct C { n: i32 }
fn foo() {
    let mut c = C { n: 0 };
    $0c.n += 1;$0
    let m = c.n + 1;
}
"#,
            r#"
struct C { n: i32 }
fn foo() {
    let mut c = C { n: 0 };
    fun_name(&mut c);
    let m = c.n + 1;
}

fn $0fun_name(c: &mut C) {
    c.n += 1;
}
"#,
        );
    }

    #[test]
    fn mut_nested_field_from_outer_scope() {
        check_assist(
            extract_function,
            r#"
struct P { n: i32}
struct C { p: P }
fn foo() {
    let mut c = C { p: P { n: 0 } };
    let mut v = C { p: P { n: 0 } };
    let u = C { p: P { n: 0 } };
    $0c.p.n += u.p.n;
    let r = &mut v.p.n;$0
    let m = c.p.n + v.p.n + u.p.n;
}
"#,
            r#"
struct P { n: i32}
struct C { p: P }
fn foo() {
    let mut c = C { p: P { n: 0 } };
    let mut v = C { p: P { n: 0 } };
    let u = C { p: P { n: 0 } };
    fun_name(&mut c, &u, &mut v);
    let m = c.p.n + v.p.n + u.p.n;
}

fn $0fun_name(c: &mut C, u: &C, v: &mut C) {
    c.p.n += u.p.n;
    let r = &mut v.p.n;
}
"#,
        );
    }

    #[test]
    fn mut_param_many_usages_stmt() {
        check_assist(
            extract_function,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    $0n += n;
    bar(n);
    bar(n+1);
    bar(n*n);
    bar(&n);
    n.inc();
    let v = &mut n;
    *v = v.succ();
    n.succ();$0
    let m = n + 1;
}
"#,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let m = n + 1;
}

fn $0fun_name(n: &mut i32) {
    *n += *n;
    bar(*n);
    bar(*n+1);
    bar(*n**n);
    bar(&*n);
    n.inc();
    let v = n;
    *v = v.succ();
    n.succ();
}
"#,
        );
    }

    #[test]
    fn mut_param_many_usages_expr() {
        check_assist(
            extract_function,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    $0{
        n += n;
        bar(n);
        bar(n+1);
        bar(n*n);
        bar(&n);
        n.inc();
        let v = &mut n;
        *v = v.succ();
        n.succ();
    }$0
    let m = n + 1;
}
"#,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let m = n + 1;
}

fn $0fun_name(n: &mut i32) {
    *n += *n;
    bar(*n);
    bar(*n+1);
    bar(*n**n);
    bar(&*n);
    n.inc();
    let v = n;
    *v = v.succ();
    n.succ();
}
"#,
        );
    }

    #[test]
    fn mut_param_by_value() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let mut n = 1;
    $0n += 1;$0
}
"#,
            r"
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    n += 1;
}
",
        );
    }

    #[test]
    fn mut_param_because_of_mut_ref() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let mut n = 1;
    $0let v = &mut n;
    *v += 1;$0
    let k = n;
}
"#,
            r#"
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let k = n;
}

fn $0fun_name(n: &mut i32) {
    let v = n;
    *v += 1;
}
"#,
        );
    }

    #[test]
    fn mut_param_by_value_because_of_mut_ref() {
        check_assist(
            extract_function,
            r"
fn foo() {
    let mut n = 1;
    $0let v = &mut n;
    *v += 1;$0
}
",
            r#"
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    let v = &mut n;
    *v += 1;
}
"#,
        );
    }

    #[test]
    fn mut_method_call() {
        check_assist(
            extract_function,
            r#"
trait I {
    fn inc(&mut self);
}
impl I for i32 {
    fn inc(&mut self) { *self += 1 }
}
fn foo() {
    let mut n = 1;
    $0n.inc();$0
}
"#,
            r#"
trait I {
    fn inc(&mut self);
}
impl I for i32 {
    fn inc(&mut self) { *self += 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    n.inc();
}
"#,
        );
    }

    #[test]
    fn shared_method_call() {
        check_assist(
            extract_function,
            r#"
trait I {
    fn succ(&self);
}
impl I for i32 {
    fn succ(&self) { *self + 1 }
}
fn foo() {
    let mut n = 1;
    $0n.succ();$0
}
"#,
            r"
trait I {
    fn succ(&self);
}
impl I for i32 {
    fn succ(&self) { *self + 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(n: i32) {
    n.succ();
}
",
        );
    }

    #[test]
    fn mut_method_call_with_other_receiver() {
        check_assist(
            extract_function,
            r#"
trait I {
    fn inc(&mut self, n: i32);
}
impl I for i32 {
    fn inc(&mut self, n: i32) { *self += n }
}
fn foo() {
    let mut n = 1;
    $0let mut m = 2;
    m.inc(n);$0
}
"#,
            r"
trait I {
    fn inc(&mut self, n: i32);
}
impl I for i32 {
    fn inc(&mut self, n: i32) { *self += n }
}
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(n: i32) {
    let mut m = 2;
    m.inc(n);
}
",
        );
    }

    #[test]
    fn non_copy_without_usages_after() {
        check_assist(
            extract_function,
            r#"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    $0let n = c.0;$0
}
"#,
            r"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    fun_name(c);
}

fn $0fun_name(c: Counter) {
    let n = c.0;
}
",
        );
    }

    #[test]
    fn non_copy_used_after() {
        check_assist(
            extract_function,
            r"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    $0let n = c.0;$0
    let m = c.0;
}
",
            r#"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    fun_name(&c);
    let m = c.0;
}

fn $0fun_name(c: &Counter) {
    let n = c.0;
}
"#,
        );
    }

    #[test]
    fn copy_used_after() {
        check_assist(
            extract_function,
            r#"
//- minicore: copy
fn foo() {
    let n = 0;
    $0let m = n;$0
    let k = n;
}
"#,
            r#"
fn foo() {
    let n = 0;
    fun_name(n);
    let k = n;
}

fn $0fun_name(n: i32) {
    let m = n;
}
"#,
        )
    }

    #[test]
    fn copy_custom_used_after() {
        check_assist(
            extract_function,
            r#"
//- minicore: copy, derive
#[derive(Clone, Copy)]
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    $0let n = c.0;$0
    let m = c.0;
}
"#,
            r#"
#[derive(Clone, Copy)]
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    fun_name(c);
    let m = c.0;
}

fn $0fun_name(c: Counter) {
    let n = c.0;
}
"#,
        );
    }

    #[test]
    fn indented_stmts() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    if true {
        loop {
            $0let n = 1;
            let m = 2;$0
        }
    }
}
"#,
            r#"
fn foo() {
    if true {
        loop {
            fun_name();
        }
    }
}

fn $0fun_name() {
    let n = 1;
    let m = 2;
}
"#,
        );
    }

    #[test]
    fn indented_stmts_inside_mod() {
        check_assist(
            extract_function,
            r#"
mod bar {
    fn foo() {
        if true {
            loop {
                $0let n = 1;
                let m = 2;$0
            }
        }
    }
}
"#,
            r#"
mod bar {
    fn foo() {
        if true {
            loop {
                fun_name();
            }
        }
    }

    fn $0fun_name() {
        let n = 1;
        let m = 2;
    }
}
"#,
        );
    }

    #[test]
    fn break_loop() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn foo() {
    loop {
        let n = 1;
        $0let m = n + 1;
        break;
        let k = 2;$0
        let h = 1 + k;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let k = match fun_name(n) {
            Some(value) => value,
            None => break,
        };
        let h = 1 + k;
    }
}

fn $0fun_name(n: i32) -> Option<i32> {
    let m = n + 1;
    return None;
    let k = 2;
    Some(k)
}
"#,
        );
    }

    #[test]
    fn return_to_parent() {
        check_assist(
            extract_function,
            r#"
//- minicore: copy, result
fn foo() -> i64 {
    let n = 1;
    $0let m = n + 1;
    return 1;
    let k = 2;$0
    (n + k) as i64
}
"#,
            r#"
fn foo() -> i64 {
    let n = 1;
    let k = match fun_name(n) {
        Ok(value) => value,
        Err(value) => return value,
    };
    (n + k) as i64
}

fn $0fun_name(n: i32) -> Result<i32, i64> {
    let m = n + 1;
    return Err(1);
    let k = 2;
    Ok(k)
}
"#,
        );
    }

    #[test]
    fn break_and_continue() {
        cov_mark::check!(external_control_flow_break_and_continue);
        check_assist_not_applicable(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let m = n + 1;
        break;
        let k = 2;
        continue;
        let k = k + 1;$0
        let r = n + k;
    }
}
"#,
        );
    }

    #[test]
    fn return_and_break() {
        cov_mark::check!(external_control_flow_return_and_bc);
        check_assist_not_applicable(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let m = n + 1;
        break;
        let k = 2;
        return;
        let k = k + 1;$0
        let r = n + k;
    }
}
"#,
        );
    }

    #[test]
    fn break_loop_with_if() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let mut n = 1;
        $0let m = n + 1;
        break;
        n += m;$0
        let h = 1 + n;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let mut n = 1;
        if fun_name(&mut n) {
            break;
        }
        let h = 1 + n;
    }
}

fn $0fun_name(n: &mut i32) -> bool {
    let m = *n + 1;
    return true;
    *n += m;
    false
}
"#,
        );
    }

    #[test]
    fn break_loop_nested() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let mut n = 1;
        $0let m = n + 1;
        if m == 42 {
            break;
        }$0
        let h = 1;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let mut n = 1;
        if fun_name(n) {
            break;
        }
        let h = 1;
    }
}

fn $0fun_name(n: i32) -> bool {
    let m = n + 1;
    if m == 42 {
        return true;
    }
    false
}
"#,
        );
    }

    #[test]
    fn return_from_nested_loop() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0
        let k = 1;
        loop {
            return;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = match fun_name() {
            Some(value) => value,
            None => return,
        };
        let h = 1 + m;
    }
}

fn $0fun_name() -> Option<i32> {
    let k = 1;
    loop {
        return None;
    }
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn break_from_nested_loop() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let k = 1;
        loop {
            break;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = fun_name();
        let h = 1 + m;
    }
}

fn $0fun_name() -> i32 {
    let k = 1;
    loop {
        break;
    }
    let m = k + 1;
    m
}
"#,
        );
    }

    #[test]
    fn break_from_nested_and_outer_loops() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let k = 1;
        loop {
            break;
        }
        if k == 42 {
            break;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = match fun_name() {
            Some(value) => value,
            None => break,
        };
        let h = 1 + m;
    }
}

fn $0fun_name() -> Option<i32> {
    let k = 1;
    loop {
        break;
    }
    if k == 42 {
        return None;
    }
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn return_from_nested_fn() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let k = 1;
        fn test() {
            return;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = fun_name();
        let h = 1 + m;
    }
}

fn $0fun_name() -> i32 {
    let k = 1;
    fn test() {
        return;
    }
    let m = k + 1;
    m
}
"#,
        );
    }

    #[test]
    fn break_with_value() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    loop {
        let n = 1;
        $0let k = 1;
        if k == 42 {
            break 3;
        }
        let m = k + 1;$0
        let h = 1;
    }
}
"#,
            r#"
fn foo() -> i32 {
    loop {
        let n = 1;
        if let Some(value) = fun_name() {
            break value;
        }
        let h = 1;
    }
}

fn $0fun_name() -> Option<i32> {
    let k = 1;
    if k == 42 {
        return Some(3);
    }
    let m = k + 1;
    None
}
"#,
        );
    }

    #[test]
    fn break_with_value_and_return() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i64 {
    loop {
        let n = 1;
        $0
        let k = 1;
        if k == 42 {
            break 3;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() -> i64 {
    loop {
        let n = 1;
        let m = match fun_name() {
            Ok(value) => value,
            Err(value) => break value,
        };
        let h = 1 + m;
    }
}

fn $0fun_name() -> Result<i32, i64> {
    let k = 1;
    if k == 42 {
        return Err(3);
    }
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn try_option() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn bar() -> Option<i32> { None }
fn foo() -> Option<()> {
    let n = bar()?;
    $0let k = foo()?;
    let m = k + 1;$0
    let h = 1 + m;
    Some(())
}
"#,
            r#"
fn bar() -> Option<i32> { None }
fn foo() -> Option<()> {
    let n = bar()?;
    let m = fun_name()?;
    let h = 1 + m;
    Some(())
}

fn $0fun_name() -> Option<i32> {
    let k = foo()?;
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn try_option_unit() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn foo() -> Option<()> {
    let n = 1;
    $0let k = foo()?;
    let m = k + 1;$0
    let h = 1 + n;
    Some(())
}
"#,
            r#"
fn foo() -> Option<()> {
    let n = 1;
    fun_name()?;
    let h = 1 + n;
    Some(())
}

fn $0fun_name() -> Option<()> {
    let k = foo()?;
    let m = k + 1;
    Some(())
}
"#,
        );
    }

    #[test]
    fn try_result() {
        check_assist(
            extract_function,
            r#"
//- minicore: result
fn foo() -> Result<(), i64> {
    let n = 1;
    $0let k = foo()?;
    let m = k + 1;$0
    let h = 1 + m;
    Ok(())
}
"#,
            r#"
fn foo() -> Result<(), i64> {
    let n = 1;
    let m = fun_name()?;
    let h = 1 + m;
    Ok(())
}

fn $0fun_name() -> Result<i32, i64> {
    let k = foo()?;
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn try_option_with_return() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn foo() -> Option<()> {
    let n = 1;
    $0let k = foo()?;
    if k == 42 {
        return None;
    }
    let m = k + 1;$0
    let h = 1 + m;
    Some(())
}
"#,
            r#"
fn foo() -> Option<()> {
    let n = 1;
    let m = fun_name()?;
    let h = 1 + m;
    Some(())
}

fn $0fun_name() -> Option<i32> {
    let k = foo()?;
    if k == 42 {
        return None;
    }
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn try_result_with_return() {
        check_assist(
            extract_function,
            r#"
//- minicore: result
fn foo() -> Result<(), i64> {
    let n = 1;
    $0let k = foo()?;
    if k == 42 {
        return Err(1);
    }
    let m = k + 1;$0
    let h = 1 + m;
    Ok(())
}
"#,
            r#"
fn foo() -> Result<(), i64> {
    let n = 1;
    let m = fun_name()?;
    let h = 1 + m;
    Ok(())
}

fn $0fun_name() -> Result<i32, i64> {
    let k = foo()?;
    if k == 42 {
        return Err(1);
    }
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn try_and_break() {
        cov_mark::check!(external_control_flow_try_and_bc);
        check_assist_not_applicable(
            extract_function,
            r#"
//- minicore: option
fn foo() -> Option<()> {
    loop {
        let n = Some(1);
        $0let m = n? + 1;
        break;
        let k = 2;
        let k = k + 1;$0
        let r = n + k;
    }
    Some(())
}
"#,
        );
    }

    #[test]
    fn try_and_return_ok() {
        cov_mark::check!(external_control_flow_try_and_return_non_err);
        check_assist_not_applicable(
            extract_function,
            r#"
//- minicore: result
fn foo() -> Result<(), i64> {
    let n = 1;
    $0let k = foo()?;
    if k == 42 {
        return Ok(1);
    }
    let m = k + 1;$0
    let h = 1 + m;
    Ok(())
}
"#,
        );
    }

    #[test]
    fn param_usage_in_macro() {
        check_assist(
            extract_function,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

fn foo() {
    let n = 1;
    $0let k = n * m!(n);$0
    let m = k + 1;
}
"#,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

fn foo() {
    let n = 1;
    let k = fun_name(n);
    let m = k + 1;
}

fn $0fun_name(n: i32) -> i32 {
    let k = n * m!(n);
    k
}
"#,
        );
    }

    #[test]
    fn extract_with_await() {
        check_assist(
            extract_function,
            r#"
fn main() {
    $0some_function().await;$0
}

async fn some_function() {

}
"#,
            r#"
fn main() {
    fun_name().await;
}

async fn $0fun_name() {
    some_function().await;
}

async fn some_function() {

}
"#,
        );
    }

    #[test]
    fn extract_with_await_in_args() {
        check_assist(
            extract_function,
            r#"
fn main() {
    $0function_call("a", some_function().await);$0
}

async fn some_function() {

}
"#,
            r#"
fn main() {
    fun_name().await;
}

async fn $0fun_name() {
    function_call("a", some_function().await);
}

async fn some_function() {

}
"#,
        );
    }
}

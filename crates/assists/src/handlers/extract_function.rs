use either::Either;
use hir::{HirDisplay, Local};
use ide_db::{
    defs::{Definition, NameRefClass},
    search::{FileReference, ReferenceAccess, SearchScope},
};
use itertools::Itertools;
use stdx::format_to;
use syntax::{
    algo::SyntaxRewriter,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        AstNode,
    },
    Direction, SyntaxElement,
    SyntaxKind::{self, BLOCK_EXPR, BREAK_EXPR, COMMENT, PATH_EXPR, RETURN_EXPR},
    SyntaxNode, SyntaxToken, TextRange, TextSize, TokenAtOffset, T,
};
use test_utils::mark;

use crate::{
    assist_context::{AssistContext, Assists},
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
        mark::hit!(extract_function_in_comment_is_not_applicable);
        return None;
    }

    let node = match node {
        syntax::NodeOrToken::Node(n) => n,
        syntax::NodeOrToken::Token(t) => t.parent(),
    };

    let mut body = None;
    if node.text_range() == ctx.frange.range {
        body = FunctionBody::from_whole_node(node.clone());
    }
    if body.is_none() && node.kind() == BLOCK_EXPR {
        body = FunctionBody::from_range(&node, ctx.frange.range);
    }
    if let Some(parent) = node.parent() {
        if body.is_none() && parent.kind() == BLOCK_EXPR {
            body = FunctionBody::from_range(&parent, ctx.frange.range);
        }
    }
    if body.is_none() {
        body = FunctionBody::from_whole_node(node.clone());
    }
    if body.is_none() {
        body = node.ancestors().find_map(FunctionBody::from_whole_node);
    }
    let body = body?;

    let vars_used_in_body = vars_used_in_body(&body, &ctx);
    let mut self_param = None;
    let param_pats: Vec<_> = vars_used_in_body
        .iter()
        .map(|node| (node, node.source(ctx.db())))
        .filter(|(_, src)| {
            src.file_id.original_file(ctx.db()) == ctx.frange.file_id
                && !body.contains_node(&either_syntax(&src.value))
        })
        .filter_map(|(&node, src)| match src.value {
            Either::Left(_) => Some(node),
            Either::Right(it) => {
                // we filter self param, as there can only be one
                self_param = Some((node, it));
                None
            }
        })
        .collect();

    let anchor = if self_param.is_some() { Anchor::Method } else { Anchor::Freestanding };
    let insert_after = body.scope_for_fn_insertion(anchor)?;
    let module = ctx.sema.scope(&insert_after).module()?;

    let vars_defined_in_body = vars_defined_in_body(&body, ctx);

    let vars_defined_in_body_and_outlive: Vec<_> = vars_defined_in_body
        .iter()
        .copied()
        .filter(|node| {
            let usages = Definition::Local(*node)
                .usages(&ctx.sema)
                .in_scope(SearchScope::single_file(ctx.frange.file_id))
                .all();
            let mut usages = usages.iter().flat_map(|(_, rs)| rs.iter());

            usages.any(|reference| body.preceedes_range(reference.range))
        })
        .collect();

    let params: Vec<_> = param_pats
        .into_iter()
        .map(|node| {
            let usages = Definition::Local(node)
                .usages(&ctx.sema)
                .in_scope(SearchScope::single_file(ctx.frange.file_id))
                .all();

            let has_usages_afterwards = usages
                .iter()
                .flat_map(|(_, rs)| rs.iter())
                .any(|reference| body.preceedes_range(reference.range));
            let has_mut_inside_body = usages
                .iter()
                .flat_map(|(_, rs)| rs.iter())
                .filter(|reference| body.contains_range(reference.range))
                .any(|reference| {
                    if reference.access == Some(ReferenceAccess::Write) {
                        return true;
                    }

                    let path = path_at_offset(&body, reference);
                    if is_mut_ref_expr(path.as_ref()).unwrap_or(false) {
                        return true;
                    }

                    if is_mut_method_call(ctx, path.as_ref()).unwrap_or(false) {
                        return true;
                    }

                    false
                });

            Param { node, has_usages_afterwards, has_mut_inside_body, is_copy: true }
        })
        .collect();

    let expr = body.tail_expr();
    let ret_ty = match expr {
        Some(expr) => Some(ctx.sema.type_of_expr(&expr)?),
        None => None,
    };

    let has_unit_ret = ret_ty.as_ref().map_or(true, |it| it.is_unit());
    if stdx::never!(!vars_defined_in_body_and_outlive.is_empty() && !has_unit_ret) {
        // We should not have variables that outlive body if we have expression block
        return None;
    }

    let target_range = match &body {
        FunctionBody::Expr(expr) => expr.syntax().text_range(),
        FunctionBody::Span { .. } => ctx.frange.range,
    };

    acc.add(
        AssistId("extract_function", crate::AssistKind::RefactorExtract),
        "Extract into function",
        target_range,
        move |builder| {
            let fun = Function {
                name: "fun_name".to_string(),
                self_param: self_param.map(|(_, pat)| pat),
                params,
                ret_ty,
                body,
                vars_defined_in_body_and_outlive,
            };

            builder.replace(target_range, format_replacement(ctx, &fun));

            let indent = IndentLevel::from_node(&insert_after);

            let fn_def = format_function(ctx, module, &fun, indent);
            let insert_offset = insert_after.text_range().end();
            builder.insert(insert_offset, fn_def);
        },
    )
}

fn format_replacement(ctx: &AssistContext, fun: &Function) -> String {
    let mut buf = String::new();

    match fun.vars_defined_in_body_and_outlive.as_slice() {
        [] => {}
        [var] => format_to!(buf, "let {} = ", var.name(ctx.db()).unwrap()),
        [v0, vs @ ..] => {
            buf.push_str("let (");
            format_to!(buf, "{}", v0.name(ctx.db()).unwrap());
            for local in vs {
                format_to!(buf, ", {}", local.name(ctx.db()).unwrap());
            }
            buf.push_str(") = ");
        }
    }

    if fun.self_param.is_some() {
        format_to!(buf, "self.");
    }
    format_to!(buf, "{}(", fun.name);
    {
        let mut it = fun.params.iter();
        if let Some(param) = it.next() {
            format_to!(buf, "{}{}", param.value_prefix(), param.node.name(ctx.db()).unwrap());
        }
        for param in it {
            format_to!(buf, ", {}{}", param.value_prefix(), param.node.name(ctx.db()).unwrap());
        }
    }
    format_to!(buf, ")");

    if fun.has_unit_ret() {
        format_to!(buf, ";");
    }

    buf
}

struct Function {
    name: String,
    self_param: Option<ast::SelfParam>,
    params: Vec<Param>,
    ret_ty: Option<hir::Type>,
    body: FunctionBody,
    vars_defined_in_body_and_outlive: Vec<Local>,
}

impl Function {
    fn has_unit_ret(&self) -> bool {
        match &self.ret_ty {
            Some(ty) => ty.is_unit(),
            None => true,
        }
    }
}

#[derive(Debug)]
struct Param {
    node: Local,
    has_usages_afterwards: bool,
    has_mut_inside_body: bool,
    is_copy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    Value,
    MutValue,
    SharedRef,
    MutRef,
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

    fn value_prefix(&self) -> &'static str {
        match self.kind() {
            ParamKind::Value => "",
            ParamKind::MutValue => "",
            ParamKind::SharedRef => "&",
            ParamKind::MutRef => "&mut ",
        }
    }

    fn type_prefix(&self) -> &'static str {
        match self.kind() {
            ParamKind::Value => "",
            ParamKind::MutValue => "",
            ParamKind::SharedRef => "&",
            ParamKind::MutRef => "&mut ",
        }
    }

    fn mut_pattern(&self) -> &'static str {
        match self.kind() {
            ParamKind::MutValue => "mut ",
            _ => "",
        }
    }
}

fn format_function(
    ctx: &AssistContext,
    module: hir::Module,
    fun: &Function,
    indent: IndentLevel,
) -> String {
    let mut fn_def = String::new();
    format_to!(fn_def, "\n\n{}fn $0{}(", indent, fun.name);
    {
        let mut it = fun.params.iter();
        if let Some(self_param) = &fun.self_param {
            format_to!(fn_def, "{}", self_param);
        } else if let Some(param) = it.next() {
            format_to!(
                fn_def,
                "{}{}: {}{}",
                param.mut_pattern(),
                param.node.name(ctx.db()).unwrap(),
                param.type_prefix(),
                format_type(&param.node.ty(ctx.db()), ctx, module)
            );
        }
        for param in it {
            format_to!(
                fn_def,
                ", {}{}: {}{}",
                param.mut_pattern(),
                param.node.name(ctx.db()).unwrap(),
                param.type_prefix(),
                format_type(&param.node.ty(ctx.db()), ctx, module)
            );
        }
    }

    format_to!(fn_def, ")");
    if !fun.has_unit_ret() {
        if let Some(ty) = &fun.ret_ty {
            format_to!(fn_def, " -> {}", format_type(ty, ctx, module));
        }
    } else {
        match fun.vars_defined_in_body_and_outlive.as_slice() {
            [] => {}
            [var] => {
                format_to!(fn_def, " -> {}", format_type(&var.ty(ctx.db()), ctx, module));
            }
            [v0, vs @ ..] => {
                format_to!(fn_def, " -> ({}", format_type(&v0.ty(ctx.db()), ctx, module));
                for var in vs {
                    format_to!(fn_def, ", {}", format_type(&var.ty(ctx.db()), ctx, module));
                }
                fn_def.push(')');
            }
        }
    }
    fn_def.push_str(" {");

    match &fun.body {
        FunctionBody::Expr(expr) => {
            fn_def.push('\n');
            let expr = expr.indent(indent);
            let expr = fix_param_usages(ctx, &fun.params, expr.syntax());
            format_to!(fn_def, "{}{}", indent + 1, expr);
            fn_def.push('\n');
        }
        FunctionBody::Span { elements, leading_indent } => {
            format_to!(fn_def, "{}", leading_indent);
            for element in elements {
                match element {
                    syntax::NodeOrToken::Node(node) => {
                        format_to!(fn_def, "{}", fix_param_usages(ctx, &fun.params, node));
                    }
                    syntax::NodeOrToken::Token(token) => {
                        format_to!(fn_def, "{}", token);
                    }
                }
            }
            if !fn_def.ends_with('\n') {
                fn_def.push('\n');
            }
        }
    }

    match fun.vars_defined_in_body_and_outlive.as_slice() {
        [] => {}
        [var] => format_to!(fn_def, "{}{}\n", indent + 1, var.name(ctx.db()).unwrap()),
        [v0, vs @ ..] => {
            format_to!(fn_def, "{}({}", indent + 1, v0.name(ctx.db()).unwrap());
            for var in vs {
                format_to!(fn_def, ", {}", var.name(ctx.db()).unwrap());
            }
            fn_def.push_str(")\n");
        }
    }

    format_to!(fn_def, "{}}}", indent);

    fn_def
}

fn format_type(ty: &hir::Type, ctx: &AssistContext, module: hir::Module) -> String {
    ty.display_source_code(ctx.db(), module.into()).ok().unwrap_or_else(|| "()".to_string())
}

fn path_at_offset(body: &FunctionBody, reference: &FileReference) -> Option<ast::Expr> {
    let var = body.token_at_offset(reference.range.start()).right_biased()?;
    let path = var.ancestors().find_map(ast::Expr::cast)?;
    stdx::always!(matches!(path, ast::Expr::PathExpr(_)));
    Some(path)
}

fn is_mut_ref_expr(path: Option<&ast::Expr>) -> Option<bool> {
    let path = path?;
    let ref_expr = path.syntax().parent().and_then(ast::RefExpr::cast)?;
    Some(ref_expr.mut_token().is_some())
}

fn is_mut_method_call(ctx: &AssistContext, path: Option<&ast::Expr>) -> Option<bool> {
    let path = path?;
    let method_call = path.syntax().parent().and_then(ast::MethodCallExpr::cast)?;

    let func = ctx.sema.resolve_method_call(&method_call)?;
    let self_param = func.self_param(ctx.db())?;
    let access = self_param.access(ctx.db());

    Some(matches!(access, hir::Access::Exclusive))
}

fn fix_param_usages(ctx: &AssistContext, params: &[Param], syntax: &SyntaxNode) -> SyntaxNode {
    let mut rewriter = SyntaxRewriter::default();
    for param in params {
        if !param.kind().is_ref() {
            continue;
        }

        let usages = Definition::Local(param.node)
            .usages(&ctx.sema)
            .in_scope(SearchScope::single_file(ctx.frange.file_id))
            .all();
        let usages = usages
            .iter()
            .flat_map(|(_, rs)| rs.iter())
            .filter(|reference| syntax.text_range().contains_range(reference.range));
        for reference in usages {
            let token = match syntax.token_at_offset(reference.range.start()).right_biased() {
                Some(a) => a,
                None => {
                    stdx::never!(false, "cannot find token at variable usage: {:?}", reference);
                    continue;
                }
            };
            let path = match token.ancestors().find_map(ast::Expr::cast) {
                Some(n) => n,
                None => {
                    stdx::never!(false, "cannot find path parent of variable usage: {:?}", token);
                    continue;
                }
            };
            stdx::always!(matches!(path, ast::Expr::PathExpr(_)));
            match path.syntax().ancestors().skip(1).find_map(ast::Expr::cast) {
                Some(ast::Expr::MethodCallExpr(_)) => {
                    // do nothing
                }
                Some(ast::Expr::RefExpr(node))
                    if param.kind() == ParamKind::MutRef && node.mut_token().is_some() =>
                {
                    rewriter.replace_ast(&node.clone().into(), &node.expr().unwrap());
                }
                Some(ast::Expr::RefExpr(node))
                    if param.kind() == ParamKind::SharedRef && node.mut_token().is_none() =>
                {
                    rewriter.replace_ast(&node.clone().into(), &node.expr().unwrap());
                }
                Some(_) | None => {
                    rewriter.replace_ast(&path, &ast::make::expr_prefix(T![*], path.clone()));
                }
            };
        }
    }

    rewriter.rewrite(syntax)
}

#[derive(Debug)]
enum FunctionBody {
    Expr(ast::Expr),
    Span { elements: Vec<SyntaxElement>, leading_indent: String },
}

enum Anchor {
    Freestanding,
    Method,
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

    fn from_range(node: &SyntaxNode, range: TextRange) -> Option<FunctionBody> {
        let mut first = node.token_at_offset(range.start()).left_biased()?;
        let last = node.token_at_offset(range.end()).right_biased()?;

        let mut leading_indent = String::new();

        let leading_trivia = first
            .siblings_with_tokens(Direction::Prev)
            .skip(1)
            .take_while(|e| e.kind() == SyntaxKind::WHITESPACE && e.as_token().is_some());

        for e in leading_trivia {
            let token = e.as_token().unwrap();
            let text = token.text();
            match text.rfind('\n') {
                Some(pos) => {
                    leading_indent = text[pos..].to_owned();
                    break;
                }
                None => first = token.clone(),
            }
        }

        let mut elements: Vec<_> = first
            .siblings_with_tokens(Direction::Next)
            .take_while(|e| e.as_token() != Some(&last))
            .collect();

        if !(last.kind() == SyntaxKind::WHITESPACE && last.text().lines().count() <= 2) {
            elements.push(last.into());
        }

        Some(FunctionBody::Span { elements, leading_indent })
    }

    fn tail_expr(&self) -> Option<ast::Expr> {
        match &self {
            FunctionBody::Expr(expr) => Some(expr.clone()),
            FunctionBody::Span { elements, .. } => {
                elements.iter().rev().find_map(|e| e.as_node()).cloned().and_then(ast::Expr::cast)
            }
        }
    }

    fn scope_for_fn_insertion(&self, anchor: Anchor) -> Option<SyntaxNode> {
        match self {
            FunctionBody::Expr(e) => scope_for_fn_insertion(e.syntax(), anchor),
            FunctionBody::Span { elements, .. } => {
                let node = elements.iter().find_map(|e| e.as_node())?;
                scope_for_fn_insertion(&node, anchor)
            }
        }
    }

    fn descendants(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        match self {
            FunctionBody::Expr(expr) => Either::Right(expr.syntax().descendants()),
            FunctionBody::Span { elements, .. } => Either::Left(
                elements
                    .iter()
                    .filter_map(SyntaxElement::as_node)
                    .flat_map(SyntaxNode::descendants),
            ),
        }
    }

    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().token_at_offset(offset),
            FunctionBody::Span { elements, .. } => {
                stdx::always!(self.text_range().contains(offset));
                let mut iter = elements
                    .iter()
                    .filter(|element| element.text_range().contains_inclusive(offset));
                let element1 = iter.next().expect("offset does not fall into body");
                let element2 = iter.next();
                stdx::always!(iter.next().is_none(), "> 2 tokens at offset");
                let t1 = match element1 {
                    syntax::NodeOrToken::Node(node) => node.token_at_offset(offset),
                    syntax::NodeOrToken::Token(token) => TokenAtOffset::Single(token.clone()),
                };
                let t2 = element2.map(|e| match e {
                    syntax::NodeOrToken::Node(node) => node.token_at_offset(offset),
                    syntax::NodeOrToken::Token(token) => TokenAtOffset::Single(token.clone()),
                });

                match t2 {
                    Some(t2) => match (t1.clone().right_biased(), t2.clone().left_biased()) {
                        (Some(e1), Some(e2)) => TokenAtOffset::Between(e1, e2),
                        (Some(_), None) => t1,
                        (None, _) => t2,
                    },
                    None => t1,
                }
            }
        }
    }

    fn text_range(&self) -> TextRange {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().text_range(),
            FunctionBody::Span { elements, .. } => TextRange::new(
                elements.first().unwrap().text_range().start(),
                elements.last().unwrap().text_range().end(),
            ),
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

fn scope_for_fn_insertion(node: &SyntaxNode, anchor: Anchor) -> Option<SyntaxNode> {
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

fn either_syntax(value: &Either<ast::IdentPat, ast::SelfParam>) -> &SyntaxNode {
    match value {
        Either::Left(pat) => pat.syntax(),
        Either::Right(it) => it.syntax(),
    }
}

/// Returns a vector of local variables that are referenced in `body`
fn vars_used_in_body(body: &FunctionBody, ctx: &AssistContext) -> Vec<Local> {
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

/// Returns a vector of local variables that are defined in `body`
fn vars_defined_in_body(body: &FunctionBody, ctx: &AssistContext) -> Vec<Local> {
    body.descendants()
        .filter_map(ast::IdentPat::cast)
        .filter_map(|let_stmt| ctx.sema.to_def(&let_stmt))
        .unique()
        .collect()
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
}"#,
            r#"
fn foo() {
    foo(fun_name());
}

fn $0fun_name() -> i32 {
    1 + 1
}"#,
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
}"#,
            r#"
mod bar {
    fn foo() {
        foo(fun_name());
    }

    fn $0fun_name() -> i32 {
        1 + 1
    }
}"#,
        );
    }

    #[test]
    fn no_args_from_binary_expr_indented() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0{ 1 + 1 }$0;
}"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() -> i32 {
    { 1 + 1 }
}"#,
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
}"#,
            r#"
fn foo() -> i32 {
    let k = 1;
    fun_name()
}

fn $0fun_name() -> i32 {
    let m = 1;
    m + 1
}"#,
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
}"#,
            r#"
fn foo() {
    let k = 3;
    fun_name();
    let g = 5;
}

fn $0fun_name() {
    let m = 1;
    let n = m + 1;
}"#,
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
}"#,
            r#"
fn foo() {
    fun_name()
}

fn $0fun_name() -> ! {
    loop {
        let m = 1;
    }
}"#,
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
}"#,
            r#"
fn foo() {
    let v = fun_name();
}

fn $0fun_name() -> i32 {
    loop {
        let m = 1;
        break m;
    }
}"#,
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
}"#,
            r#"
fn foo() {
    let v: i32 = fun_name();
}

fn $0fun_name() -> i32 {
    match Some(1) {
        Some(x) => x,
        None => 0,
    }
}"#,
        );
    }

    #[test]
    fn argument_form_expr() {
        check_assist(
            extract_function,
            r"
fn foo() -> u32 {
    let n = 2;
    $0n+2$0
}",
            r"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    n+2
}",
        )
    }

    #[test]
    fn argument_used_twice_form_expr() {
        check_assist(
            extract_function,
            r"
fn foo() -> u32 {
    let n = 2;
    $0n+n$0
}",
            r"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    n+n
}",
        )
    }

    #[test]
    fn two_arguments_form_expr() {
        check_assist(
            extract_function,
            r"
fn foo() -> u32 {
    let n = 2;
    let m = 3;
    $0n+n*m$0
}",
            r"
fn foo() -> u32 {
    let n = 2;
    let m = 3;
    fun_name(n, m)
}

fn $0fun_name(n: u32, m: u32) -> u32 {
    n+n*m
}",
        )
    }

    #[test]
    fn argument_and_locals() {
        check_assist(
            extract_function,
            r"
fn foo() -> u32 {
    let n = 2;
    $0let m = 1;
    n + m$0
}",
            r"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    let m = 1;
    n + m
}",
        )
    }

    #[test]
    fn in_comment_is_not_applicable() {
        mark::check!(extract_function_in_comment_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn main() { 1 + /* $0comment$0 */ 1; }");
    }

    #[test]
    fn part_of_expr_stmt() {
        check_assist(
            extract_function,
            "
fn foo() {
    $01$0 + 1;
}",
            "
fn foo() {
    fun_name() + 1;
}

fn $0fun_name() -> i32 {
    1
}",
        );
    }

    #[test]
    fn function_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0bar(1 + 1)$0
}"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    bar(1 + 1)
}"#,
        )
    }

    #[test]
    fn extract_from_nested() {
        check_assist(
            extract_function,
            r"
fn main() {
    let x = true;
    let tuple = match x {
        true => ($02 + 2$0, true)
        _ => (0, false)
    };
}",
            r"
fn main() {
    let x = true;
    let tuple = match x {
        true => (fun_name(), true)
        _ => (0, false)
    };
}

fn $0fun_name() -> i32 {
    2 + 2
}",
        );
    }

    #[test]
    fn param_from_closure() {
        check_assist(
            extract_function,
            r"
fn main() {
    let lambda = |x: u32| $0x * 2$0;
}",
            r"
fn main() {
    let lambda = |x: u32| fun_name(x);
}

fn $0fun_name(x: u32) -> u32 {
    x * 2
}",
        );
    }

    #[test]
    fn extract_return_stmt() {
        check_assist(
            extract_function,
            r"
fn foo() -> u32 {
    $0return 2 + 2$0;
}",
            r"
fn foo() -> u32 {
    return fun_name();
}

fn $0fun_name() -> u32 {
    2 + 2
}",
        );
    }

    #[test]
    fn does_not_add_extra_whitespace() {
        check_assist(
            extract_function,
            r"
fn foo() -> u32 {


    $0return 2 + 2$0;
}",
            r"
fn foo() -> u32 {


    return fun_name();
}

fn $0fun_name() -> u32 {
    2 + 2
}",
        );
    }

    #[test]
    fn break_stmt() {
        check_assist(
            extract_function,
            r"
fn main() {
    let result = loop {
        $0break 2 + 2$0;
    };
}",
            r"
fn main() {
    let result = loop {
        break fun_name();
    };
}

fn $0fun_name() -> i32 {
    2 + 2
}",
        );
    }

    #[test]
    fn extract_cast() {
        check_assist(
            extract_function,
            r"
fn main() {
    let v = $00f32 as u32$0;
}",
            r"
fn main() {
    let v = fun_name();
}

fn $0fun_name() -> u32 {
    0f32 as u32
}",
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
            r"
struct S;

impl S {
    fn foo(&self) -> i32 {
        $01+1$0
    }
}",
            r"
struct S;

impl S {
    fn foo(&self) -> i32 {
        fun_name()
    }
}

fn $0fun_name() -> i32 {
    1+1
}",
        );
    }

    #[test]
    fn method_with_reference() {
        check_assist(
            extract_function,
            r"
struct S { f: i32 };

impl S {
    fn foo(&self) -> i32 {
        $01+self.f$0
    }
}",
            r"
struct S { f: i32 };

impl S {
    fn foo(&self) -> i32 {
        self.fun_name()
    }

    fn $0fun_name(&self) -> i32 {
        1+self.f
    }
}",
        );
    }

    #[test]
    fn method_with_mut() {
        check_assist(
            extract_function,
            r"
struct S { f: i32 };

impl S {
    fn foo(&mut self) {
        $0self.f += 1;$0
    }
}",
            r"
struct S { f: i32 };

impl S {
    fn foo(&mut self) {
        self.fun_name();
    }

    fn $0fun_name(&mut self) {
        self.f += 1;
    }
}",
        );
    }

    #[test]
    fn variable_defined_inside_and_used_after_no_ret() {
        check_assist(
            extract_function,
            r"
fn foo() {
    let n = 1;
    $0let k = n * n;$0
    let m = k + 1;
}",
            r"
fn foo() {
    let n = 1;
    let k = fun_name(n);
    let m = k + 1;
}

fn $0fun_name(n: i32) -> i32 {
    let k = n * n;
    k
}",
        );
    }

    #[test]
    fn two_variables_defined_inside_and_used_after_no_ret() {
        check_assist(
            extract_function,
            r"
fn foo() {
    let n = 1;
    $0let k = n * n;
    let m = k + 2;$0
    let h = k + m;
}",
            r"
fn foo() {
    let n = 1;
    let (k, m) = fun_name(n);
    let h = k + m;
}

fn $0fun_name(n: i32) -> (i32, i32) {
    let k = n * n;
    let m = k + 2;
    (k, m)
}",
        );
    }

    #[test]
    fn mut_var_from_outer_scope() {
        check_assist(
            extract_function,
            r"
fn foo() {
    let mut n = 1;
    $0n += 1;$0
    let m = n + 1;
}",
            r"
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let m = n + 1;
}

fn $0fun_name(n: &mut i32) {
    *n += 1;
}",
        );
    }

    #[test]
    fn mut_param_many_usages_stmt() {
        check_assist(
            extract_function,
            r"
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
}",
            r"
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
}",
        );
    }

    #[test]
    fn mut_param_many_usages_expr() {
        check_assist(
            extract_function,
            r"
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
}",
            r"
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
    {
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
}",
        );
    }

    #[test]
    fn mut_param_by_value() {
        check_assist(
            extract_function,
            r"
fn foo() {
    let mut n = 1;
    $0n += 1;$0
}",
            r"
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    n += 1;
}",
        );
    }

    #[test]
    fn mut_param_because_of_mut_ref() {
        check_assist(
            extract_function,
            r"
fn foo() {
    let mut n = 1;
    $0let v = &mut n;
    *v += 1;$0
    let k = n;
}",
            r"
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let k = n;
}

fn $0fun_name(n: &mut i32) {
    let v = n;
    *v += 1;
}",
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
}",
            r"
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    let v = &mut n;
    *v += 1;
}",
        );
    }

    #[test]
    fn mut_method_call() {
        check_assist(
            extract_function,
            r"
trait I {
    fn inc(&mut self);
}
impl I for i32 {
    fn inc(&mut self) { *self += 1 }
}
fn foo() {
    let mut n = 1;
    $0n.inc();$0
}",
            r"
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
}",
        );
    }

    #[test]
    fn shared_method_call() {
        check_assist(
            extract_function,
            r"
trait I {
    fn succ(&self);
}
impl I for i32 {
    fn succ(&self) { *self + 1 }
}
fn foo() {
    let mut n = 1;
    $0n.succ();$0
}",
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
}",
        );
    }

    #[test]
    fn mut_method_call_with_other_receiver() {
        check_assist(
            extract_function,
            r"
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
}",
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
}",
        );
    }
}

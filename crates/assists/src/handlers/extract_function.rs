use either::Either;
use hir::{HirDisplay, Local};
use ide_db::defs::{Definition, NameRefClass};
use rustc_hash::FxHashSet;
use stdx::format_to;
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        AstNode, NameOwner,
    },
    Direction, SyntaxElement,
    SyntaxKind::{self, BLOCK_EXPR, BREAK_EXPR, COMMENT, PATH_EXPR, RETURN_EXPR},
    SyntaxNode, TextRange,
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

    let mut self_param = None;
    let mut param_pats: Vec<_> = local_variables(&body, &ctx)
        .into_iter()
        .map(|node| node.source(ctx.db()))
        .filter(|src| {
            src.file_id.original_file(ctx.db()) == ctx.frange.file_id
                && !body.contains_node(&either_syntax(&src.value))
        })
        .filter_map(|src| match src.value {
            Either::Left(pat) => Some(pat),
            Either::Right(it) => {
                // we filter self param, as there can only be one
                self_param = Some(it);
                None
            }
        })
        .collect();
    deduplicate_params(&mut param_pats);

    let anchor = if self_param.is_some() { Anchor::Method } else { Anchor::Freestanding };
    let insert_after = body.scope_for_fn_insertion(anchor)?;
    let module = ctx.sema.scope(&insert_after).module()?;

    let params = param_pats
        .into_iter()
        .map(|pat| {
            let ty = pat
                .pat()
                .and_then(|pat| ctx.sema.type_of_pat(&pat))
                .and_then(|ty| ty.display_source_code(ctx.db(), module.into()).ok())
                .unwrap_or_else(|| "()".to_string());

            let name = pat.name().unwrap().to_string();

            Param { name, ty }
        })
        .collect::<Vec<_>>();

    let self_param =
        if let Some(self_param) = self_param { Some(self_param.to_string()) } else { None };

    let expr = body.tail_expr();
    let ret_ty = match expr {
        Some(expr) => {
            // TODO: can we do assist when type is unknown?
            //       We can insert something like `-> ()`
            let ty = ctx.sema.type_of_expr(&expr)?;
            Some(ty.display_source_code(ctx.db(), module.into()).ok()?)
        }
        None => None,
    };

    let target_range = match &body {
        FunctionBody::Expr(expr) => expr.syntax().text_range(),
        FunctionBody::Span { .. } => ctx.frange.range,
    };

    acc.add(
        AssistId("extract_function", crate::AssistKind::RefactorExtract),
        "Extract into function",
        target_range,
        move |builder| {
            let fun = Function { name: "fun_name".to_string(), self_param, params, ret_ty, body };

            builder.replace(target_range, format_replacement(&fun));

            let indent = IndentLevel::from_node(&insert_after);

            let fn_def = format_function(&fun, indent);
            let insert_offset = insert_after.text_range().end();
            builder.insert(insert_offset, fn_def);
        },
    )
}

fn format_replacement(fun: &Function) -> String {
    let mut buf = String::new();
    if fun.self_param.is_some() {
        format_to!(buf, "self.");
    }
    format_to!(buf, "{}(", fun.name);
    {
        let mut it = fun.params.iter();
        if let Some(param) = it.next() {
            format_to!(buf, "{}", param.name);
        }
        for param in it {
            format_to!(buf, ", {}", param.name);
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
    self_param: Option<String>,
    params: Vec<Param>,
    ret_ty: Option<String>,
    body: FunctionBody,
}

impl Function {
    fn has_unit_ret(&self) -> bool {
        match &self.ret_ty {
            Some(ty) => ty == "()",
            None => true,
        }
    }
}

#[derive(Debug)]
struct Param {
    name: String,
    ty: String,
}

fn format_function(fun: &Function, indent: IndentLevel) -> String {
    let mut fn_def = String::new();
    format_to!(fn_def, "\n\n{}fn $0{}(", indent, fun.name);
    {
        let mut it = fun.params.iter();
        if let Some(self_param) = &fun.self_param {
            format_to!(fn_def, "{}", self_param);
        } else if let Some(param) = it.next() {
            format_to!(fn_def, "{}: {}", param.name, param.ty);
        }
        for param in it {
            format_to!(fn_def, ", {}: {}", param.name, param.ty);
        }
    }

    format_to!(fn_def, ")");
    if !fun.has_unit_ret() {
        if let Some(ty) = &fun.ret_ty {
            format_to!(fn_def, " -> {}", ty);
        }
    }
    format_to!(fn_def, " {{");

    match &fun.body {
        FunctionBody::Expr(expr) => {
            fn_def.push('\n');
            let expr = expr.indent(indent);
            format_to!(fn_def, "{}{}", indent + 1, expr.syntax());
            fn_def.push('\n');
        }
        FunctionBody::Span { elements, leading_indent } => {
            format_to!(fn_def, "{}", leading_indent);
            for e in elements {
                format_to!(fn_def, "{}", e);
            }
            if !fn_def.ends_with('\n') {
                fn_def.push('\n');
            }
        }
    }
    format_to!(fn_def, "{}}}", indent);

    fn_def
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

    fn contains_node(&self, node: &SyntaxNode) -> bool {
        fn is_node(body: &FunctionBody, n: &SyntaxNode) -> bool {
            match body {
                FunctionBody::Expr(expr) => n == expr.syntax(),
                FunctionBody::Span { elements, .. } => {
                    // FIXME: can it be quadratic?
                    elements.iter().filter_map(SyntaxElement::as_node).any(|e| e == n)
                }
            }
        }

        node.ancestors().any(|a| is_node(self, &a))
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

fn deduplicate_params(params: &mut Vec<ast::IdentPat>) {
    let mut seen_params = FxHashSet::default();
    params.retain(|p| seen_params.insert(p.clone()));
}

fn either_syntax(value: &Either<ast::IdentPat, ast::SelfParam>) -> &SyntaxNode {
    match value {
        Either::Left(pat) => pat.syntax(),
        Either::Right(it) => it.syntax(),
    }
}

/// Returns a vector of local variables that are refferenced in `body`
fn local_variables(body: &FunctionBody, ctx: &AssistContext) -> Vec<Local> {
    body.descendants()
        .filter_map(ast::NameRef::cast)
        .filter_map(|name_ref| NameRefClass::classify(&ctx.sema, &name_ref))
        .map(|name_kind| name_kind.referenced(ctx.db()))
        .filter_map(|definition| match definition {
            Definition::Local(local) => Some(local),
            _ => None,
        })
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
    fn method_with_mut_downgrade_to_shared() {
        check_assist(
            extract_function,
            r"
struct S { f: i32 };

impl S {
    fn foo(&mut self) -> i32 {
        $01+self.f$0
    }
}",
            r"
struct S { f: i32 };

impl S {
    fn foo(&mut self) -> i32 {
        self.fun_name()
    }

    fn $0fun_name(&self) -> i32 {
        1+self.f
    }
}",
        );
    }
}

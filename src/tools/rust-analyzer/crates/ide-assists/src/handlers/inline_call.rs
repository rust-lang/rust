use std::collections::BTreeSet;

use ast::make;
use either::Either;
use hir::{db::HirDatabase, PathResolution, Semantics, TypeInfo};
use ide_db::{
    base_db::{FileId, FileRange},
    defs::Definition,
    imports::insert_use::remove_path_if_in_use_stmt,
    path_transform::PathTransform,
    search::{FileReference, SearchScope},
    source_change::SourceChangeBuilder,
    syntax_helpers::{insert_whitespace_into_node::insert_ws_into, node_ext::expr_as_name_ref},
    RootDatabase,
};
use itertools::{izip, Itertools};
use syntax::{
    ast::{self, edit_in_place::Indent, HasArgList, PathExpr},
    ted, AstNode, NodeOrToken, SyntaxKind,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: inline_into_callers
//
// Inline a function or method body into all of its callers where possible, creating a `let` statement per parameter
// unless the parameter can be inlined. The parameter will be inlined either if it the supplied argument is a simple local
// or if the parameter is only accessed inside the function body once.
// If all calls can be inlined the function will be removed.
//
// ```
// fn print(_: &str) {}
// fn foo$0(word: &str) {
//     if !word.is_empty() {
//         print(word);
//     }
// }
// fn bar() {
//     foo("안녕하세요");
//     foo("여러분");
// }
// ```
// ->
// ```
// fn print(_: &str) {}
//
// fn bar() {
//     {
//         let word = "안녕하세요";
//         if !word.is_empty() {
//             print(word);
//         }
//     };
//     {
//         let word = "여러분";
//         if !word.is_empty() {
//             print(word);
//         }
//     };
// }
// ```
pub(crate) fn inline_into_callers(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let def_file = ctx.file_id();
    let name = ctx.find_node_at_offset::<ast::Name>()?;
    let ast_func = name.syntax().parent().and_then(ast::Fn::cast)?;
    let func_body = ast_func.body()?;
    let param_list = ast_func.param_list()?;

    let function = ctx.sema.to_def(&ast_func)?;

    let params = get_fn_params(ctx.sema.db, function, &param_list)?;

    let usages = Definition::Function(function).usages(&ctx.sema);
    if !usages.at_least_one() {
        return None;
    }

    let is_recursive_fn = usages
        .clone()
        .in_scope(SearchScope::file_range(FileRange {
            file_id: def_file,
            range: func_body.syntax().text_range(),
        }))
        .at_least_one();
    if is_recursive_fn {
        cov_mark::hit!(inline_into_callers_recursive);
        return None;
    }

    acc.add(
        AssistId("inline_into_callers", AssistKind::RefactorInline),
        "Inline into all callers",
        name.syntax().text_range(),
        |builder| {
            let mut usages = usages.all();
            let current_file_usage = usages.references.remove(&def_file);

            let mut remove_def = true;
            let mut inline_refs_for_file = |file_id, refs: Vec<FileReference>| {
                builder.edit_file(file_id);
                let count = refs.len();
                // The collects are required as we are otherwise iterating while mutating 🙅‍♀️🙅‍♂️
                let (name_refs, name_refs_use) = split_refs_and_uses(builder, refs, Some);
                let call_infos: Vec<_> = name_refs
                    .into_iter()
                    .filter_map(CallInfo::from_name_ref)
                    .map(|call_info| {
                        let mut_node = builder.make_syntax_mut(call_info.node.syntax().clone());
                        (call_info, mut_node)
                    })
                    .collect();
                let replaced = call_infos
                    .into_iter()
                    .map(|(call_info, mut_node)| {
                        let replacement =
                            inline(&ctx.sema, def_file, function, &func_body, &params, &call_info);
                        ted::replace(mut_node, replacement.syntax());
                    })
                    .count();
                if replaced + name_refs_use.len() == count {
                    // we replaced all usages in this file, so we can remove the imports
                    name_refs_use.iter().for_each(remove_path_if_in_use_stmt);
                } else {
                    remove_def = false;
                }
            };
            for (file_id, refs) in usages.into_iter() {
                inline_refs_for_file(file_id, refs);
            }
            match current_file_usage {
                Some(refs) => inline_refs_for_file(def_file, refs),
                None => builder.edit_file(def_file),
            }
            if remove_def {
                builder.delete(ast_func.syntax().text_range());
            }
        },
    )
}

pub(super) fn split_refs_and_uses<T: ast::AstNode>(
    builder: &mut SourceChangeBuilder,
    iter: impl IntoIterator<Item = FileReference>,
    mut map_ref: impl FnMut(ast::NameRef) -> Option<T>,
) -> (Vec<T>, Vec<ast::Path>) {
    iter.into_iter()
        .filter_map(|file_ref| match file_ref.name {
            ast::NameLike::NameRef(name_ref) => Some(name_ref),
            _ => None,
        })
        .filter_map(|name_ref| match name_ref.syntax().ancestors().find_map(ast::UseTree::cast) {
            Some(use_tree) => builder.make_mut(use_tree).path().map(Either::Right),
            None => map_ref(name_ref).map(Either::Left),
        })
        .partition_map(|either| either)
}

// Assist: inline_call
//
// Inlines a function or method body creating a `let` statement per parameter unless the parameter
// can be inlined. The parameter will be inlined either if it the supplied argument is a simple local
// or if the parameter is only accessed inside the function body once.
//
// ```
// # //- minicore: option
// fn foo(name: Option<&str>) {
//     let name = name.unwrap$0();
// }
// ```
// ->
// ```
// fn foo(name: Option<&str>) {
//     let name = match name {
//             Some(val) => val,
//             None => panic!("called `Option::unwrap()` on a `None` value"),
//         };
// }
// ```
pub(crate) fn inline_call(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let name_ref: ast::NameRef = ctx.find_node_at_offset()?;
    let call_info = CallInfo::from_name_ref(name_ref.clone())?;
    let (function, label) = match &call_info.node {
        ast::CallableExpr::Call(call) => {
            let path = match call.expr()? {
                ast::Expr::PathExpr(path) => path.path(),
                _ => None,
            }?;
            let function = match ctx.sema.resolve_path(&path)? {
                PathResolution::Def(hir::ModuleDef::Function(f)) => f,
                _ => return None,
            };
            (function, format!("Inline `{path}`"))
        }
        ast::CallableExpr::MethodCall(call) => {
            (ctx.sema.resolve_method_call(call)?, format!("Inline `{name_ref}`"))
        }
    };

    let fn_source = ctx.sema.source(function)?;
    let fn_body = fn_source.value.body()?;
    let param_list = fn_source.value.param_list()?;

    let FileRange { file_id, range } = fn_source.syntax().original_file_range(ctx.sema.db);
    if file_id == ctx.file_id() && range.contains(ctx.offset()) {
        cov_mark::hit!(inline_call_recursive);
        return None;
    }
    let params = get_fn_params(ctx.sema.db, function, &param_list)?;

    if call_info.arguments.len() != params.len() {
        // Can't inline the function because they've passed the wrong number of
        // arguments to this function
        cov_mark::hit!(inline_call_incorrect_number_of_arguments);
        return None;
    }

    let syntax = call_info.node.syntax().clone();
    acc.add(
        AssistId("inline_call", AssistKind::RefactorInline),
        label,
        syntax.text_range(),
        |builder| {
            let replacement = inline(&ctx.sema, file_id, function, &fn_body, &params, &call_info);

            builder.replace_ast(
                match call_info.node {
                    ast::CallableExpr::Call(it) => ast::Expr::CallExpr(it),
                    ast::CallableExpr::MethodCall(it) => ast::Expr::MethodCallExpr(it),
                },
                replacement,
            );
        },
    )
}

struct CallInfo {
    node: ast::CallableExpr,
    arguments: Vec<ast::Expr>,
    generic_arg_list: Option<ast::GenericArgList>,
}

impl CallInfo {
    fn from_name_ref(name_ref: ast::NameRef) -> Option<CallInfo> {
        let parent = name_ref.syntax().parent()?;
        if let Some(call) = ast::MethodCallExpr::cast(parent.clone()) {
            let receiver = call.receiver()?;
            let mut arguments = vec![receiver];
            arguments.extend(call.arg_list()?.args());
            Some(CallInfo {
                generic_arg_list: call.generic_arg_list(),
                node: ast::CallableExpr::MethodCall(call),
                arguments,
            })
        } else if let Some(segment) = ast::PathSegment::cast(parent) {
            let path = segment.syntax().parent().and_then(ast::Path::cast)?;
            let path = path.syntax().parent().and_then(ast::PathExpr::cast)?;
            let call = path.syntax().parent().and_then(ast::CallExpr::cast)?;

            Some(CallInfo {
                arguments: call.arg_list()?.args().collect(),
                node: ast::CallableExpr::Call(call),
                generic_arg_list: segment.generic_arg_list(),
            })
        } else {
            None
        }
    }
}

fn get_fn_params(
    db: &dyn HirDatabase,
    function: hir::Function,
    param_list: &ast::ParamList,
) -> Option<Vec<(ast::Pat, Option<ast::Type>, hir::Param)>> {
    let mut assoc_fn_params = function.assoc_fn_params(db).into_iter();

    let mut params = Vec::new();
    if let Some(self_param) = param_list.self_param() {
        // FIXME this should depend on the receiver as well as the self_param
        params.push((
            make::ident_pat(
                self_param.amp_token().is_some(),
                self_param.mut_token().is_some(),
                make::name("this"),
            )
            .into(),
            None,
            assoc_fn_params.next()?,
        ));
    }
    for param in param_list.params() {
        params.push((param.pat()?, param.ty(), assoc_fn_params.next()?));
    }

    Some(params)
}

fn inline(
    sema: &Semantics<'_, RootDatabase>,
    function_def_file_id: FileId,
    function: hir::Function,
    fn_body: &ast::BlockExpr,
    params: &[(ast::Pat, Option<ast::Type>, hir::Param)],
    CallInfo { node, arguments, generic_arg_list }: &CallInfo,
) -> ast::Expr {
    let body = if sema.hir_file_for(fn_body.syntax()).is_macro() {
        cov_mark::hit!(inline_call_defined_in_macro);
        if let Some(body) = ast::BlockExpr::cast(insert_ws_into(fn_body.syntax().clone())) {
            body
        } else {
            fn_body.clone_for_update()
        }
    } else {
        fn_body.clone_for_update()
    };
    if let Some(imp) = body.syntax().ancestors().find_map(ast::Impl::cast) {
        if !node.syntax().ancestors().any(|anc| &anc == imp.syntax()) {
            if let Some(t) = imp.self_ty() {
                body.syntax()
                    .descendants_with_tokens()
                    .filter_map(NodeOrToken::into_token)
                    .filter(|tok| tok.kind() == SyntaxKind::SELF_TYPE_KW)
                    .for_each(|tok| ted::replace(tok, t.syntax()));
            }
        }
    }
    let usages_for_locals = |local| {
        Definition::Local(local)
            .usages(sema)
            .all()
            .references
            .remove(&function_def_file_id)
            .unwrap_or_default()
            .into_iter()
    };
    let param_use_nodes: Vec<Vec<_>> = params
        .iter()
        .map(|(pat, _, param)| {
            if !matches!(pat, ast::Pat::IdentPat(pat) if pat.is_simple_ident()) {
                return Vec::new();
            }
            // FIXME: we need to fetch all locals declared in the parameter here
            // not only the local if it is a simple binding
            match param.as_local(sema.db) {
                Some(l) => usages_for_locals(l)
                    .map(|FileReference { name, range, .. }| match name {
                        ast::NameLike::NameRef(_) => body
                            .syntax()
                            .covering_element(range)
                            .ancestors()
                            .nth(3)
                            .and_then(ast::PathExpr::cast),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default(),
                None => Vec::new(),
            }
        })
        .collect();

    if function.self_param(sema.db).is_some() {
        let this = || make::name_ref("this").syntax().clone_for_update().first_token().unwrap();
        if let Some(self_local) = params[0].2.as_local(sema.db) {
            usages_for_locals(self_local)
                .filter_map(|FileReference { name, range, .. }| match name {
                    ast::NameLike::NameRef(_) => Some(body.syntax().covering_element(range)),
                    _ => None,
                })
                .for_each(|it| {
                    ted::replace(it, &this());
                })
        }
    }

    let mut func_let_vars: BTreeSet<String> = BTreeSet::new();

    // grab all of the local variable declarations in the function
    for stmt in fn_body.statements() {
        if let Some(let_stmt) = ast::LetStmt::cast(stmt.syntax().to_owned()) {
            for has_token in let_stmt.syntax().children_with_tokens() {
                if let Some(node) = has_token.as_node() {
                    if let Some(ident_pat) = ast::IdentPat::cast(node.to_owned()) {
                        func_let_vars.insert(ident_pat.syntax().text().to_string());
                    }
                }
            }
        }
    }

    // Inline parameter expressions or generate `let` statements depending on whether inlining works or not.
    for ((pat, param_ty, _), usages, expr) in izip!(params, param_use_nodes, arguments).rev() {
        // izip confuses RA due to our lack of hygiene info currently losing us type info causing incorrect errors
        let usages: &[ast::PathExpr] = &usages;
        let expr: &ast::Expr = expr;

        let insert_let_stmt = || {
            let ty = sema.type_of_expr(expr).filter(TypeInfo::has_adjustment).and(param_ty.clone());
            if let Some(stmt_list) = body.stmt_list() {
                stmt_list.push_front(
                    make::let_stmt(pat.clone(), ty, Some(expr.clone())).clone_for_update().into(),
                )
            }
        };

        // check if there is a local var in the function that conflicts with parameter
        // if it does then emit a let statement and continue
        if func_let_vars.contains(&expr.syntax().text().to_string()) {
            insert_let_stmt();
            continue;
        }

        let inline_direct = |usage, replacement: &ast::Expr| {
            if let Some(field) = path_expr_as_record_field(usage) {
                cov_mark::hit!(inline_call_inline_direct_field);
                field.replace_expr(replacement.clone_for_update());
            } else {
                ted::replace(usage.syntax(), &replacement.syntax().clone_for_update());
            }
        };

        match usages {
            // inline single use closure arguments
            [usage]
                if matches!(expr, ast::Expr::ClosureExpr(_))
                    && usage.syntax().parent().and_then(ast::Expr::cast).is_some() =>
            {
                cov_mark::hit!(inline_call_inline_closure);
                let expr = make::expr_paren(expr.clone());
                inline_direct(usage, &expr);
            }
            // inline single use literals
            [usage] if matches!(expr, ast::Expr::Literal(_)) => {
                cov_mark::hit!(inline_call_inline_literal);
                inline_direct(usage, expr);
            }
            // inline direct local arguments
            [_, ..] if expr_as_name_ref(expr).is_some() => {
                cov_mark::hit!(inline_call_inline_locals);
                usages.iter().for_each(|usage| inline_direct(usage, expr));
            }
            // can't inline, emit a let statement
            _ => {
                insert_let_stmt();
            }
        }
    }

    if let Some(generic_arg_list) = generic_arg_list.clone() {
        if let Some((target, source)) = &sema.scope(node.syntax()).zip(sema.scope(fn_body.syntax()))
        {
            PathTransform::function_call(target, source, function, generic_arg_list)
                .apply(body.syntax());
        }
    }

    let original_indentation = match node {
        ast::CallableExpr::Call(it) => it.indent_level(),
        ast::CallableExpr::MethodCall(it) => it.indent_level(),
    };
    body.reindent_to(original_indentation);

    match body.tail_expr() {
        Some(expr) if body.statements().next().is_none() => expr,
        _ => match node
            .syntax()
            .parent()
            .and_then(ast::BinExpr::cast)
            .and_then(|bin_expr| bin_expr.lhs())
        {
            Some(lhs) if lhs.syntax() == node.syntax() => {
                make::expr_paren(ast::Expr::BlockExpr(body)).clone_for_update()
            }
            _ => ast::Expr::BlockExpr(body),
        },
    }
}

fn path_expr_as_record_field(usage: &PathExpr) -> Option<ast::RecordExprField> {
    let path = usage.path()?;
    let name_ref = path.as_single_name_ref()?;
    ast::RecordExprField::for_name_ref(&name_ref)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn no_args_or_return_value_gets_inlined_without_block() {
        check_assist(
            inline_call,
            r#"
fn foo() { println!("Hello, World!"); }
fn main() {
    fo$0o();
}
"#,
            r#"
fn foo() { println!("Hello, World!"); }
fn main() {
    { println!("Hello, World!"); };
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_incorrect_number_of_parameters_are_provided() {
        cov_mark::check!(inline_call_incorrect_number_of_arguments);
        check_assist_not_applicable(
            inline_call,
            r#"
fn add(a: u32, b: u32) -> u32 { a + b }
fn main() { let x = add$0(42); }
"#,
        );
    }

    #[test]
    fn args_with_side_effects() {
        check_assist(
            inline_call,
            r#"
fn foo(name: String) {
    println!("Hello, {}!", name);
}
fn main() {
    foo$0(String::from("Michael"));
}
"#,
            r#"
fn foo(name: String) {
    println!("Hello, {}!", name);
}
fn main() {
    {
        let name = String::from("Michael");
        println!("Hello, {}!", name);
    };
}
"#,
        );
    }

    #[test]
    fn function_with_multiple_statements() {
        check_assist(
            inline_call,
            r#"
fn foo(a: u32, b: u32) -> u32 {
    let x = a + b;
    let y = x - b;
    x * y
}

fn main() {
    let x = foo$0(1, 2);
}
"#,
            r#"
fn foo(a: u32, b: u32) -> u32 {
    let x = a + b;
    let y = x - b;
    x * y
}

fn main() {
    let x = {
        let b = 2;
        let x = 1 + b;
        let y = x - b;
        x * y
    };
}
"#,
        );
    }

    #[test]
    fn function_with_self_param() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = Foo::add$0(Foo(3), 2);
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = {
        let this = Foo(3);
        Foo(this.0 + 2)
    };
}
"#,
        );
    }

    #[test]
    fn method_by_val() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = Foo(3).add$0(2);
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = {
        let this = Foo(3);
        Foo(this.0 + 2)
    };
}
"#,
        );
    }

    #[test]
    fn method_by_ref() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn add(&self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = Foo(3).add$0(2);
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn add(&self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = {
        let ref this = Foo(3);
        Foo(this.0 + 2)
    };
}
"#,
        );
    }

    #[test]
    fn generic_method_by_ref() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn add<T>(&self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = Foo(3).add$0::<usize>(2);
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn add<T>(&self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = {
        let ref this = Foo(3);
        Foo(this.0 + 2)
    };
}
"#,
        );
    }

    #[test]
    fn method_by_ref_mut() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn clear(&mut self) {
        self.0 = 0;
    }
}

fn main() {
    let mut foo = Foo(3);
    foo.clear$0();
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn clear(&mut self) {
        self.0 = 0;
    }
}

fn main() {
    let mut foo = Foo(3);
    {
        let ref mut this = foo;
        this.0 = 0;
    };
}
"#,
        );
    }

    #[test]
    fn function_multi_use_expr_in_param() {
        check_assist(
            inline_call,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let x = 51;
    let y = square$0(10 + x);
}
"#,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let x = 51;
    let y = {
        let x = 10 + x;
        x * x
    };
}
"#,
        );
    }

    #[test]
    fn function_use_local_in_param() {
        cov_mark::check!(inline_call_inline_locals);
        check_assist(
            inline_call,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let local = 51;
    let y = square$0(local);
}
"#,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let local = 51;
    let y = local * local;
}
"#,
        );
    }

    #[test]
    fn method_in_impl() {
        check_assist(
            inline_call,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {
        self;
        self;
    }
    fn bar(&self) {
        self.foo$0();
    }
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {
        self;
        self;
    }
    fn bar(&self) {
        {
            let ref this = self;
            this;
            this;
        };
    }
}
"#,
        );
    }

    #[test]
    fn wraps_closure_in_paren() {
        cov_mark::check!(inline_call_inline_closure);
        check_assist(
            inline_call,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    foo$0(|| {})
}
"#,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    {
        (|| {})();
    }
}
"#,
        );
        check_assist(
            inline_call,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    foo$0(main)
}
"#,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    {
        main();
    }
}
"#,
        );
    }

    #[test]
    fn inline_single_literal_expr() {
        cov_mark::check!(inline_call_inline_literal);
        check_assist(
            inline_call,
            r#"
fn foo(x: u32) -> u32{
    x
}

fn main() {
    foo$0(222);
}
"#,
            r#"
fn foo(x: u32) -> u32{
    x
}

fn main() {
    222;
}
"#,
        );
    }

    #[test]
    fn inline_emits_type_for_coercion() {
        check_assist(
            inline_call,
            r#"
fn foo(x: *const u32) -> u32 {
    x as u32
}

fn main() {
    foo$0(&222);
}
"#,
            r#"
fn foo(x: *const u32) -> u32 {
    x as u32
}

fn main() {
    {
        let x: *const u32 = &222;
        x as u32
    };
}
"#,
        );
    }

    // FIXME: const generics aren't being substituted, this is blocked on better support for them
    #[test]
    fn inline_substitutes_generics() {
        check_assist(
            inline_call,
            r#"
fn foo<T, const N: usize>() {
    bar::<T, N>()
}

fn bar<U, const M: usize>() {}

fn main() {
    foo$0::<usize, {0}>();
}
"#,
            r#"
fn foo<T, const N: usize>() {
    bar::<T, N>()
}

fn bar<U, const M: usize>() {}

fn main() {
    bar::<usize, N>();
}
"#,
        );
    }

    #[test]
    fn inline_callers() {
        check_assist(
            inline_into_callers,
            r#"
fn do_the_math$0(b: u32) -> u32 {
    let foo = 10;
    foo * b + foo
}
fn foo() {
    do_the_math(0);
    let bar = 10;
    do_the_math(bar);
}
"#,
            r#"

fn foo() {
    {
        let foo = 10;
        foo * 0 + foo
    };
    let bar = 10;
    {
        let foo = 10;
        foo * bar + foo
    };
}
"#,
        );
    }

    #[test]
    fn inline_callers_across_files() {
        check_assist(
            inline_into_callers,
            r#"
//- /lib.rs
mod foo;
fn do_the_math$0(b: u32) -> u32 {
    let foo = 10;
    foo * b + foo
}
//- /foo.rs
use super::do_the_math;
fn foo() {
    do_the_math(0);
    let bar = 10;
    do_the_math(bar);
}
"#,
            r#"
//- /lib.rs
mod foo;

//- /foo.rs
fn foo() {
    {
        let foo = 10;
        foo * 0 + foo
    };
    let bar = 10;
    {
        let foo = 10;
        foo * bar + foo
    };
}
"#,
        );
    }

    #[test]
    fn inline_callers_across_files_with_def_file() {
        check_assist(
            inline_into_callers,
            r#"
//- /lib.rs
mod foo;
fn do_the_math$0(b: u32) -> u32 {
    let foo = 10;
    foo * b + foo
}
fn bar(a: u32, b: u32) -> u32 {
    do_the_math(0);
}
//- /foo.rs
use super::do_the_math;
fn foo() {
    do_the_math(0);
}
"#,
            r#"
//- /lib.rs
mod foo;

fn bar(a: u32, b: u32) -> u32 {
    {
        let foo = 10;
        foo * 0 + foo
    };
}
//- /foo.rs
fn foo() {
    {
        let foo = 10;
        foo * 0 + foo
    };
}
"#,
        );
    }

    #[test]
    fn inline_callers_recursive() {
        cov_mark::check!(inline_into_callers_recursive);
        check_assist_not_applicable(
            inline_into_callers,
            r#"
fn foo$0() {
    foo();
}
"#,
        );
    }

    #[test]
    fn inline_call_recursive() {
        cov_mark::check!(inline_call_recursive);
        check_assist_not_applicable(
            inline_call,
            r#"
fn foo() {
    foo$0();
}
"#,
        );
    }

    #[test]
    fn inline_call_field_shorthand() {
        cov_mark::check!(inline_call_inline_direct_field);
        check_assist(
            inline_call,
            r#"
struct Foo {
    field: u32,
    field1: u32,
    field2: u32,
    field3: u32,
}
fn foo(field: u32, field1: u32, val2: u32, val3: u32) -> Foo {
    Foo {
        field,
        field1,
        field2: val2,
        field3: val3,
    }
}
fn main() {
    let bar = 0;
    let baz = 0;
    foo$0(bar, 0, baz, 0);
}
"#,
            r#"
struct Foo {
    field: u32,
    field1: u32,
    field2: u32,
    field3: u32,
}
fn foo(field: u32, field1: u32, val2: u32, val3: u32) -> Foo {
    Foo {
        field,
        field1,
        field2: val2,
        field3: val3,
    }
}
fn main() {
    let bar = 0;
    let baz = 0;
    Foo {
            field: bar,
            field1: 0,
            field2: baz,
            field3: 0,
        };
}
"#,
        );
    }

    #[test]
    fn inline_callers_wrapped_in_parentheses() {
        check_assist(
            inline_into_callers,
            r#"
fn foo$0() -> u32 {
    let x = 0;
    x
}
fn bar() -> u32 {
    foo() + foo()
}
"#,
            r#"

fn bar() -> u32 {
    ({
        let x = 0;
        x
    }) + {
        let x = 0;
        x
    }
}
"#,
        )
    }

    #[test]
    fn inline_call_wrapped_in_parentheses() {
        check_assist(
            inline_call,
            r#"
fn foo() -> u32 {
    let x = 0;
    x
}
fn bar() -> u32 {
    foo$0() + foo()
}
"#,
            r#"
fn foo() -> u32 {
    let x = 0;
    x
}
fn bar() -> u32 {
    ({
        let x = 0;
        x
    }) + foo()
}
"#,
        )
    }

    #[test]
    fn inline_call_defined_in_macro() {
        cov_mark::check!(inline_call_defined_in_macro);
        check_assist(
            inline_call,
            r#"
macro_rules! define_foo {
    () => { fn foo() -> u32 {
        let x = 0;
        x
    } };
}
define_foo!();
fn bar() -> u32 {
    foo$0()
}
"#,
            r#"
macro_rules! define_foo {
    () => { fn foo() -> u32 {
        let x = 0;
        x
    } };
}
define_foo!();
fn bar() -> u32 {
    {
      let x = 0;
      x
    }
}
"#,
        )
    }

    #[test]
    fn inline_call_with_self_type() {
        check_assist(
            inline_call,
            r#"
struct A(u32);
impl A {
    fn f() -> Self { Self(114514) }
}
fn main() {
    A::f$0();
}
"#,
            r#"
struct A(u32);
impl A {
    fn f() -> Self { Self(114514) }
}
fn main() {
    A(114514);
}
"#,
        )
    }

    #[test]
    fn inline_call_with_self_type_but_within_same_impl() {
        check_assist(
            inline_call,
            r#"
struct A(u32);
impl A {
    fn f() -> Self { Self(1919810) }
    fn main() {
        Self::f$0();
    }
}
"#,
            r#"
struct A(u32);
impl A {
    fn f() -> Self { Self(1919810) }
    fn main() {
        Self(1919810);
    }
}
"#,
        )
    }

    #[test]
    fn local_variable_shadowing_callers_argument() {
        check_assist(
            inline_call,
            r#"
fn foo(bar: u32, baz: u32) -> u32 {
    let a = 1;
    bar * baz * a * 6
}
fn main() {
    let a = 7;
    let b = 1;
    let res = foo$0(a, b);
}
"#,
            r#"
fn foo(bar: u32, baz: u32) -> u32 {
    let a = 1;
    bar * baz * a * 6
}
fn main() {
    let a = 7;
    let b = 1;
    let res = {
        let bar = a;
        let a = 1;
        bar * b * a * 6
    };
}
"#,
        );
    }
}

//! This module contains functions to suggest names for expressions, functions and other items

use hir::Semantics;
use ide_db::RootDatabase;
use itertools::Itertools;
use stdx::to_lower_snake_case;
use syntax::{
    ast::{self, NameOwner},
    match_ast, AstNode,
};

/// Trait names, that will be ignored when in `impl Trait` and `dyn Trait`
const USELESS_TRAITS: &[&str] = &["Send", "Sync", "Copy", "Clone", "Eq", "PartialEq"];
/// Identifier names that won't be suggested, ever
///
/// **NOTE**: they all must be snake lower case
const USELESS_NAMES: &[&str] =
    &["new", "default", "option", "some", "none", "ok", "err", "str", "string"];
/// Generic types replaced by their first argument
///
/// # Examples
/// `Option<Name>` -> `Name`
/// `Result<User, Error>` -> `User`
const WRAPPER_TYPES: &[&str] = &["Box", "Option", "Result"];
/// Prefixes to strip from methods names
///
/// # Examples
/// `vec.as_slice()` -> `slice`
/// `args.into_config()` -> `config`
/// `bytes.to_vec()` -> `vec`
const USELESS_METHOD_PREFIXES: &[&str] = &["into_", "as_", "to_"];
/// Useless methods that are stripped from expression
///
/// # Examples
/// `var.name().to_string()` -> `var.name()`
const USELESS_METHODS: &[&str] = &[
    "to_string",
    "as_str",
    "to_owned",
    "as_ref",
    "clone",
    "cloned",
    "expect",
    "expect_none",
    "unwrap",
    "unwrap_none",
    "unwrap_or",
    "unwrap_or_default",
    "unwrap_or_else",
    "unwrap_unchecked",
    "iter",
    "into_iter",
    "iter_mut",
];

/// Suggest name of variable for given expression
///
/// **NOTE**: it is caller's responsibility to guarantee uniqueness of the name.
/// I.e. it doesn't look for names in scope.
///
/// # Current implementation
///
/// In current implementation, the function tries to get the name from
/// the following sources:
///
/// * if expr is an argument to function/method, use paramter name
/// * if expr is a function/method call, use function name
/// * expression type name if it exists (E.g. `()`, `fn() -> ()` or `!` do not have names)
/// * fallback: `var_name`
///
/// It also applies heuristics to filter out less informative names
///
/// Currently it sticks to the first name found.
pub(crate) fn variable(expr: &ast::Expr, sema: &Semantics<'_, RootDatabase>) -> String {
    // `from_param` does not benifit from stripping
    // it need the largest context possible
    // so we check firstmost
    if let Some(name) = from_param(expr, sema) {
        return name;
    }

    let mut next_expr = Some(expr.clone());
    while let Some(expr) = next_expr {
        let name = from_call(&expr).or_else(|| from_type(&expr, sema));
        if let Some(name) = name {
            return name;
        }

        match expr {
            ast::Expr::RefExpr(inner) => next_expr = inner.expr(),
            ast::Expr::BoxExpr(inner) => next_expr = inner.expr(),
            ast::Expr::AwaitExpr(inner) => next_expr = inner.expr(),
            // ast::Expr::BlockExpr(block) => expr = block.tail_expr(),
            ast::Expr::CastExpr(inner) => next_expr = inner.expr(),
            ast::Expr::MethodCallExpr(method) if is_useless_method(&method) => {
                next_expr = method.receiver();
            }
            ast::Expr::ParenExpr(inner) => next_expr = inner.expr(),
            ast::Expr::TryExpr(inner) => next_expr = inner.expr(),
            ast::Expr::PrefixExpr(prefix) if prefix.op_kind() == Some(ast::PrefixOp::Deref) => {
                next_expr = prefix.expr()
            }
            _ => break,
        }
    }

    "var_name".to_string()
}

fn normalize(name: &str) -> Option<String> {
    let name = to_lower_snake_case(name);

    if USELESS_NAMES.contains(&name.as_str()) {
        return None;
    }

    if !is_valid_name(&name) {
        return None;
    }

    Some(name)
}

fn is_valid_name(name: &str) -> bool {
    match syntax::lex_single_syntax_kind(name) {
        Some((syntax::SyntaxKind::IDENT, _error)) => true,
        _ => false,
    }
}

fn is_useless_method(method: &ast::MethodCallExpr) -> bool {
    let ident = method.name_ref().and_then(|it| it.ident_token());

    if let Some(ident) = ident {
        USELESS_METHODS.contains(&ident.text())
    } else {
        false
    }
}

fn from_call(expr: &ast::Expr) -> Option<String> {
    from_func_call(expr).or_else(|| from_method_call(expr))
}

fn from_func_call(expr: &ast::Expr) -> Option<String> {
    let call = match expr {
        ast::Expr::CallExpr(call) => call,
        _ => return None,
    };
    let func = match call.expr()? {
        ast::Expr::PathExpr(path) => path,
        _ => return None,
    };
    let ident = func.path()?.segment()?.name_ref()?.ident_token()?;
    normalize(ident.text())
}

fn from_method_call(expr: &ast::Expr) -> Option<String> {
    let method = match expr {
        ast::Expr::MethodCallExpr(call) => call,
        _ => return None,
    };
    let ident = method.name_ref()?.ident_token()?;
    let mut name = ident.text();

    if USELESS_METHODS.contains(&name) {
        return None;
    }

    for prefix in USELESS_METHOD_PREFIXES {
        if let Some(suffix) = name.strip_prefix(prefix) {
            name = suffix;
            break;
        }
    }

    normalize(&name)
}

fn from_param(expr: &ast::Expr, sema: &Semantics<'_, RootDatabase>) -> Option<String> {
    let arg_list = expr.syntax().parent().and_then(ast::ArgList::cast)?;
    let args_parent = arg_list.syntax().parent()?;
    let func = match_ast! {
        match args_parent {
            ast::CallExpr(call) => {
                let func = call.expr()?;
                let func_ty = sema.type_of_expr(&func)?;
                func_ty.as_callable(sema.db)?
            },
            ast::MethodCallExpr(method) => sema.resolve_method_call_as_callable(&method)?,
            _ => return None,
        }
    };

    let (idx, _) = arg_list.args().find_position(|it| it == expr).unwrap();
    let (pat, _) = func.params(sema.db).into_iter().nth(idx)?;
    let pat = match pat? {
        either::Either::Right(pat) => pat,
        _ => return None,
    };
    let name = var_name_from_pat(&pat)?;
    normalize(&name.to_string())
}

fn var_name_from_pat(pat: &ast::Pat) -> Option<ast::Name> {
    match pat {
        ast::Pat::IdentPat(var) => var.name(),
        ast::Pat::RefPat(ref_pat) => var_name_from_pat(&ref_pat.pat()?),
        ast::Pat::BoxPat(box_pat) => var_name_from_pat(&box_pat.pat()?),
        _ => None,
    }
}

fn from_type(expr: &ast::Expr, sema: &Semantics<'_, RootDatabase>) -> Option<String> {
    let ty = sema.type_of_expr(expr)?;
    let ty = ty.remove_ref().unwrap_or(ty);

    name_of_type(&ty, sema.db)
}

fn name_of_type(ty: &hir::Type, db: &RootDatabase) -> Option<String> {
    let name = if let Some(adt) = ty.as_adt() {
        let name = adt.name(db).to_string();

        if WRAPPER_TYPES.contains(&name.as_str()) {
            let inner_ty = ty.type_parameters().next()?;
            return name_of_type(&inner_ty, db);
        }

        name
    } else if let Some(trait_) = ty.as_dyn_trait() {
        trait_name(&trait_, db)?
    } else if let Some(traits) = ty.as_impl_traits(db) {
        let mut iter = traits.into_iter().filter_map(|t| trait_name(&t, db));
        let name = iter.next()?;
        if iter.next().is_some() {
            return None;
        }
        name
    } else {
        return None;
    };
    normalize(&name)
}

fn trait_name(trait_: &hir::Trait, db: &RootDatabase) -> Option<String> {
    let name = trait_.name(db).to_string();
    if USELESS_TRAITS.contains(&name.as_str()) {
        return None;
    }
    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::check_name_suggestion;

    mod from_func_call {
        use super::*;

        #[test]
        fn no_args() {
            check_name_suggestion(
                |e, _| from_func_call(e),
                r#"
                fn foo() {
                    $0bar()$0
                }"#,
                "bar",
            );
        }

        #[test]
        fn single_arg() {
            check_name_suggestion(
                |e, _| from_func_call(e),
                r#"
                fn foo() {
                    $0bar(1)$0
                }"#,
                "bar",
            );
        }

        #[test]
        fn many_args() {
            check_name_suggestion(
                |e, _| from_func_call(e),
                r#"
                fn foo() {
                    $0bar(1, 2, 3)$0
                }"#,
                "bar",
            );
        }

        #[test]
        fn path() {
            check_name_suggestion(
                |e, _| from_func_call(e),
                r#"
                fn foo() {
                    $0i32::bar(1, 2, 3)$0
                }"#,
                "bar",
            );
        }

        #[test]
        fn generic_params() {
            check_name_suggestion(
                |e, _| from_func_call(e),
                r#"
                fn foo() {
                    $0bar::<i32>(1, 2, 3)$0
                }"#,
                "bar",
            );
        }
    }

    mod from_method_call {
        use super::*;

        #[test]
        fn no_args() {
            check_name_suggestion(
                |e, _| from_method_call(e),
                r#"
                fn foo() {
                    $0bar.frobnicate()$0
                }"#,
                "frobnicate",
            );
        }

        #[test]
        fn generic_params() {
            check_name_suggestion(
                |e, _| from_method_call(e),
                r#"
                fn foo() {
                    $0bar.frobnicate::<i32, u32>()$0
                }"#,
                "frobnicate",
            );
        }

        #[test]
        fn to_name() {
            check_name_suggestion(
                |e, _| from_method_call(e),
                r#"
                struct Args;
                struct Config;
                impl Args {
                    fn to_config(&self) -> Config {}
                }
                fn foo() {
                    $0Args.to_config()$0;
                }"#,
                "config",
            );
        }
    }

    mod from_param {
        use crate::tests::check_name_suggestion_not_applicable;

        use super::*;

        #[test]
        fn plain_func() {
            check_name_suggestion(
                from_param,
                r#"
                fn bar(n: i32, m: u32);
                fn foo() {
                    bar($01$0, 2)
                }"#,
                "n",
            );
        }

        #[test]
        fn mut_param() {
            check_name_suggestion(
                from_param,
                r#"
                fn bar(mut n: i32, m: u32);
                fn foo() {
                    bar($01$0, 2)
                }"#,
                "n",
            );
        }

        #[test]
        fn func_does_not_exist() {
            check_name_suggestion_not_applicable(
                from_param,
                r#"
                fn foo() {
                    bar($01$0, 2)
                }"#,
            );
        }

        #[test]
        fn unnamed_param() {
            check_name_suggestion_not_applicable(
                from_param,
                r#"
                fn bar(_: i32, m: u32);
                fn foo() {
                    bar($01$0, 2)
                }"#,
            );
        }

        #[test]
        fn tuple_pat() {
            check_name_suggestion_not_applicable(
                from_param,
                r#"
                fn bar((n, k): (i32, i32), m: u32);
                fn foo() {
                    bar($0(1, 2)$0, 3)
                }"#,
            );
        }

        #[test]
        fn ref_pat() {
            check_name_suggestion(
                from_param,
                r#"
                fn bar(&n: &i32, m: u32);
                fn foo() {
                    bar($0&1$0, 3)
                }"#,
                "n",
            );
        }

        #[test]
        fn box_pat() {
            check_name_suggestion(
                from_param,
                r#"
                fn bar(box n: &i32, m: u32);
                fn foo() {
                    bar($01$0, 3)
                }"#,
                "n",
            );
        }

        #[test]
        fn param_out_of_index() {
            check_name_suggestion_not_applicable(
                from_param,
                r#"
                fn bar(n: i32, m: u32);
                fn foo() {
                    bar(1, 2, $03$0)
                }"#,
            );
        }

        #[test]
        fn generic_param_resolved() {
            check_name_suggestion(
                from_param,
                r#"
                fn bar<T>(n: T, m: u32);
                fn foo() {
                    bar($01$0, 2)
                }"#,
                "n",
            );
        }

        #[test]
        fn generic_param_unresolved() {
            check_name_suggestion(
                from_param,
                r#"
                fn bar<T>(n: T, m: u32);
                fn foo<T>(x: T) {
                    bar($0x$0, 2)
                }"#,
                "n",
            );
        }

        #[test]
        fn method() {
            check_name_suggestion(
                from_param,
                r#"
                struct S;
                impl S {
                    fn bar(&self, n: i32, m: u32);
                }
                fn foo() {
                    S.bar($01$0, 2)
                }"#,
                "n",
            );
        }

        #[test]
        fn method_ufcs() {
            check_name_suggestion(
                from_param,
                r#"
                struct S;
                impl S {
                    fn bar(&self, n: i32, m: u32);
                }
                fn foo() {
                    S::bar(&S, $01$0, 2)
                }"#,
                "n",
            );
        }

        #[test]
        fn method_self() {
            check_name_suggestion_not_applicable(
                from_param,
                r#"
                struct S;
                impl S {
                    fn bar(&self, n: i32, m: u32);
                }
                fn foo() {
                    S::bar($0&S$0, 1, 2)
                }"#,
            );
        }

        #[test]
        fn method_self_named() {
            check_name_suggestion(
                from_param,
                r#"
                struct S;
                impl S {
                    fn bar(strukt: &Self, n: i32, m: u32);
                }
                fn foo() {
                    S::bar($0&S$0, 1, 2)
                }"#,
                "strukt",
            );
        }
    }

    mod from_type {
        use crate::tests::check_name_suggestion_not_applicable;

        use super::*;

        #[test]
        fn i32() {
            check_name_suggestion_not_applicable(
                from_type,
                r#"
                fn foo() {
                    let _: i32 = $01$0;
                }"#,
            );
        }

        #[test]
        fn u64() {
            check_name_suggestion_not_applicable(
                from_type,
                r#"
                fn foo() {
                    let _: u64 = $01$0;
                }"#,
            );
        }

        #[test]
        fn bool() {
            check_name_suggestion_not_applicable(
                from_type,
                r#"
                fn foo() {
                    let _: bool = $0true$0;
                }"#,
            );
        }

        #[test]
        fn struct_unit() {
            check_name_suggestion(
                from_type,
                r#"
                struct Seed;
                fn foo() {
                    let _ = $0Seed$0;
                }"#,
                "seed",
            );
        }

        #[test]
        fn struct_unit_to_snake() {
            check_name_suggestion(
                from_type,
                r#"
                struct SeedState;
                fn foo() {
                    let _ = $0SeedState$0;
                }"#,
                "seed_state",
            );
        }

        #[test]
        fn struct_single_arg() {
            check_name_suggestion(
                from_type,
                r#"
                struct Seed(u32);
                fn foo() {
                    let _ = $0Seed(0)$0;
                }"#,
                "seed",
            );
        }

        #[test]
        fn struct_with_fields() {
            check_name_suggestion(
                from_type,
                r#"
                struct Seed { value: u32 }
                fn foo() {
                    let _ = $0Seed { value: 0 }$0;
                }"#,
                "seed",
            );
        }

        #[test]
        fn enum_() {
            check_name_suggestion(
                from_type,
                r#"
                enum Kind { A, B }
                fn foo() {
                    let _ = $0Kind::A$0;
                }"#,
                "kind",
            );
        }

        #[test]
        fn enum_generic_resolved() {
            check_name_suggestion(
                from_type,
                r#"
                enum Kind<T> { A(T), B }
                fn foo() {
                    let _ = $0Kind::A(1)$0;
                }"#,
                "kind",
            );
        }

        #[test]
        fn enum_generic_unresolved() {
            check_name_suggestion(
                from_type,
                r#"
                enum Kind<T> { A(T), B }
                fn foo<T>(x: T) {
                    let _ = $0Kind::A(x)$0;
                }"#,
                "kind",
            );
        }

        #[test]
        fn dyn_trait() {
            check_name_suggestion(
                from_type,
                r#"
                trait DynHandler {}
                fn bar() -> dyn DynHandler {}
                fn foo() {
                    $0bar()$0;
                }"#,
                "dyn_handler",
            );
        }

        #[test]
        fn impl_trait() {
            check_name_suggestion(
                from_type,
                r#"
                trait StaticHandler {}
                fn bar() -> impl StaticHandler {}
                fn foo() {
                    $0bar()$0;
                }"#,
                "static_handler",
            );
        }

        #[test]
        fn impl_trait_plus_clone() {
            check_name_suggestion(
                from_type,
                r#"
                trait StaticHandler {}
                trait Clone {}
                fn bar() -> impl StaticHandler + Clone {}
                fn foo() {
                    $0bar()$0;
                }"#,
                "static_handler",
            );
        }

        #[test]
        fn impl_trait_plus_lifetime() {
            check_name_suggestion(
                from_type,
                r#"
                trait StaticHandler {}
                trait Clone {}
                fn bar<'a>(&'a i32) -> impl StaticHandler + 'a {}
                fn foo() {
                    $0bar(&1)$0;
                }"#,
                "static_handler",
            );
        }

        #[test]
        fn impl_trait_plus_trait() {
            check_name_suggestion_not_applicable(
                from_type,
                r#"
                trait Handler {}
                trait StaticHandler {}
                fn bar() -> impl StaticHandler + Handler {}
                fn foo() {
                    $0bar()$0;
                }"#,
            );
        }

        #[test]
        fn ref_value() {
            check_name_suggestion(
                from_type,
                r#"
                struct Seed;
                fn bar() -> &Seed {}
                fn foo() {
                    $0bar()$0;
                }"#,
                "seed",
            );
        }

        #[test]
        fn box_value() {
            check_name_suggestion(
                from_type,
                r#"
                struct Box<T>(*const T);
                struct Seed;
                fn bar() -> Box<Seed> {}
                fn foo() {
                    $0bar()$0;
                }"#,
                "seed",
            );
        }

        #[test]
        fn box_generic() {
            check_name_suggestion_not_applicable(
                from_type,
                r#"
                struct Box<T>(*const T);
                fn bar<T>() -> Box<T> {}
                fn foo<T>() {
                    $0bar::<T>()$0;
                }"#,
            );
        }

        #[test]
        fn option_value() {
            check_name_suggestion(
                from_type,
                r#"
                enum Option<T> { Some(T) }
                struct Seed;
                fn bar() -> Option<Seed> {}
                fn foo() {
                    $0bar()$0;
                }"#,
                "seed",
            );
        }

        #[test]
        fn result_value() {
            check_name_suggestion(
                from_type,
                r#"
                enum Result<T, E> { Ok(T), Err(E) }
                struct Seed;
                struct Error;
                fn bar() -> Result<Seed, Error> {}
                fn foo() {
                    $0bar()$0;
                }"#,
                "seed",
            );
        }
    }

    mod variable {
        use super::*;

        #[test]
        fn ref_call() {
            check_name_suggestion(
                |e, c| Some(variable(e, c)),
                r#"
                fn foo() {
                    $0&bar(1, 3)$0
                }"#,
                "bar",
            );
        }

        #[test]
        fn name_to_string() {
            check_name_suggestion(
                |e, c| Some(variable(e, c)),
                r#"
                fn foo() {
                    $0function.name().to_string()$0
                }"#,
                "name",
            );
        }

        #[test]
        fn nested_useless_method() {
            check_name_suggestion(
                |e, c| Some(variable(e, c)),
                r#"
                fn foo() {
                    $0function.name().as_ref().unwrap().to_string()$0
                }"#,
                "name",
            );
        }
    }
}

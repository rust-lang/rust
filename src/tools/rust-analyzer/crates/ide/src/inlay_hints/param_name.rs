//! Implementation of "param name" inlay hints:
//! ```no_run
//! fn max(x: i32, y: i32) -> i32 { x + y }
//! _ = max(/*x*/4, /*y*/4);
//! ```
use either::Either;
use hir::{Callable, Semantics};
use ide_db::{base_db::FileRange, RootDatabase};

use stdx::to_lower_snake_case;
use syntax::ast::{self, AstNode, HasArgList, HasName, UnaryOp};

use crate::{InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    expr: ast::Expr,
) -> Option<()> {
    if !config.parameter_hints {
        return None;
    }

    let (callable, arg_list) = get_callable(sema, &expr)?;
    let hints = callable
        .params(sema.db)
        .into_iter()
        .zip(arg_list.args())
        .filter_map(|((param, _ty), arg)| {
            // Only annotate hints for expressions that exist in the original file
            let range = sema.original_range_opt(arg.syntax())?;
            let (param_name, name_syntax) = match param.as_ref()? {
                Either::Left(pat) => (pat.name()?, pat.name()),
                Either::Right(pat) => match pat {
                    ast::Pat::IdentPat(it) => (it.name()?, it.name()),
                    _ => return None,
                },
            };
            Some((name_syntax, param_name, arg, range))
        })
        .filter(|(_, param_name, arg, _)| {
            !should_hide_param_name_hint(sema, &callable, &param_name.text(), arg)
        })
        .map(|(param, param_name, _, FileRange { range, .. })| {
            let mut linked_location = None;
            if let Some(name) = param {
                if let hir::CallableKind::Function(f) = callable.kind() {
                    // assert the file is cached so we can map out of macros
                    if let Some(_) = sema.source(f) {
                        linked_location = sema.original_range_opt(name.syntax());
                    }
                }
            }

            let colon = if config.render_colons { ":" } else { "" };
            let label =
                InlayHintLabel::simple(format!("{param_name}{colon}"), None, linked_location);
            InlayHint {
                range,
                kind: InlayKind::Parameter,
                label,
                text_edit: None,
                position: InlayHintPosition::Before,
                pad_left: false,
                pad_right: true,
            }
        });

    acc.extend(hints);
    Some(())
}

fn get_callable(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
) -> Option<(hir::Callable, ast::ArgList)> {
    match expr {
        ast::Expr::CallExpr(expr) => {
            let descended = sema.descend_node_into_attributes(expr.clone()).pop();
            let expr = descended.as_ref().unwrap_or(expr);
            sema.type_of_expr(&expr.expr()?)?.original.as_callable(sema.db).zip(expr.arg_list())
        }
        ast::Expr::MethodCallExpr(expr) => {
            let descended = sema.descend_node_into_attributes(expr.clone()).pop();
            let expr = descended.as_ref().unwrap_or(expr);
            sema.resolve_method_call_as_callable(expr).zip(expr.arg_list())
        }
        _ => None,
    }
}

fn should_hide_param_name_hint(
    sema: &Semantics<'_, RootDatabase>,
    callable: &hir::Callable,
    param_name: &str,
    argument: &ast::Expr,
) -> bool {
    // These are to be tested in the `parameter_hint_heuristics` test
    // hide when:
    // - the parameter name is a suffix of the function's name
    // - the argument is a qualified constructing or call expression where the qualifier is an ADT
    // - exact argument<->parameter match(ignoring leading underscore) or parameter is a prefix/suffix
    //   of argument with _ splitting it off
    // - param starts with `ra_fixture`
    // - param is a well known name in a unary function

    let param_name = param_name.trim_start_matches('_');
    if param_name.is_empty() {
        return true;
    }

    if matches!(argument, ast::Expr::PrefixExpr(prefix) if prefix.op_kind() == Some(UnaryOp::Not)) {
        return false;
    }

    let fn_name = match callable.kind() {
        hir::CallableKind::Function(it) => Some(it.name(sema.db).to_smol_str()),
        _ => None,
    };
    let fn_name = fn_name.as_deref();
    is_param_name_suffix_of_fn_name(param_name, callable, fn_name)
        || is_argument_similar_to_param_name(argument, param_name)
        || param_name.starts_with("ra_fixture")
        || (callable.n_params() == 1 && is_obvious_param(param_name))
        || is_adt_constructor_similar_to_param_name(sema, argument, param_name)
}

/// Hide the parameter name of a unary function if it is a `_` - prefixed suffix of the function's name, or equal.
///
/// `fn strip_suffix(suffix)` will be hidden.
/// `fn stripsuffix(suffix)` will not be hidden.
fn is_param_name_suffix_of_fn_name(
    param_name: &str,
    callable: &Callable,
    fn_name: Option<&str>,
) -> bool {
    match (callable.n_params(), fn_name) {
        (1, Some(function)) => {
            function == param_name
                || function
                    .len()
                    .checked_sub(param_name.len())
                    .and_then(|at| function.is_char_boundary(at).then(|| function.split_at(at)))
                    .map_or(false, |(prefix, suffix)| {
                        suffix.eq_ignore_ascii_case(param_name) && prefix.ends_with('_')
                    })
        }
        _ => false,
    }
}

fn is_argument_similar_to_param_name(argument: &ast::Expr, param_name: &str) -> bool {
    // check whether param_name and argument are the same or
    // whether param_name is a prefix/suffix of argument(split at `_`)
    let argument = match get_string_representation(argument) {
        Some(argument) => argument,
        None => return false,
    };

    // std is honestly too panic happy...
    let str_split_at = |str: &str, at| str.is_char_boundary(at).then(|| argument.split_at(at));

    let param_name = param_name.trim_start_matches('_');
    let argument = argument.trim_start_matches('_');

    match str_split_at(argument, param_name.len()) {
        Some((prefix, rest)) if prefix.eq_ignore_ascii_case(param_name) => {
            return rest.is_empty() || rest.starts_with('_');
        }
        _ => (),
    }
    match argument.len().checked_sub(param_name.len()).and_then(|at| str_split_at(argument, at)) {
        Some((rest, suffix)) if param_name.eq_ignore_ascii_case(suffix) => {
            return rest.is_empty() || rest.ends_with('_');
        }
        _ => (),
    }
    false
}

fn get_string_representation(expr: &ast::Expr) -> Option<String> {
    match expr {
        ast::Expr::MethodCallExpr(method_call_expr) => {
            let name_ref = method_call_expr.name_ref()?;
            match name_ref.text().as_str() {
                "clone" | "as_ref" => method_call_expr.receiver().map(|rec| rec.to_string()),
                name_ref => Some(name_ref.to_owned()),
            }
        }
        ast::Expr::MacroExpr(macro_expr) => {
            Some(macro_expr.macro_call()?.path()?.segment()?.to_string())
        }
        ast::Expr::FieldExpr(field_expr) => Some(field_expr.name_ref()?.to_string()),
        ast::Expr::PathExpr(path_expr) => Some(path_expr.path()?.segment()?.to_string()),
        ast::Expr::PrefixExpr(prefix_expr) => get_string_representation(&prefix_expr.expr()?),
        ast::Expr::RefExpr(ref_expr) => get_string_representation(&ref_expr.expr()?),
        ast::Expr::CastExpr(cast_expr) => get_string_representation(&cast_expr.expr()?),
        _ => None,
    }
}

fn is_obvious_param(param_name: &str) -> bool {
    // avoid displaying hints for common functions like map, filter, etc.
    // or other obvious words used in std
    let is_obvious_param_name =
        matches!(param_name, "predicate" | "value" | "pat" | "rhs" | "other");
    param_name.len() == 1 || is_obvious_param_name
}

fn is_adt_constructor_similar_to_param_name(
    sema: &Semantics<'_, RootDatabase>,
    argument: &ast::Expr,
    param_name: &str,
) -> bool {
    let path = match argument {
        ast::Expr::CallExpr(c) => c.expr().and_then(|e| match e {
            ast::Expr::PathExpr(p) => p.path(),
            _ => None,
        }),
        ast::Expr::PathExpr(p) => p.path(),
        ast::Expr::RecordExpr(r) => r.path(),
        _ => return false,
    };
    let path = match path {
        Some(it) => it,
        None => return false,
    };
    (|| match sema.resolve_path(&path)? {
        hir::PathResolution::Def(hir::ModuleDef::Adt(_)) => {
            Some(to_lower_snake_case(&path.segment()?.name_ref()?.text()) == param_name)
        }
        hir::PathResolution::Def(hir::ModuleDef::Function(_) | hir::ModuleDef::Variant(_)) => {
            if to_lower_snake_case(&path.segment()?.name_ref()?.text()) == param_name {
                return Some(true);
            }
            let qual = path.qualifier()?;
            match sema.resolve_path(&qual)? {
                hir::PathResolution::Def(hir::ModuleDef::Adt(_)) => {
                    Some(to_lower_snake_case(&qual.segment()?.name_ref()?.text()) == param_name)
                }
                _ => None,
            }
        }
        _ => None,
    })()
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use crate::{
        inlay_hints::tests::{check_with_config, DISABLED_CONFIG},
        InlayHintsConfig,
    };

    #[track_caller]
    fn check_params(ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig { parameter_hints: true, ..DISABLED_CONFIG },
            ra_fixture,
        );
    }

    #[test]
    fn param_hints_only() {
        check_params(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(
        4,
      //^ a
        4,
      //^ b
    );
}"#,
        );
    }

    #[test]
    fn param_hints_on_closure() {
        check_params(
            r#"
fn main() {
    let clo = |a: u8, b: u8| a + b;
    clo(
        1,
      //^ a
        2,
      //^ b
    );
}
            "#,
        );
    }

    #[test]
    fn param_name_similar_to_fn_name_still_hints() {
        check_params(
            r#"
fn max(x: i32, y: i32) -> i32 { x + y }
fn main() {
    let _x = max(
        4,
      //^ x
        4,
      //^ y
    );
}"#,
        );
    }

    #[test]
    fn param_name_similar_to_fn_name() {
        check_params(
            r#"
fn param_with_underscore(with_underscore: i32) -> i32 { with_underscore }
fn main() {
    let _x = param_with_underscore(
        4,
    );
}"#,
        );
        check_params(
            r#"
fn param_with_underscore(underscore: i32) -> i32 { underscore }
fn main() {
    let _x = param_with_underscore(
        4,
    );
}"#,
        );
    }

    #[test]
    fn param_name_same_as_fn_name() {
        check_params(
            r#"
fn foo(foo: i32) -> i32 { foo }
fn main() {
    let _x = foo(
        4,
    );
}"#,
        );
    }

    #[test]
    fn never_hide_param_when_multiple_params() {
        check_params(
            r#"
fn foo(foo: i32, bar: i32) -> i32 { bar + baz }
fn main() {
    let _x = foo(
        4,
      //^ foo
        8,
      //^ bar
    );
}"#,
        );
    }

    #[test]
    fn param_hints_look_through_as_ref_and_clone() {
        check_params(
            r#"
fn foo(bar: i32, baz: f32) {}

fn main() {
    let bar = 3;
    let baz = &"baz";
    let fez = 1.0;
    foo(bar.clone(), bar.clone());
                   //^^^^^^^^^^^ baz
    foo(bar.as_ref(), bar.as_ref());
                    //^^^^^^^^^^^^ baz
}
"#,
        );
    }

    #[test]
    fn self_param_hints() {
        check_params(
            r#"
struct Foo;

impl Foo {
    fn foo(self: Self) {}
    fn bar(self: &Self) {}
}

fn main() {
    Foo::foo(Foo);
           //^^^ self
    Foo::bar(&Foo);
           //^^^^ self
}
"#,
        )
    }

    #[test]
    fn param_name_hints_show_for_literals() {
        check_params(
            r#"pub fn test(a: i32, b: i32) -> [i32; 2] { [a, b] }
fn main() {
    test(
        0xa_b,
      //^^^^^ a
        0xa_b,
      //^^^^^ b
    );
}"#,
        )
    }

    #[test]
    fn function_call_parameter_hint() {
        check_params(
            r#"
//- minicore: option
struct FileId {}
struct SmolStr {}

struct TextRange {}
struct SyntaxKind {}
struct NavigationTarget {}

struct Test {}

impl Test {
    fn method(&self, mut param: i32) -> i32 { param * 2 }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        full_range: TextRange,
        kind: SyntaxKind,
        docs: Option<String>,
    ) -> NavigationTarget {
        NavigationTarget {}
    }
}

fn test_func(mut foo: i32, bar: i32, msg: &str, _: i32, last: i32) -> i32 {
    foo + bar
}

fn main() {
    let not_literal = 1;
    let _: i32 = test_func(1,    2,      "hello", 3,  not_literal);
                         //^ foo ^ bar   ^^^^^^^ msg  ^^^^^^^^^^^ last
    let t: Test = Test {};
    t.method(123);
           //^^^ param
    Test::method(&t,      3456);
               //^^ self  ^^^^ param
    Test::from_syntax(
        FileId {},
        "impl".into(),
      //^^^^^^^^^^^^^ name
        None,
      //^^^^ focus_range
        TextRange {},
      //^^^^^^^^^^^^ full_range
        SyntaxKind {},
      //^^^^^^^^^^^^^ kind
        None,
      //^^^^ docs
    );
}"#,
        );
    }

    #[test]
    fn parameter_hint_heuristics() {
        check_params(
            r#"
fn check(ra_fixture_thing: &str) {}

fn map(f: i32) {}
fn filter(predicate: i32) {}

fn strip_suffix(suffix: &str) {}
fn stripsuffix(suffix: &str) {}
fn same(same: u32) {}
fn same2(_same2: u32) {}

fn enum_matches_param_name(completion_kind: CompletionKind) {}

fn foo(param: u32) {}
fn bar(param_eter: u32) {}

enum CompletionKind {
    Keyword,
}

fn non_ident_pat((a, b): (u32, u32)) {}

fn main() {
    const PARAM: u32 = 0;
    foo(PARAM);
    foo(!PARAM);
     // ^^^^^^ param
    check("");

    map(0);
    filter(0);

    strip_suffix("");
    stripsuffix("");
              //^^ suffix
    same(0);
    same2(0);

    enum_matches_param_name(CompletionKind::Keyword);

    let param = 0;
    foo(param);
    foo(param as _);
    let param_end = 0;
    foo(param_end);
    let start_param = 0;
    foo(start_param);
    let param2 = 0;
    foo(param2);
      //^^^^^^ param

    macro_rules! param {
        () => {};
    };
    foo(param!());

    let param_eter = 0;
    bar(param_eter);
    let param_eter_end = 0;
    bar(param_eter_end);
    let start_param_eter = 0;
    bar(start_param_eter);
    let param_eter2 = 0;
    bar(param_eter2);
      //^^^^^^^^^^^ param_eter

    non_ident_pat((0, 0));
}"#,
        );
    }
}

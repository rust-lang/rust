//! Implementation of "param name" inlay hints:
//! ```no_run
//! fn max(x: i32, y: i32) -> i32 { x + y }
//! _ = max(/*x*/4, /*y*/4);
//! ```

use std::iter::zip;

use either::Either;
use hir::Semantics;
use ide_db::{RootDatabase, famous_defs::FamousDefs};

use stdx::to_lower_snake_case;
use syntax::ast::{self, AstNode, HasArgList, HasName, UnaryOp};

use crate::{InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, krate): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    expr: ast::Expr,
) -> Option<()> {
    if !config.parameter_hints {
        return None;
    }

    let (callable, arg_list) = get_callable(sema, &expr)?;
    let unary_function = callable.n_params() == 1;
    let function_name = match callable.kind() {
        hir::CallableKind::Function(function) => Some(function.name(sema.db)),
        _ => None,
    };
    let function_name = function_name.as_ref().map(|it| it.as_str());
    let hints = callable
        .params()
        .into_iter()
        .zip(arg_list.args())
        .filter_map(|(p, arg)| {
            // Only annotate hints for expressions that exist in the original file
            let range = sema.original_range_opt(arg.syntax())?;
            let param_name = p.name(sema.db)?;
            Some((p, param_name, arg, range))
        })
        .filter(|(_, param_name, arg, _)| {
            !should_hide_param_name_hint(
                sema,
                unary_function,
                function_name,
                param_name.as_str(),
                arg,
            )
        })
        .map(|(param, param_name, _, hir::FileRange { range, .. })| {
            let colon = if config.render_colons { ":" } else { "" };
            let label = InlayHintLabel::simple(
                format!("{}{colon}", param_name.display(sema.db, krate.edition(sema.db))),
                None,
                config.lazy_location_opt(|| {
                    let source = sema.source(param)?;
                    let name_syntax = match source.value.as_ref() {
                        Either::Left(pat) => pat.name(),
                        Either::Right(param) => match param.pat()? {
                            ast::Pat::IdentPat(it) => it.name(),
                            _ => None,
                        },
                    }?;
                    sema.original_range_opt(name_syntax.syntax()).map(|frange| ide_db::FileRange {
                        file_id: frange.file_id.file_id(sema.db),
                        range: frange.range,
                    })
                }),
            );
            InlayHint {
                range,
                kind: InlayKind::Parameter,
                label,
                text_edit: None,
                position: InlayHintPosition::Before,
                pad_left: false,
                pad_right: true,
                resolve_parent: Some(expr.syntax().text_range()),
            }
        });

    acc.extend(hints);
    Some(())
}

fn get_callable<'db>(
    sema: &Semantics<'db, RootDatabase>,
    expr: &ast::Expr,
) -> Option<(hir::Callable<'db>, ast::ArgList)> {
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

const INSIGNIFICANT_METHOD_NAMES: &[&str] = &["clone", "as_ref", "into"];
const INSIGNIFICANT_PARAMETER_NAMES: &[&str] = &["predicate", "value", "pat", "rhs", "other"];

fn should_hide_param_name_hint(
    sema: &Semantics<'_, RootDatabase>,
    unary_function: bool,
    function_name: Option<&str>,
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

    if param_name.starts_with("ra_fixture") {
        return true;
    }

    if unary_function {
        if let Some(function_name) = function_name {
            if is_param_name_suffix_of_fn_name(param_name, function_name) {
                return true;
            }
        }
        if is_obvious_param(param_name) {
            return true;
        }
    }

    is_argument_expr_similar_to_param_name(sema, argument, param_name)
}

/// Hide the parameter name of a unary function if it is a `_` - prefixed suffix of the function's name, or equal.
///
/// `fn strip_suffix(suffix)` will be hidden.
/// `fn stripsuffix(suffix)` will not be hidden.
fn is_param_name_suffix_of_fn_name(param_name: &str, fn_name: &str) -> bool {
    fn_name == param_name
        || fn_name
            .len()
            .checked_sub(param_name.len())
            .and_then(|at| fn_name.is_char_boundary(at).then(|| fn_name.split_at(at)))
            .is_some_and(|(prefix, suffix)| {
                suffix.eq_ignore_ascii_case(param_name) && prefix.ends_with('_')
            })
}

fn is_argument_expr_similar_to_param_name(
    sema: &Semantics<'_, RootDatabase>,
    argument: &ast::Expr,
    param_name: &str,
) -> bool {
    match get_segment_representation(argument) {
        Some(Either::Left(argument)) => is_argument_similar_to_param_name(&argument, param_name),
        Some(Either::Right(path)) => {
            path.segment()
                .and_then(|it| it.name_ref())
                .is_some_and(|name_ref| name_ref.text().eq_ignore_ascii_case(param_name))
                || is_adt_constructor_similar_to_param_name(sema, &path, param_name)
        }
        None => false,
    }
}

/// Check whether param_name and argument are the same or
/// whether param_name is a prefix/suffix of argument(split at `_`).
pub(super) fn is_argument_similar_to_param_name(
    argument: &[ast::NameRef],
    param_name: &str,
) -> bool {
    debug_assert!(!argument.is_empty());
    debug_assert!(!param_name.is_empty());
    let param_name = param_name.split('_');
    let argument = argument.iter().flat_map(|it| it.text_non_mutable().split('_'));

    let prefix_match = zip(argument.clone(), param_name.clone())
        .all(|(arg, param)| arg.eq_ignore_ascii_case(param));
    let postfix_match = || {
        zip(argument.rev(), param_name.rev()).all(|(arg, param)| arg.eq_ignore_ascii_case(param))
    };
    prefix_match || postfix_match()
}

pub(super) fn get_segment_representation(
    expr: &ast::Expr,
) -> Option<Either<Vec<ast::NameRef>, ast::Path>> {
    match expr {
        ast::Expr::MethodCallExpr(method_call_expr) => {
            let receiver =
                method_call_expr.receiver().and_then(|expr| get_segment_representation(&expr));
            let name_ref = method_call_expr.name_ref()?;
            if INSIGNIFICANT_METHOD_NAMES.contains(&name_ref.text().as_str()) {
                return receiver;
            }
            Some(Either::Left(match receiver {
                Some(Either::Left(mut left)) => {
                    left.push(name_ref);
                    left
                }
                Some(Either::Right(_)) | None => vec![name_ref],
            }))
        }
        ast::Expr::FieldExpr(field_expr) => {
            let expr = field_expr.expr().and_then(|expr| get_segment_representation(&expr));
            let name_ref = field_expr.name_ref()?;
            let res = match expr {
                Some(Either::Left(mut left)) => {
                    left.push(name_ref);
                    left
                }
                Some(Either::Right(_)) | None => vec![name_ref],
            };
            Some(Either::Left(res))
        }
        // paths
        ast::Expr::MacroExpr(macro_expr) => macro_expr.macro_call()?.path().map(Either::Right),
        ast::Expr::RecordExpr(record_expr) => record_expr.path().map(Either::Right),
        ast::Expr::PathExpr(path_expr) => {
            let path = path_expr.path()?;
            // single segment paths are likely locals
            Some(match path.as_single_name_ref() {
                None => Either::Right(path),
                Some(name_ref) => Either::Left(vec![name_ref]),
            })
        }
        ast::Expr::PrefixExpr(prefix_expr) if prefix_expr.op_kind() == Some(UnaryOp::Not) => None,
        // recurse
        ast::Expr::PrefixExpr(prefix_expr) => get_segment_representation(&prefix_expr.expr()?),
        ast::Expr::RefExpr(ref_expr) => get_segment_representation(&ref_expr.expr()?),
        ast::Expr::CastExpr(cast_expr) => get_segment_representation(&cast_expr.expr()?),
        ast::Expr::CallExpr(call_expr) => get_segment_representation(&call_expr.expr()?),
        ast::Expr::AwaitExpr(await_expr) => get_segment_representation(&await_expr.expr()?),
        ast::Expr::IndexExpr(index_expr) => get_segment_representation(&index_expr.base()?),
        ast::Expr::ParenExpr(paren_expr) => get_segment_representation(&paren_expr.expr()?),
        ast::Expr::TryExpr(try_expr) => get_segment_representation(&try_expr.expr()?),
        // ast::Expr::ClosureExpr(closure_expr) => todo!(),
        _ => None,
    }
}

fn is_obvious_param(param_name: &str) -> bool {
    // avoid displaying hints for common functions like map, filter, etc.
    // or other obvious words used in std
    param_name.len() == 1 || INSIGNIFICANT_PARAMETER_NAMES.contains(&param_name)
}

fn is_adt_constructor_similar_to_param_name(
    sema: &Semantics<'_, RootDatabase>,
    path: &ast::Path,
    param_name: &str,
) -> bool {
    (|| match sema.resolve_path(path)? {
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
        InlayHintsConfig,
        inlay_hints::tests::{DISABLED_CONFIG, check_with_config},
    };

    #[track_caller]
    fn check_params(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
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
fn baz(a_d_e: u32) {}

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

    baz(a.d.e);
    baz(a.dc.e);
     // ^^^^^^ a_d_e
    baz(ac.d.e);
     // ^^^^^^ a_d_e
    baz(a.d.ec);
     // ^^^^^^ a_d_e
}"#,
        );
    }
}

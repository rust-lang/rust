use either::Either;
use hir::{Callable, Semantics};
use ide_db::{base_db::FileRange, RootDatabase};

use stdx::to_lower_snake_case;
use syntax::ast::{self, AstNode, HasArgList, HasName, UnaryOp};

use crate::{InlayHint, InlayHintsConfig, InlayKind, InlayTooltip};

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
                Either::Left(pat) => ("self".to_string(), pat.name()),
                Either::Right(pat) => match pat {
                    ast::Pat::IdentPat(it) => (it.name()?.to_string(), it.name()),
                    _ => return None,
                },
            };
            Some((name_syntax, param_name, arg, range))
        })
        .filter(|(_, param_name, arg, _)| {
            !should_hide_param_name_hint(sema, &callable, param_name, arg)
        })
        .map(|(param, param_name, _, FileRange { range, .. })| {
            let mut tooltip = None;
            if let Some(name) = param {
                if let hir::CallableKind::Function(f) = callable.kind() {
                    // assert the file is cached so we can map out of macros
                    if let Some(_) = sema.source(f) {
                        tooltip = sema.original_range_opt(name.syntax());
                    }
                }
            }

            InlayHint {
                range,
                kind: InlayKind::ParameterHint,
                label: param_name.into(),
                tooltip: tooltip.map(|it| InlayTooltip::HoverOffset(it.file_id, it.range.start())),
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

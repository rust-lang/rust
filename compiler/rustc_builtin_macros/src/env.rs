// The compiler code necessary to support the env! extension. Eventually this
// should all get sucked into either the compiler syntax extension plugin
// interface.
//

use rustc_ast::token::{self, LitKind};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AstDeref, ExprKind, GenericArg, Mutability};
use rustc_expand::base::{
    expr_to_string, get_exprs_from_tts, get_single_str_from_tts, DummyResult, ExtCtxt, MacEager,
    MacResult,
};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use std::env;
use thin_vec::thin_vec;

use crate::errors;

fn lookup_env<'cx>(cx: &'cx ExtCtxt<'_>, var: Symbol) -> Option<Symbol> {
    let var = var.as_str();
    if let Some(value) = cx.sess.opts.logical_env.get(var) {
        return Some(Symbol::intern(value));
    }
    // If the environment variable was not defined with the `--env-set` option, we try to retrieve it
    // from rustc's environment.
    env::var(var).ok().as_deref().map(Symbol::intern)
}

pub fn expand_option_env<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'cx> {
    let var = match get_single_str_from_tts(cx, sp, tts, "option_env!") {
        Ok(var) => var,
        Err(guar) => return DummyResult::any(sp, guar),
    };

    let sp = cx.with_def_site_ctxt(sp);
    let value = lookup_env(cx, var);
    cx.sess.psess.env_depinfo.borrow_mut().insert((var, value));
    let e = match value {
        None => {
            let lt = cx.lifetime(sp, Ident::new(kw::StaticLifetime, sp));
            cx.expr_path(cx.path_all(
                sp,
                true,
                cx.std_path(&[sym::option, sym::Option, sym::None]),
                vec![GenericArg::Type(cx.ty_ref(
                    sp,
                    cx.ty_ident(sp, Ident::new(sym::str, sp)),
                    Some(lt),
                    Mutability::Not,
                ))],
            ))
        }
        Some(value) => cx.expr_call_global(
            sp,
            cx.std_path(&[sym::option, sym::Option, sym::Some]),
            thin_vec![cx.expr_str(sp, value)],
        ),
    };
    MacEager::expr(e)
}

pub fn expand_env<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'cx> {
    let mut exprs = match get_exprs_from_tts(cx, tts) {
        Ok(exprs) if exprs.is_empty() || exprs.len() > 2 => {
            let guar = cx.dcx().emit_err(errors::EnvTakesArgs { span: sp });
            return DummyResult::any(sp, guar);
        }
        Err(guar) => return DummyResult::any(sp, guar),
        Ok(exprs) => exprs.into_iter(),
    };

    let var_expr = exprs.next().unwrap();
    let var = match expr_to_string(cx, var_expr.clone(), "expected string literal") {
        Ok((var, _)) => var,
        Err(guar) => return DummyResult::any(sp, guar),
    };

    let custom_msg = match exprs.next() {
        None => None,
        Some(second) => match expr_to_string(cx, second, "expected string literal") {
            Ok((s, _)) => Some(s),
            Err(guar) => return DummyResult::any(sp, guar),
        },
    };

    let span = cx.with_def_site_ctxt(sp);
    let value = lookup_env(cx, var);
    cx.sess.psess.env_depinfo.borrow_mut().insert((var, value));
    let e = match value {
        None => {
            let ExprKind::Lit(token::Lit {
                kind: LitKind::Str | LitKind::StrRaw(..), symbol, ..
            }) = &var_expr.kind
            else {
                unreachable!("`expr_to_string` ensures this is a string lit")
            };

            let guar = if let Some(msg_from_user) = custom_msg {
                cx.dcx().emit_err(errors::EnvNotDefinedWithUserMessage { span, msg_from_user })
            } else if is_cargo_env_var(var.as_str()) {
                cx.dcx().emit_err(errors::EnvNotDefined::CargoEnvVar {
                    span,
                    var: *symbol,
                    var_expr: var_expr.ast_deref(),
                })
            } else {
                cx.dcx().emit_err(errors::EnvNotDefined::CustomEnvVar {
                    span,
                    var: *symbol,
                    var_expr: var_expr.ast_deref(),
                })
            };

            return DummyResult::any(sp, guar);
        }
        Some(value) => cx.expr_str(span, value),
    };
    MacEager::expr(e)
}

/// Returns `true` if an environment variable from `env!` is one used by Cargo.
fn is_cargo_env_var(var: &str) -> bool {
    var.starts_with("CARGO_")
        || var.starts_with("DEP_")
        || matches!(var, "OUT_DIR" | "OPT_LEVEL" | "PROFILE" | "HOST" | "TARGET")
}

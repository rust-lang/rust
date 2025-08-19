// The compiler code necessary to support the env! extension. Eventually this
// should all get sucked into either the compiler syntax extension plugin
// interface.
//

use std::env;
use std::env::VarError;

use rustc_ast::token::{self, LitKind};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{ExprKind, GenericArg, Mutability};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult};
use rustc_span::{Ident, Span, Symbol, kw, sym};
use thin_vec::thin_vec;

use crate::errors;
use crate::util::{expr_to_string, get_exprs_from_tts, get_single_expr_from_tts};

fn lookup_env<'cx>(cx: &'cx ExtCtxt<'_>, var: Symbol) -> Result<Symbol, VarError> {
    let var = var.as_str();
    if let Some(value) = cx.sess.opts.logical_env.get(var) {
        return Ok(Symbol::intern(value));
    }
    // If the environment variable was not defined with the `--env-set` option, we try to retrieve it
    // from rustc's environment.
    Ok(Symbol::intern(&env::var(var)?))
}

pub(crate) fn expand_option_env<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let ExpandResult::Ready(mac_expr) = get_single_expr_from_tts(cx, sp, tts, "option_env!") else {
        return ExpandResult::Retry(());
    };
    let var_expr = match mac_expr {
        Ok(var_expr) => var_expr,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };
    let ExpandResult::Ready(mac) =
        expr_to_string(cx, var_expr.clone(), "argument must be a string literal")
    else {
        return ExpandResult::Retry(());
    };
    let var = match mac {
        Ok((var, _)) => var,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };

    let sp = cx.with_def_site_ctxt(sp);
    let value = lookup_env(cx, var);
    cx.sess.psess.env_depinfo.borrow_mut().insert((var, value.as_ref().ok().copied()));
    let e = match value {
        Err(VarError::NotPresent) => {
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
        Err(VarError::NotUnicode(_)) => {
            let ExprKind::Lit(token::Lit {
                kind: LitKind::Str | LitKind::StrRaw(..), symbol, ..
            }) = &var_expr.kind
            else {
                unreachable!("`expr_to_string` ensures this is a string lit")
            };

            let guar = cx.dcx().emit_err(errors::EnvNotUnicode { span: sp, var: *symbol });
            return ExpandResult::Ready(DummyResult::any(sp, guar));
        }
        Ok(value) => cx.expr_call_global(
            sp,
            cx.std_path(&[sym::option, sym::Option, sym::Some]),
            thin_vec![cx.expr_str(sp, value)],
        ),
    };
    ExpandResult::Ready(MacEager::expr(e))
}

pub(crate) fn expand_env<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let ExpandResult::Ready(mac) = get_exprs_from_tts(cx, tts) else {
        return ExpandResult::Retry(());
    };
    let mut exprs = match mac {
        Ok(exprs) if exprs.is_empty() || exprs.len() > 2 => {
            let guar = cx.dcx().emit_err(errors::EnvTakesArgs { span: sp });
            return ExpandResult::Ready(DummyResult::any(sp, guar));
        }
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
        Ok(exprs) => exprs.into_iter(),
    };

    let var_expr = exprs.next().unwrap();
    let ExpandResult::Ready(mac) = expr_to_string(cx, var_expr.clone(), "expected string literal")
    else {
        return ExpandResult::Retry(());
    };
    let var = match mac {
        Ok((var, _)) => var,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };

    let custom_msg = match exprs.next() {
        None => None,
        Some(second) => {
            let ExpandResult::Ready(mac) = expr_to_string(cx, second, "expected string literal")
            else {
                return ExpandResult::Retry(());
            };
            match mac {
                Ok((s, _)) => Some(s),
                Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
            }
        }
    };

    let span = cx.with_def_site_ctxt(sp);
    let value = lookup_env(cx, var);
    cx.sess.psess.env_depinfo.borrow_mut().insert((var, value.as_ref().ok().copied()));
    let e = match value {
        Err(err) => {
            let ExprKind::Lit(token::Lit {
                kind: LitKind::Str | LitKind::StrRaw(..), symbol, ..
            }) = &var_expr.kind
            else {
                unreachable!("`expr_to_string` ensures this is a string lit")
            };

            let guar = match err {
                VarError::NotPresent => {
                    if let Some(msg_from_user) = custom_msg {
                        cx.dcx()
                            .emit_err(errors::EnvNotDefinedWithUserMessage { span, msg_from_user })
                    } else if is_cargo_env_var(var.as_str()) {
                        cx.dcx().emit_err(errors::EnvNotDefined::CargoEnvVar {
                            span,
                            var: *symbol,
                            var_expr: &var_expr,
                        })
                    } else {
                        cx.dcx().emit_err(errors::EnvNotDefined::CustomEnvVar {
                            span,
                            var: *symbol,
                            var_expr: &var_expr,
                        })
                    }
                }
                VarError::NotUnicode(_) => {
                    cx.dcx().emit_err(errors::EnvNotUnicode { span, var: *symbol })
                }
            };

            return ExpandResult::Ready(DummyResult::any(sp, guar));
        }
        Ok(value) => cx.expr_str(span, value),
    };
    ExpandResult::Ready(MacEager::expr(e))
}

/// Returns `true` if an environment variable from `env!` is one used by Cargo.
fn is_cargo_env_var(var: &str) -> bool {
    var.starts_with("CARGO_")
        || var.starts_with("DEP_")
        || matches!(var, "OUT_DIR" | "OPT_LEVEL" | "PROFILE" | "HOST" | "TARGET")
}

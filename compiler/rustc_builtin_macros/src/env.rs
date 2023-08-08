// The compiler code necessary to support the env! extension. Eventually this
// should all get sucked into either the compiler syntax extension plugin
// interface.
//

use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{self as ast, AstDeref, GenericArg};
use rustc_expand::base::{self, *};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use std::env;
use thin_vec::thin_vec;

use crate::errors;

pub fn expand_option_env<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    let Some(var) = get_single_str_from_tts(cx, sp, tts, "option_env!") else {
        return DummyResult::any(sp);
    };

    let sp = cx.with_def_site_ctxt(sp);
    let value = env::var(var.as_str()).ok().as_deref().map(Symbol::intern);
    cx.sess.parse_sess.env_depinfo.borrow_mut().insert((var, value));
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
                    ast::Mutability::Not,
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
) -> Box<dyn base::MacResult + 'cx> {
    let mut exprs = match get_exprs_from_tts(cx, tts) {
        Some(exprs) if exprs.is_empty() || exprs.len() > 2 => {
            cx.emit_err(errors::EnvTakesArgs { span: sp });
            return DummyResult::any(sp);
        }
        None => return DummyResult::any(sp),
        Some(exprs) => exprs.into_iter(),
    };

    let var_expr = exprs.next().unwrap();
    let Some((var, _)) = expr_to_string(cx, var_expr.clone(), "expected string literal") else {
        return DummyResult::any(sp);
    };

    let custom_msg = match exprs.next() {
        None => None,
        Some(second) => match expr_to_string(cx, second, "expected string literal") {
            None => return DummyResult::any(sp),
            Some((s, _)) => Some(s),
        },
    };

    let span = cx.with_def_site_ctxt(sp);
    let value = env::var(var.as_str()).ok().as_deref().map(Symbol::intern);
    cx.sess.parse_sess.env_depinfo.borrow_mut().insert((var, value));
    let e = match value {
        None => {
            let ast::ExprKind::Lit(ast::token::Lit {
                kind: ast::token::LitKind::Str | ast::token::LitKind::StrRaw(..),
                symbol,
                ..
            }) = &var_expr.kind
            else {
                unreachable!("`expr_to_string` ensures this is a string lit")
            };

            if let Some(msg_from_user) = custom_msg {
                cx.emit_err(errors::EnvNotDefinedWithUserMessage { span, msg_from_user });
            } else if is_cargo_env_var(var.as_str()) {
                cx.emit_err(errors::EnvNotDefined::CargoEnvVar {
                    span,
                    var: *symbol,
                    var_expr: var_expr.ast_deref(),
                });
            } else {
                cx.emit_err(errors::EnvNotDefined::CustomEnvVar {
                    span,
                    var: *symbol,
                    var_expr: var_expr.ast_deref(),
                });
            }

            return DummyResult::any(sp);
        }
        Some(value) => cx.expr_str(sp, value),
    };
    MacEager::expr(e)
}

/// Returns `true` if an environment variable from `env!` is one used by Cargo.
fn is_cargo_env_var(var: &str) -> bool {
    var.starts_with("CARGO_")
        || var.starts_with("DEP_")
        || matches!(var, "OUT_DIR" | "OPT_LEVEL" | "PROFILE" | "HOST" | "TARGET")
}

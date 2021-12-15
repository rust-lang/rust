// The compiler code necessary to support the env! extension.  Eventually this
// should all get sucked into either the compiler syntax extension plugin
// interface.
//

use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{self as ast, GenericArg};
use rustc_expand::base::{self, *};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;

use std::env;

pub fn expand_option_env<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    let var = match get_single_str_from_tts(cx, sp, tts, "option_env!") {
        None => return DummyResult::any(sp),
        Some(v) => v,
    };

    let sp = cx.with_def_site_ctxt(sp);
    let value = env::var(&var.as_str()).ok().as_deref().map(Symbol::intern);
    cx.sess.parse_sess.env_depinfo.borrow_mut().insert((Symbol::intern(&var), value));
    let e = match value {
        None => {
            let lt = cx.lifetime(sp, Ident::new(kw::StaticLifetime, sp));
            cx.expr_path(cx.path_all(
                sp,
                true,
                cx.std_path(&[sym::option, sym::Option, sym::None]),
                vec![GenericArg::Type(cx.ty_rptr(
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
            vec![cx.expr_str(sp, value)],
        ),
    };
    MacEager::expr(e)
}

pub fn expand_env<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    let mut exprs = match get_exprs_from_tts(cx, sp, tts) {
        Some(ref exprs) if exprs.is_empty() => {
            cx.span_err(sp, "env! takes 1 or 2 arguments");
            return DummyResult::any(sp);
        }
        None => return DummyResult::any(sp),
        Some(exprs) => exprs.into_iter(),
    };

    let var = match expr_to_string(cx, exprs.next().unwrap(), "expected string literal") {
        None => return DummyResult::any(sp),
        Some((v, _style)) => v,
    };
    let msg = match exprs.next() {
        None => Symbol::intern(&format!("environment variable `{}` not defined", var)),
        Some(second) => match expr_to_string(cx, second, "expected string literal") {
            None => return DummyResult::any(sp),
            Some((s, _style)) => s,
        },
    };

    if exprs.next().is_some() {
        cx.span_err(sp, "env! takes 1 or 2 arguments");
        return DummyResult::any(sp);
    }

    let sp = cx.with_def_site_ctxt(sp);
    let value = env::var(var.as_str()).ok().as_deref().map(Symbol::intern);
    cx.sess.parse_sess.env_depinfo.borrow_mut().insert((var, value));
    let e = match value {
        None => {
            cx.span_err(sp, msg.as_str());
            return DummyResult::any(sp);
        }
        Some(value) => cx.expr_str(sp, value),
    };
    MacEager::expr(e)
}

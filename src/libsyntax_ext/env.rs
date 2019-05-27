// The compiler code necessary to support the env! extension.  Eventually this
// should all get sucked into either the compiler syntax extension plugin
// interface.
//

use syntax::ast::{self, Ident, GenericArg};
use syntax::ext::base::{self, *};
use syntax::ext::build::AstBuilder;
use syntax::symbol::{kw, sym, Symbol};
use syntax_pos::Span;
use syntax::tokenstream;

use std::env;

pub fn expand_option_env<'cx>(cx: &'cx mut ExtCtxt<'_>,
                              sp: Span,
                              tts: &[tokenstream::TokenTree])
                              -> Box<dyn base::MacResult + 'cx> {
    let var = match get_single_str_from_tts(cx, sp, tts, "option_env!") {
        None => return DummyResult::expr(sp),
        Some(v) => v,
    };

    let sp = sp.apply_mark(cx.current_expansion.mark);
    let e = match env::var(&*var.as_str()) {
        Err(..) => {
            let lt = cx.lifetime(sp, Ident::with_empty_ctxt(kw::StaticLifetime));
            cx.expr_path(cx.path_all(sp,
                                     true,
                                     cx.std_path(&[sym::option, sym::Option, sym::None]),
                                     vec![GenericArg::Type(cx.ty_rptr(sp,
                                                     cx.ty_ident(sp,
                                                                 Ident::with_empty_ctxt(sym::str)),
                                                     Some(lt),
                                                     ast::Mutability::Immutable))],
                                     vec![]))
        }
        Ok(s) => {
            cx.expr_call_global(sp,
                                cx.std_path(&[sym::option, sym::Option, sym::Some]),
                                vec![cx.expr_str(sp, Symbol::intern(&s))])
        }
    };
    MacEager::expr(e)
}

pub fn expand_env<'cx>(cx: &'cx mut ExtCtxt<'_>,
                       sp: Span,
                       tts: &[tokenstream::TokenTree])
                       -> Box<dyn base::MacResult + 'cx> {
    let mut exprs = match get_exprs_from_tts(cx, sp, tts) {
        Some(ref exprs) if exprs.is_empty() => {
            cx.span_err(sp, "env! takes 1 or 2 arguments");
            return DummyResult::expr(sp);
        }
        None => return DummyResult::expr(sp),
        Some(exprs) => exprs.into_iter(),
    };

    let var = match expr_to_string(cx, exprs.next().unwrap(), "expected string literal") {
        None => return DummyResult::expr(sp),
        Some((v, _style)) => v,
    };
    let msg = match exprs.next() {
        None => Symbol::intern(&format!("environment variable `{}` not defined", var)),
        Some(second) => {
            match expr_to_string(cx, second, "expected string literal") {
                None => return DummyResult::expr(sp),
                Some((s, _style)) => s,
            }
        }
    };

    if exprs.next().is_some() {
        cx.span_err(sp, "env! takes 1 or 2 arguments");
        return DummyResult::expr(sp);
    }

    let e = match env::var(&*var.as_str()) {
        Err(_) => {
            cx.span_err(sp, &msg.as_str());
            return DummyResult::expr(sp);
        }
        Ok(s) => cx.expr_str(sp, Symbol::intern(&s)),
    };
    MacEager::expr(e)
}

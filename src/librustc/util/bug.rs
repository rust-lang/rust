// These functions are used by macro expansion for bug! and span_bug!

use crate::ty::tls;
use std::fmt;
use syntax_pos::{Span, MultiSpan};

#[cold]
#[inline(never)]
pub fn bug_fmt(file: &'static str, line: u32, args: fmt::Arguments<'_>) -> ! {
    // this wrapper mostly exists so I don't have to write a fully
    // qualified path of None::<Span> inside the bug!() macro definition
    opt_span_bug_fmt(file, line, None::<Span>, args);
}

#[cold]
#[inline(never)]
pub fn span_bug_fmt<S: Into<MultiSpan>>(
    file: &'static str,
    line: u32,
    span: S,
    args: fmt::Arguments<'_>,
) -> ! {
    opt_span_bug_fmt(file, line, Some(span), args);
}

fn opt_span_bug_fmt<S: Into<MultiSpan>>(
    file: &'static str,
    line: u32,
    span: Option<S>,
    args: fmt::Arguments<'_>,
) -> ! {
    tls::with_opt(move |tcx| {
        let msg = format!("{}:{}: {}", file, line, args);
        match (tcx, span) {
            (Some(tcx), Some(span)) => tcx.sess.diagnostic().span_bug(span, &msg),
            (Some(tcx), None) => tcx.sess.diagnostic().bug(&msg),
            (None, _) => panic!(msg),
        }
    });
    unreachable!();
}

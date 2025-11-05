use rustc_ast::token::Delimiter;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::*;
use rustc_expand::base::*;
use rustc_span::edition::Edition;
use rustc_span::{Span, Symbol, sym};

// Use an enum to ensure that no new macro calls are added without also updating the message in the
// optimized path below.
enum InnerCall {
    Panic2015,
    Panic2021,
    Unreachable2015,
    Unreachable2021,
}

impl InnerCall {
    fn symbol(&self) -> Symbol {
        match self {
            Self::Panic2015 => sym::panic_2015,
            Self::Panic2021 => sym::panic_2021,
            Self::Unreachable2015 => sym::unreachable_2015,
            Self::Unreachable2021 => sym::unreachable_2021,
        }
    }
}

/// This expands to either
/// - `$crate::panic::panic_2015!(...)` or
/// - `$crate::panic::panic_2021!(...)`
/// depending on the edition. If the entire message is known at compile time,
/// `core::panicking::panic` may be called as an optimization.
///
/// This is used for both std::panic!() and core::panic!().
///
/// `$crate` will refer to either the `std` or `core` crate depending on which
/// one we're expanding from.
pub(crate) fn expand_panic<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let mac = if use_panic_2021(sp) { InnerCall::Panic2021 } else { InnerCall::Panic2015 };
    expand(mac, cx, sp, tts)
}

/// This expands to either
/// - `$crate::panic::unreachable_2015!(...)` or
/// - `$crate::panic::unreachable_2021!(...)`
/// depending on the edition. If the entire message is known at compile time,
/// `core::panicking::panic` may be called as an optimization.
pub(crate) fn expand_unreachable<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let mac =
        if use_panic_2021(sp) { InnerCall::Unreachable2021 } else { InnerCall::Unreachable2015 };
    expand(mac, cx, sp, tts)
}

fn expand<'cx>(
    mac: InnerCall,
    cx: &'cx ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let sp = cx.with_call_site_ctxt(sp);

    ExpandResult::Ready(MacEager::expr(
        cx.expr_macro_call(
            sp,
            cx.macro_call(
                sp,
                Path {
                    span: sp,
                    segments: cx
                        .std_path(&[sym::panic, mac.symbol()])
                        .into_iter()
                        .map(PathSegment::from_ident)
                        .collect(),
                    tokens: None,
                },
                Delimiter::Parenthesis,
                tts,
            ),
        ),
    ))
}

pub(crate) fn use_panic_2021(mut span: Span) -> bool {
    // To determine the edition, we check the first span up the expansion
    // stack that does not have #[allow_internal_unstable(edition_panic)].
    // (To avoid using the edition of e.g. the assert!() or debug_assert!() definition.)
    loop {
        let expn = span.ctxt().outer_expn_data();
        if let Some(features) = expn.allow_internal_unstable
            && features.contains(&sym::edition_panic)
        {
            span = expn.call_site;
            continue;
        }
        break expn.edition >= Edition::Edition2021;
    }
}

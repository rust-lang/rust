use rustc_ast::token::Delimiter;
use rustc_ast::tokenstream::{DelimSpan, TokenStream};
use rustc_ast::*;
use rustc_expand::base::*;
use rustc_span::edition::Edition;
use rustc_span::{Span, sym};

/// This expands to either
/// - `$crate::panic::panic_2015!(...)` or
/// - `$crate::panic::panic_2021!(...)`
/// depending on the edition.
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
    let mac = if use_panic_2021(sp) { sym::panic_2021 } else { sym::panic_2015 };
    expand(mac, cx, sp, tts)
}

/// This expands to either
/// - `$crate::panic::unreachable_2015!(...)` or
/// - `$crate::panic::unreachable_2021!(...)`
/// depending on the edition.
pub(crate) fn expand_unreachable<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let mac = if use_panic_2021(sp) { sym::unreachable_2021 } else { sym::unreachable_2015 };
    expand(mac, cx, sp, tts)
}

fn expand<'cx>(
    mac: rustc_span::Symbol,
    cx: &'cx ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let sp = cx.with_call_site_ctxt(sp);

    ExpandResult::Ready(MacEager::expr(
        cx.expr(
            sp,
            ExprKind::MacCall(Box::new(MacCall {
                path: Path {
                    span: sp,
                    segments: cx
                        .std_path(&[sym::panic, mac])
                        .into_iter()
                        .map(|ident| PathSegment::from_ident(ident))
                        .collect(),
                    tokens: None,
                },
                args: Box::new(DelimArgs {
                    dspan: DelimSpan::from_single(sp),
                    delim: Delimiter::Parenthesis,
                    tokens: tts,
                }),
            })),
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

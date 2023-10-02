use crate::errors;
use rustc_ast::{
    attr::mk_attr,
    ptr::P,
    token,
    tokenstream::{DelimSpan, TokenStream, TokenTree},
    AssocItem, AssocItemKind, AttrArgs, AttrStyle, Attribute, DelimArgs, Item, ItemKind, Path,
    Stmt, StmtKind,
};
use rustc_errors::PResult;
use rustc_expand::base::{DummyResult, ExtCtxt, MacResult};
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_span::{
    symbol::{kw, sym, Ident},
    Span,
};
use smallvec::SmallVec;

type ItemOffset = (usize, bool, TokenStream);

// ```rust
// cfg_match! {
//     cfg(unix) => { fn foo() -> i32 { 1 } },
//     _ => { fn foo() -> i32 { 2 } },
// }
// ```
//
// The above `cfg_match!` is expanded to the following elements:
//
// ```rust
// #[cfg(all(unix, not(any())))]
// fn foo() -> i32 { 1 }
//
// #[cfg(all(not(any(unix))))]
// fn foo() -> i32 { 2 }
// ```
pub fn expand_cfg_match(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let rslt = || {
        let (does_not_have_wildcard, mut items, mut items_offsets) = parse(cx, tts)?;
        iter(
            cx,
            &mut items,
            &mut items_offsets,
            sp,
            |local_cx, no, yes, items| {
                let attr = mk_cfg(local_cx, no, sp, Some(yes));
                items.iter_mut().for_each(|item| item.attrs.push(attr.clone()));
            },
            |local_cx, no, yes, items| {
                let attr = mk_cfg(local_cx, no, sp, does_not_have_wildcard.then_some(yes));
                items.iter_mut().for_each(|item| item.attrs.push(attr.clone()));
            },
        )?;
        PResult::Ok(items)
    };
    match rslt() {
        Err(mut err) => {
            err.emit();
            return DummyResult::any(sp);
        }
        Ok(items) => Box::new(CfgMatchOutput(items)),
    }
}

fn iter<'session>(
    cx: &mut ExtCtxt<'session>,
    items: &mut [P<Item>],
    items_offsets: &[ItemOffset],
    sp: Span,
    mut cb: impl FnMut(&mut ExtCtxt<'session>, &[ItemOffset], &TokenStream, &mut [P<Item>]),
    mut cb_last: impl FnMut(&mut ExtCtxt<'session>, &[ItemOffset], &TokenStream, &mut [P<Item>]),
) -> PResult<'session, ()> {
    match items_offsets {
        [] => {}
        [first] => {
            if first.1 {
                return Err(cx.sess.create_err(errors::CfgMatchMeaninglessArms { span: sp }));
            }
            cb_last(cx, &[], &first.2, items.get_mut(..first.0).unwrap_or_default());
        }
        [first, rest @ .., last] => {
            let mut no_idx = 1;
            let mut offset = first.0;
            let mut prev_offset = 0;
            cb(cx, &[], &first.2, items.get_mut(prev_offset..offset).unwrap_or_default());
            for elem in rest {
                prev_offset = offset;
                offset = elem.0;
                cb(
                    cx,
                    items_offsets.get(..no_idx).unwrap_or_default(),
                    &elem.2,
                    items.get_mut(prev_offset..offset).unwrap_or_default(),
                );
                no_idx = no_idx.wrapping_add(1);
            }
            prev_offset = offset;
            offset = last.0;
            cb_last(
                cx,
                items_offsets.get(..no_idx).unwrap_or_default(),
                &last.2,
                items.get_mut(prev_offset..offset).unwrap_or_default(),
            );
        }
    }
    Ok(())
}

// #[cfg(all(** YES **, not(any(** NO **, ** NO **, ..))))]
fn mk_cfg(
    cx: &mut ExtCtxt<'_>,
    no: &[ItemOffset],
    sp: Span,
    yes: Option<&TokenStream>,
) -> Attribute {
    let mut any_tokens = TokenStream::new(Vec::with_capacity(4));
    if let [first, ref rest @ ..] = no {
        any_tokens.push_stream(first.2.clone());
        for elem in rest.iter() {
            any_tokens.push_tree(TokenTree::token_alone(token::Comma, sp));
            any_tokens.push_stream(elem.2.clone());
        }
    }

    let mut not_tokens = TokenStream::new(Vec::with_capacity(2));
    not_tokens.push_tree(TokenTree::token_alone(token::Ident(sym::any, false), sp));
    not_tokens.push_tree(TokenTree::Delimited(
        DelimSpan::from_single(sp),
        token::Delimiter::Parenthesis,
        any_tokens,
    ));

    let mut all_tokens = TokenStream::new(Vec::with_capacity(4));
    if let Some(elem) = yes {
        all_tokens.push_stream(elem.clone());
        all_tokens.push_tree(TokenTree::token_alone(token::Comma, sp));
    }
    all_tokens.push_tree(TokenTree::token_alone(token::Ident(sym::not, false), sp));
    all_tokens.push_tree(TokenTree::Delimited(
        DelimSpan::from_single(sp),
        token::Delimiter::Parenthesis,
        not_tokens,
    ));

    let mut tokens = TokenStream::new(Vec::with_capacity(2));
    tokens.push_tree(TokenTree::token_alone(token::Ident(sym::all, false), sp));
    tokens.push_tree(TokenTree::Delimited(
        DelimSpan::from_single(sp),
        token::Delimiter::Parenthesis,
        all_tokens,
    ));

    mk_attr(
        &cx.sess.parse_sess.attr_id_generator,
        AttrStyle::Outer,
        Path::from_ident(Ident::new(sym::cfg, sp)),
        AttrArgs::Delimited(DelimArgs {
            dspan: DelimSpan::from_single(sp),
            delim: token::Delimiter::Parenthesis,
            tokens,
        }),
        sp,
    )
}

fn parse<'session>(
    cx: &mut ExtCtxt<'session>,
    tts: TokenStream,
) -> PResult<'session, (bool, Vec<P<Item>>, Vec<ItemOffset>)> {
    let mut parser = cx.new_parser_from_tts(tts);
    if parser.token == token::Eof {
        return parser.unexpected();
    }
    let mut does_not_have_wildcard = true;
    let mut items = Vec::with_capacity(4);
    let mut items_offsets = Vec::with_capacity(4);
    loop {
        match parse_cfg_arm(&mut items, &mut parser)? {
            None => break,
            Some((has_single_elem, ts)) => {
                items_offsets.push((items.len(), has_single_elem, ts));
            }
        }
    }
    if parser.token != token::Eof && !items.is_empty() {
        let has_single_elem = parse_wildcard_arm(&mut items, &mut parser)?;
        does_not_have_wildcard = false;
        items_offsets.push((items.len(), has_single_elem, TokenStream::new(vec![])));
    }
    if parser.token != token::Eof {
        return parser.unexpected();
    }
    Ok((does_not_have_wildcard, items, items_offsets))
}

fn parse_arbitrary_arm_block<'session>(
    items: &mut Vec<P<Item>>,
    mandatory_comma: bool,
    parser: &mut Parser<'session>,
) -> PResult<'session, bool> {
    if parser.eat(&token::OpenDelim(token::Delimiter::Brace)) {
        loop {
            let item = match parser.parse_item(ForceCollect::No) {
                Ok(Some(elem)) => elem,
                _ => break,
            };
            items.push(item);
        }
        parser.expect(&token::CloseDelim(token::Delimiter::Brace))?;
        Ok(false)
    } else {
        let Ok(Some(item)) = parser.parse_item(ForceCollect::No) else {
            return parser.unexpected();
        };
        if !matches!(item.kind, ItemKind::Fn(_)) {
            return Err(parser
                .sess
                .create_err(errors::CfgMatchBadSingleArm { span: parser.token.span }));
        }
        let has_comma = parser.eat(&token::Comma);
        if mandatory_comma && !has_comma {
            return Err(parser
                .sess
                .create_err(errors::CfgMatchMissingComma { span: parser.token.span }));
        }
        items.push(item);
        Ok(true)
    }
}

fn parse_cfg_arm<'session>(
    items: &mut Vec<P<Item>>,
    parser: &mut Parser<'session>,
) -> PResult<'session, Option<(bool, TokenStream)>> {
    if !parser.eat_keyword(sym::cfg) {
        return Ok(None);
    }
    let TokenTree::Delimited(_, delim, tokens) = parser.parse_token_tree() else {
        return parser.unexpected();
    };
    if delim != token::Delimiter::Parenthesis || !parser.eat(&token::FatArrow) {
        return Err(parser.sess.create_err(errors::CfgMatchBadArm { span: parser.token.span }));
    }
    let has_single_elem = parse_arbitrary_arm_block(items, true, parser)?;
    Ok(Some((has_single_elem, tokens)))
}

fn parse_wildcard_arm<'session>(
    items: &mut Vec<P<Item>>,
    parser: &mut Parser<'session>,
) -> PResult<'session, bool> {
    if !parser.eat_keyword(kw::Underscore) || !parser.eat(&token::FatArrow) {
        return Err(parser
            .sess
            .create_err(errors::CfgMatchBadWildcard { span: parser.token.span }));
    }
    parse_arbitrary_arm_block(items, false, parser)
}

struct CfgMatchOutput(Vec<P<Item>>);

impl MacResult for CfgMatchOutput {
    fn make_impl_items(self: Box<Self>) -> Option<SmallVec<[P<AssocItem>; 1]>> {
        let mut rslt = SmallVec::with_capacity(self.0.len());
        rslt.extend(self.0.into_iter().filter_map(|item| {
            let Item { attrs, id, span, vis, ident, kind, tokens } = item.into_inner();
            let ItemKind::Fn(fun) = kind else {
                return None;
            };
            Some(P(Item { attrs, id, ident, kind: AssocItemKind::Fn(fun), span, tokens, vis }))
        }));
        Some(rslt)
    }

    fn make_items(self: Box<Self>) -> Option<SmallVec<[P<Item>; 1]>> {
        Some(<_>::from(self.0))
    }

    fn make_stmts(self: Box<Self>) -> Option<SmallVec<[Stmt; 1]>> {
        let mut rslt = SmallVec::with_capacity(self.0.len());
        rslt.extend(self.0.into_iter().map(|item| {
            let id = item.id;
            let span = item.span;
            Stmt { id, kind: StmtKind::Item(item), span }
        }));
        Some(rslt)
    }
}

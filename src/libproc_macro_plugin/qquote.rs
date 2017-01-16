// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Quasiquoter
//! This file contains the implementation internals of the quasiquoter provided by `quote!`.
//!
//! ## Ouput
//! The quasiquoter produces output of the form:
//! let tmp0 = ...;
//! let tmp1 = ...;
//! ...
//! concat(from_tokens(...), concat(...))
//!
//! To the more explicit, the quasiquoter produces a series of bindings that each
//! construct TokenStreams via constructing Tokens and using `from_tokens`, ultimately
//! invoking `concat` on these bindings (and inlined expressions) to construct a
//! TokenStream that resembles the output syntax.
//!

use proc_macro_tokens::build::*;
use proc_macro_tokens::parse::lex;

use qquote::int_build::*;

use syntax::ast::Ident;
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::ext::base;
use syntax::ext::proc_macro_shim::build_block_emitter;
use syntax::parse::token::{self, Token};
use syntax::print::pprust;
use syntax::symbol::Symbol;
use syntax::tokenstream::{TokenTree, TokenStream};

// ____________________________________________________________________________________________
// Main definition
/// The user should use the macro, not this procedure.
pub fn qquote<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[TokenTree])
                   -> Box<base::MacResult + 'cx> {

    debug!("\nTTs in: {:?}\n", pprust::tts_to_string(&tts[..]));
    let output = qquoter(cx, TokenStream::from_tts(tts.clone().to_owned()));
    debug!("\nQQ out: {}\n", pprust::tts_to_string(&output.to_tts()[..]));
    let imports = concat(lex("use syntax::ext::proc_macro_shim::prelude::*;"),
                         lex("use proc_macro_tokens::prelude::*;"));
    build_block_emitter(cx, sp, build_brace_delimited(concat(imports, output)))
}

// ____________________________________________________________________________________________
// Datatype Definitions

#[derive(Debug)]
struct QDelimited {
    delim: token::DelimToken,
    open_span: Span,
    tts: Vec<Qtt>,
    close_span: Span,
}

#[derive(Debug)]
enum Qtt {
    TT(TokenTree),
    Delimited(QDelimited),
    QIdent(TokenTree),
}

type Bindings = Vec<(Ident, TokenStream)>;

// ____________________________________________________________________________________________
// Quasiquoter Algorithm
// This algorithm works as follows:
// Input: TokenStream
// 1. Walk the TokenStream, gathering up the unquoted expressions and marking them separately.
// 2. Hoist any unquoted term into its own let-binding via a gensym'd identifier
// 3. Convert the body from a `complex expression` into a simplified one via `convert_complex_tts
// 4. Stitch everything together with `concat`.
fn qquoter<'cx>(cx: &'cx mut ExtCtxt, ts: TokenStream) -> TokenStream {
    if ts.is_empty() {
        return lex("TokenStream::mk_empty()");
    }
    let qq_res = qquote_iter(cx, 0, ts);
    let mut bindings = qq_res.0;
    let body = qq_res.1;
    let mut cct_res = convert_complex_tts(cx, body);

    bindings.append(&mut cct_res.0);

    if bindings.is_empty() {
        cct_res.1
    } else {
        debug!("BINDINGS");
        for b in bindings.clone() {
            debug!("{:?} = {}", b.0, pprust::tts_to_string(&b.1.to_tts()[..]));
        }
        TokenStream::concat(unravel(bindings), cct_res.1)
   }
}

fn qquote_iter<'cx>(cx: &'cx mut ExtCtxt, depth: i64, ts: TokenStream) -> (Bindings, Vec<Qtt>) {
    let mut depth = depth;
    let mut bindings: Bindings = Vec::new();
    let mut output: Vec<Qtt> = Vec::new();

    let mut iter = ts.iter();

    loop {
        let next = iter.next();
        if next.is_none() {
            break;
        }
        let next = next.unwrap().clone();
        match next {
            TokenTree::Token(_, Token::Ident(id)) if is_unquote(id) => {
                if depth == 0 {
                    let exp = iter.next();
                    if exp.is_none() {
                        break;
                    } // produce an error or something first
                    let exp = vec![exp.unwrap().to_owned()];
                    debug!("RHS: {:?}", exp.clone());
                    let new_id = Ident::with_empty_ctxt(Symbol::gensym("tmp"));
                    debug!("RHS TS: {:?}", TokenStream::from_tts(exp.clone()));
                    debug!("RHS TS TT: {:?}", TokenStream::from_tts(exp.clone()).to_vec());
                    bindings.push((new_id, TokenStream::from_tts(exp)));
                    debug!("BINDINGS");
                    for b in bindings.clone() {
                        debug!("{:?} = {}", b.0, pprust::tts_to_string(&b.1.to_tts()[..]));
                    }
                    output.push(Qtt::QIdent(as_tt(Token::Ident(new_id.clone()))));
                } else {
                    depth = depth - 1;
                    output.push(Qtt::TT(next.clone()));
                }
            }
            TokenTree::Token(_, Token::Ident(id)) if is_qquote(id) => {
                depth = depth + 1;
            }
            TokenTree::Delimited(_, ref dl) => {
                let br = qquote_iter(cx, depth, TokenStream::from_tts(dl.tts.clone().to_owned()));
                let mut nested_bindings = br.0;
                let nested = br.1;
                bindings.append(&mut nested_bindings);

                let new_dl = QDelimited {
                    delim: dl.delim,
                    open_span: dl.open_span,
                    tts: nested,
                    close_span: dl.close_span,
                };

                output.push(Qtt::Delimited(new_dl));
            }
            t => {
                output.push(Qtt::TT(t));
            }
        }
    }

    (bindings, output)
}

// ____________________________________________________________________________________________
// Turns QQTs into a TokenStream and some Bindings.
/// Construct a chain of concatenations.
fn unravel_concats(tss: Vec<TokenStream>) -> TokenStream {
    let mut pushes: Vec<TokenStream> =
        tss.into_iter().filter(|&ref ts| !ts.is_empty()).collect();
    let mut output = match pushes.pop() {
        Some(ts) => ts,
        None => {
            return TokenStream::mk_empty();
        }
    };

    while let Some(ts) = pushes.pop() {
        output = build_fn_call(Ident::from_str("concat"),
                               concat(concat(ts,
                                             from_tokens(vec![Token::Comma])),
                                      output));
    }
    output
}

/// This converts the vector of Qtts into a set of Bindings for construction and the main
/// body as a TokenStream.
fn convert_complex_tts<'cx>(cx: &'cx mut ExtCtxt, tts: Vec<Qtt>) -> (Bindings, TokenStream) {
    let mut pushes: Vec<TokenStream> = Vec::new();
    let mut bindings: Bindings = Vec::new();

    let mut iter = tts.into_iter();

    loop {
        let next = iter.next();
        if next.is_none() {
            break;
        }
        let next = next.unwrap();
        match next {
            Qtt::TT(TokenTree::Token(_, t)) => {
                let token_out = emit_token(t);
                pushes.push(token_out);
            }
            // FIXME handle sequence repetition tokens
            Qtt::Delimited(qdl) => {
                debug!("  Delimited: {:?} ", qdl.tts);
                let fresh_id = Ident::with_empty_ctxt(Symbol::gensym("qdl_tmp"));
                let (mut nested_bindings, nested_toks) = convert_complex_tts(cx, qdl.tts);

                let body = if nested_toks.is_empty() {
                    assert!(nested_bindings.is_empty());
                    build_mod_call(vec![Ident::from_str("TokenStream"),
                                        Ident::from_str("mk_empty")],
                                   TokenStream::mk_empty())
                } else {
                    bindings.append(&mut nested_bindings);
                    bindings.push((fresh_id, nested_toks));
                    TokenStream::from_tokens(vec![Token::Ident(fresh_id)])
                };

                let delimitiers = build_delim_tok(qdl.delim);

                pushes.push(build_mod_call(vec![Ident::from_str("proc_macro_tokens"),
                                                Ident::from_str("build"),
                                                Ident::from_str("build_delimited")],
                                           flatten(vec![body,
                                                        lex(","),
                                                        delimitiers].into_iter())));
            }
            Qtt::QIdent(t) => {
                pushes.push(TokenStream::from_tts(vec![t]));
                pushes.push(TokenStream::mk_empty());
            }
            _ => panic!("Unhandled case!"),
        }

    }

    (bindings, unravel_concats(pushes))
}

// ____________________________________________________________________________________________
// Utilities

/// Unravels Bindings into a TokenStream of `let` declarations.
fn unravel(bindings: Bindings) -> TokenStream {
    flatten(bindings.into_iter().map(|(a, b)| build_let(a, b)))
}

/// Checks if the Ident is `unquote`.
fn is_unquote(id: Ident) -> bool {
    let qq = Ident::from_str("unquote");
    id.name == qq.name  // We disregard context; unquote is _reserved_
}

/// Checks if the Ident is `quote`.
fn is_qquote(id: Ident) -> bool {
    let qq = Ident::from_str("qquote");
    id.name == qq.name  // We disregard context; qquote is _reserved_
}

mod int_build {
    use proc_macro_tokens::build::*;
    use proc_macro_tokens::parse::*;

    use syntax::ast::{self, Ident};
    use syntax::codemap::{DUMMY_SP};
    use syntax::parse::token::{self, Token, Lit};
    use syntax::symbol::keywords;
    use syntax::tokenstream::{TokenTree, TokenStream};

    // ____________________________________________________________________________________________
    // Emitters

    pub fn emit_token(t: Token) -> TokenStream {
        concat(lex("TokenStream::from_tokens"),
               build_paren_delimited(build_vec(build_token_tt(t))))
    }

    pub fn emit_lit(l: Lit, n: Option<ast::Name>) -> TokenStream {
        let suf = match n {
            Some(n) => format!("Some(ast::Name({}))", n.as_u32()),
            None => "None".to_string(),
        };

        let lit = match l {
            Lit::Byte(n) => format!("Lit::Byte(Symbol::intern(\"{}\"))", n.to_string()),
            Lit::Char(n) => format!("Lit::Char(Symbol::intern(\"{}\"))", n.to_string()),
            Lit::Float(n) => format!("Lit::Float(Symbol::intern(\"{}\"))", n.to_string()),
            Lit::Str_(n) => format!("Lit::Str_(Symbol::intern(\"{}\"))", n.to_string()),
            Lit::Integer(n) => format!("Lit::Integer(Symbol::intern(\"{}\"))", n.to_string()),
            Lit::ByteStr(n) => format!("Lit::ByteStr(Symbol::intern(\"{}\"))", n.to_string()),
            _ => panic!("Unsupported literal"),
        };

        let res = format!("Token::Literal({},{})", lit, suf);
        debug!("{}", res);
        lex(&res)
    }

    // ____________________________________________________________________________________________
    // Token Builders

    pub fn build_binop_tok(bot: token::BinOpToken) -> TokenStream {
        match bot {
            token::BinOpToken::Plus => lex("Token::BinOp(BinOpToken::Plus)"),
            token::BinOpToken::Minus => lex("Token::BinOp(BinOpToken::Minus)"),
            token::BinOpToken::Star => lex("Token::BinOp(BinOpToken::Star)"),
            token::BinOpToken::Slash => lex("Token::BinOp(BinOpToken::Slash)"),
            token::BinOpToken::Percent => lex("Token::BinOp(BinOpToken::Percent)"),
            token::BinOpToken::Caret => lex("Token::BinOp(BinOpToken::Caret)"),
            token::BinOpToken::And => lex("Token::BinOp(BinOpToken::And)"),
            token::BinOpToken::Or => lex("Token::BinOp(BinOpToken::Or)"),
            token::BinOpToken::Shl => lex("Token::BinOp(BinOpToken::Shl)"),
            token::BinOpToken::Shr => lex("Token::BinOp(BinOpToken::Shr)"),
        }
    }

    pub fn build_binopeq_tok(bot: token::BinOpToken) -> TokenStream {
        match bot {
            token::BinOpToken::Plus => lex("Token::BinOpEq(BinOpToken::Plus)"),
            token::BinOpToken::Minus => lex("Token::BinOpEq(BinOpToken::Minus)"),
            token::BinOpToken::Star => lex("Token::BinOpEq(BinOpToken::Star)"),
            token::BinOpToken::Slash => lex("Token::BinOpEq(BinOpToken::Slash)"),
            token::BinOpToken::Percent => lex("Token::BinOpEq(BinOpToken::Percent)"),
            token::BinOpToken::Caret => lex("Token::BinOpEq(BinOpToken::Caret)"),
            token::BinOpToken::And => lex("Token::BinOpEq(BinOpToken::And)"),
            token::BinOpToken::Or => lex("Token::BinOpEq(BinOpToken::Or)"),
            token::BinOpToken::Shl => lex("Token::BinOpEq(BinOpToken::Shl)"),
            token::BinOpToken::Shr => lex("Token::BinOpEq(BinOpToken::Shr)"),
        }
    }

    pub fn build_delim_tok(dt: token::DelimToken) -> TokenStream {
        match dt {
            token::DelimToken::Paren => lex("DelimToken::Paren"),
            token::DelimToken::Bracket => lex("DelimToken::Bracket"),
            token::DelimToken::Brace => lex("DelimToken::Brace"),
            token::DelimToken::NoDelim => lex("DelimToken::NoDelim"),
        }
    }

    pub fn build_token_tt(t: Token) -> TokenStream {
        match t {
            Token::Eq => lex("Token::Eq"),
            Token::Lt => lex("Token::Lt"),
            Token::Le => lex("Token::Le"),
            Token::EqEq => lex("Token::EqEq"),
            Token::Ne => lex("Token::Ne"),
            Token::Ge => lex("Token::Ge"),
            Token::Gt => lex("Token::Gt"),
            Token::AndAnd => lex("Token::AndAnd"),
            Token::OrOr => lex("Token::OrOr"),
            Token::Not => lex("Token::Not"),
            Token::Tilde => lex("Token::Tilde"),
            Token::BinOp(tok) => build_binop_tok(tok),
            Token::BinOpEq(tok) => build_binopeq_tok(tok),
            Token::At => lex("Token::At"),
            Token::Dot => lex("Token::Dot"),
            Token::DotDot => lex("Token::DotDot"),
            Token::DotDotDot => lex("Token::DotDotDot"),
            Token::Comma => lex("Token::Comma"),
            Token::Semi => lex("Token::Semi"),
            Token::Colon => lex("Token::Colon"),
            Token::ModSep => lex("Token::ModSep"),
            Token::RArrow => lex("Token::RArrow"),
            Token::LArrow => lex("Token::LArrow"),
            Token::FatArrow => lex("Token::FatArrow"),
            Token::Pound => lex("Token::Pound"),
            Token::Dollar => lex("Token::Dollar"),
            Token::Question => lex("Token::Question"),
            Token::OpenDelim(dt) => {
                match dt {
                    token::DelimToken::Paren => lex("Token::OpenDelim(DelimToken::Paren)"),
                    token::DelimToken::Bracket => lex("Token::OpenDelim(DelimToken::Bracket)"),
                    token::DelimToken::Brace => lex("Token::OpenDelim(DelimToken::Brace)"),
                    token::DelimToken::NoDelim => lex("DelimToken::NoDelim"),
                }
            }
            Token::CloseDelim(dt) => {
                match dt {
                    token::DelimToken::Paren => lex("Token::CloseDelim(DelimToken::Paren)"),
                    token::DelimToken::Bracket => lex("Token::CloseDelim(DelimToken::Bracket)"),
                    token::DelimToken::Brace => lex("Token::CloseDelim(DelimToken::Brace)"),
                    token::DelimToken::NoDelim => lex("DelimToken::NoDelim"),
                }
            }
            Token::Underscore => lex("_"),
            Token::Literal(lit, sfx) => emit_lit(lit, sfx),
            // fix ident expansion information... somehow
            Token::Ident(ident) =>
                lex(&format!("Token::Ident(Ident::from_str(\"{}\"))", ident.name)),
            Token::Lifetime(ident) =>
                lex(&format!("Token::Ident(Ident::from_str(\"{}\"))", ident.name)),
            _ => panic!("Unhandled case!"),
        }
    }

    // ____________________________________________________________________________________________
    // Conversion operators

    pub fn as_tt(t: Token) -> TokenTree {
        // FIXME do something nicer with the spans
        TokenTree::Token(DUMMY_SP, t)
    }

    // ____________________________________________________________________________________________
    // Build Procedures

    /// Takes `input` and returns `vec![input]`.
    pub fn build_vec(ts: TokenStream) -> TokenStream {
        build_mac_call(Ident::from_str("vec"), ts)
        // tts.clone().to_owned()
    }

    /// Takes `ident` and `rhs` and produces `let ident = rhs;`.
    pub fn build_let(id: Ident, tts: TokenStream) -> TokenStream {
        concat(from_tokens(vec![keyword_to_token_ident(keywords::Let),
                                Token::Ident(id),
                                Token::Eq]),
               concat(tts, from_tokens(vec![Token::Semi])))
    }

    /// Takes `ident ...`, and `args ...` and produces `ident::...(args ...)`.
    pub fn build_mod_call(ids: Vec<Ident>, args: TokenStream) -> TokenStream {
        let call = from_tokens(intersperse(ids.into_iter().map(|id| Token::Ident(id)).collect(),
                                     Token::ModSep));
        concat(call, build_paren_delimited(args))
    }

    /// Takes `ident` and `args ...` and produces `ident(args ...)`.
    pub fn build_fn_call(name: Ident, args: TokenStream) -> TokenStream {
        concat(from_tokens(vec![Token::Ident(name)]), build_paren_delimited(args))
    }

    /// Takes `ident` and `args ...` and produces `ident!(args ...)`.
    pub fn build_mac_call(name: Ident, args: TokenStream) -> TokenStream {
        concat(from_tokens(vec![Token::Ident(name), Token::Not]),
               build_paren_delimited(args))
    }

    // ____________________________________________________________________________________________
    // Utilities

    /// A wrapper around `TokenStream::from_tokens` to avoid extra namespace specification and
    /// provide it as a generic operator.
    pub fn from_tokens(tokens: Vec<Token>) -> TokenStream {
        TokenStream::from_tokens(tokens)
    }

    pub fn intersperse<T>(vs: Vec<T>, t: T) -> Vec<T>
        where T: Clone
    {
        if vs.len() < 2 {
            return vs;
        }
        let mut output = vec![vs.get(0).unwrap().to_owned()];

        for v in vs.into_iter().skip(1) {
            output.push(t.clone());
            output.push(v);
        }
        output
    }
}

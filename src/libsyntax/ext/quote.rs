// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{self, Arg, Arm, Block, Expr, Item, Pat, Stmt, Ty};
use source_map::respan;
use syntax_pos::Span;
use ext::base::ExtCtxt;
use ext::base;
use ext::build::AstBuilder;
use parse::parser::{Parser, PathStyle};
use parse::token;
use ptr::P;
use tokenstream::{TokenStream, TokenTree};

/// Quasiquoting works via token trees.
///
/// This is registered as a set of expression syntax extension called quote!
/// that lifts its argument token-tree to an AST representing the
/// construction of the same token tree, with `token::SubstNt` interpreted
/// as antiquotes (splices).

pub mod rt {
    use ast;
    use source_map::Spanned;
    use ext::base::ExtCtxt;
    use parse::{self, classify};
    use parse::token::{self, Token};
    use ptr::P;
    use symbol::Symbol;
    use ThinVec;

    use tokenstream::{self, TokenTree, TokenStream};

    pub use parse::new_parser_from_tts;
    pub use syntax_pos::{BytePos, Span, DUMMY_SP, FileName};
    pub use source_map::{dummy_spanned};

    pub trait ToTokens {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree>;
    }

    impl ToTokens for TokenTree {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![self.clone()]
        }
    }

    impl<T: ToTokens> ToTokens for Vec<T> {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            self.iter().flat_map(|t| t.to_tokens(cx)).collect()
        }
    }

    impl<T: ToTokens> ToTokens for Spanned<T> {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            // FIXME: use the span?
            self.node.to_tokens(cx)
        }
    }

    impl<T: ToTokens> ToTokens for Option<T> {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            match *self {
                Some(ref t) => t.to_tokens(cx),
                None => Vec::new(),
            }
        }
    }

    impl ToTokens for ast::Ident {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![TokenTree::Token(self.span, Token::from_ast_ident(*self))]
        }
    }

    impl ToTokens for ast::Path {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtPath(self.clone());
            vec![TokenTree::Token(DUMMY_SP, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Ty {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtTy(P(self.clone()));
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Block {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtBlock(P(self.clone()));
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Generics {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtGenerics(self.clone());
            vec![TokenTree::Token(DUMMY_SP, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::WhereClause {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtWhereClause(self.clone());
            vec![TokenTree::Token(DUMMY_SP, Token::interpolated(nt))]
        }
    }

    impl ToTokens for P<ast::Item> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtItem(self.clone());
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::ImplItem {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtImplItem(self.clone());
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for P<ast::ImplItem> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtImplItem((**self).clone());
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::TraitItem {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtTraitItem(self.clone());
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Stmt {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtStmt(self.clone());
            let mut tts = vec![TokenTree::Token(self.span, Token::interpolated(nt))];

            // Some statements require a trailing semicolon.
            if classify::stmt_ends_with_semi(&self.node) {
                tts.push(TokenTree::Token(self.span, token::Semi));
            }

            tts
        }
    }

    impl ToTokens for P<ast::Expr> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtExpr(self.clone());
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for P<ast::Pat> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtPat(self.clone());
            vec![TokenTree::Token(self.span, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Arm {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtArm(self.clone());
            vec![TokenTree::Token(DUMMY_SP, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Arg {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtArg(self.clone());
            vec![TokenTree::Token(DUMMY_SP, Token::interpolated(nt))]
        }
    }

    impl ToTokens for P<ast::Block> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtBlock(self.clone());
            vec![TokenTree::Token(DUMMY_SP, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Lifetime {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![TokenTree::Token(self.ident.span, token::Lifetime(self.ident))]
        }
    }

    macro_rules! impl_to_tokens_slice {
        ($t: ty, $sep: expr) => {
            impl ToTokens for [$t] {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    let mut v = vec![];
                    for (i, x) in self.iter().enumerate() {
                        if i > 0 {
                            v.extend_from_slice(&$sep);
                        }
                        v.extend(x.to_tokens(cx));
                    }
                    v
                }
            }
        };
    }

    impl_to_tokens_slice! { ast::Ty, [TokenTree::Token(DUMMY_SP, token::Comma)] }
    impl_to_tokens_slice! { P<ast::Item>, [] }
    impl_to_tokens_slice! { ast::Arg, [TokenTree::Token(DUMMY_SP, token::Comma)] }

    impl ToTokens for ast::MetaItem {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let nt = token::NtMeta(self.clone());
            vec![TokenTree::Token(DUMMY_SP, Token::interpolated(nt))]
        }
    }

    impl ToTokens for ast::Attribute {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let mut r = vec![];
            // FIXME: The spans could be better
            r.push(TokenTree::Token(self.span, token::Pound));
            if self.style == ast::AttrStyle::Inner {
                r.push(TokenTree::Token(self.span, token::Not));
            }
            let mut inner = Vec::new();
            for (i, segment) in self.path.segments.iter().enumerate() {
                if i > 0 {
                    inner.push(TokenTree::Token(self.span, token::Colon).into());
                }
                inner.push(TokenTree::Token(
                    self.span, token::Token::from_ast_ident(segment.ident)
                ).into());
            }
            inner.push(self.tokens.clone());

            r.push(TokenTree::Delimited(self.span, tokenstream::Delimited {
                delim: token::Bracket, tts: TokenStream::concat(inner).into()
            }));
            r
        }
    }

    impl ToTokens for str {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            let lit = ast::LitKind::Str(Symbol::intern(self), ast::StrStyle::Cooked);
            dummy_spanned(lit).to_tokens(cx)
        }
    }

    impl ToTokens for () {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![TokenTree::Delimited(DUMMY_SP, tokenstream::Delimited {
                delim: token::Paren,
                tts: TokenStream::empty().into(),
            })]
        }
    }

    impl ToTokens for ast::Lit {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            // FIXME: This is wrong
            P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: ast::ExprKind::Lit(P(self.clone())),
                span: DUMMY_SP,
                attrs: ThinVec::new(),
            }).to_tokens(cx)
        }
    }

    impl ToTokens for bool {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            dummy_spanned(ast::LitKind::Bool(*self)).to_tokens(cx)
        }
    }

    impl ToTokens for char {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            dummy_spanned(ast::LitKind::Char(*self)).to_tokens(cx)
        }
    }

    macro_rules! impl_to_tokens_int {
        (signed, $t:ty, $tag:expr) => (
            impl ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    let val = if *self < 0 {
                        -self
                    } else {
                        *self
                    };
                    let lit = ast::LitKind::Int(val as u128, ast::LitIntType::Signed($tag));
                    let lit = P(ast::Expr {
                        id: ast::DUMMY_NODE_ID,
                        node: ast::ExprKind::Lit(P(dummy_spanned(lit))),
                        span: DUMMY_SP,
                        attrs: ThinVec::new(),
                    });
                    if *self >= 0 {
                        return lit.to_tokens(cx);
                    }
                    P(ast::Expr {
                        id: ast::DUMMY_NODE_ID,
                        node: ast::ExprKind::Unary(ast::UnOp::Neg, lit),
                        span: DUMMY_SP,
                        attrs: ThinVec::new(),
                    }).to_tokens(cx)
                }
            }
        );
        (unsigned, $t:ty, $tag:expr) => (
            impl ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    let lit = ast::LitKind::Int(*self as u128, ast::LitIntType::Unsigned($tag));
                    dummy_spanned(lit).to_tokens(cx)
                }
            }
        );
    }

    impl_to_tokens_int! { signed, isize, ast::IntTy::Isize }
    impl_to_tokens_int! { signed, i8,  ast::IntTy::I8 }
    impl_to_tokens_int! { signed, i16, ast::IntTy::I16 }
    impl_to_tokens_int! { signed, i32, ast::IntTy::I32 }
    impl_to_tokens_int! { signed, i64, ast::IntTy::I64 }

    impl_to_tokens_int! { unsigned, usize, ast::UintTy::Usize }
    impl_to_tokens_int! { unsigned, u8,   ast::UintTy::U8 }
    impl_to_tokens_int! { unsigned, u16,  ast::UintTy::U16 }
    impl_to_tokens_int! { unsigned, u32,  ast::UintTy::U32 }
    impl_to_tokens_int! { unsigned, u64,  ast::UintTy::U64 }

    pub trait ExtParseUtils {
        fn parse_item(&self, s: String) -> P<ast::Item>;
        fn parse_expr(&self, s: String) -> P<ast::Expr>;
        fn parse_stmt(&self, s: String) -> ast::Stmt;
        fn parse_tts(&self, s: String) -> Vec<TokenTree>;
    }

    impl<'a> ExtParseUtils for ExtCtxt<'a> {
        fn parse_item(&self, s: String) -> P<ast::Item> {
            panictry!(parse::parse_item_from_source_str(
                FileName::QuoteExpansion,
                s,
                self.parse_sess())).expect("parse error")
        }

        fn parse_stmt(&self, s: String) -> ast::Stmt {
            panictry!(parse::parse_stmt_from_source_str(
                FileName::QuoteExpansion,
                s,
                self.parse_sess())).expect("parse error")
        }

        fn parse_expr(&self, s: String) -> P<ast::Expr> {
            panictry!(parse::parse_expr_from_source_str(
                FileName::QuoteExpansion,
                s,
                self.parse_sess()))
        }

        fn parse_tts(&self, s: String) -> Vec<TokenTree> {
            let source_name = FileName::QuoteExpansion;
            parse::parse_stream_from_source_str(source_name, s, self.parse_sess(), None)
                .into_trees().collect()
        }
    }
}

// Replaces `Token::OpenDelim .. Token::CloseDelim` with `TokenTree::Delimited(..)`.
pub fn unflatten(tts: Vec<TokenTree>) -> Vec<TokenTree> {
    use tokenstream::Delimited;

    let mut results = Vec::new();
    let mut result = Vec::new();
    for tree in tts {
        match tree {
            TokenTree::Token(_, token::OpenDelim(..)) => {
                results.push(::std::mem::replace(&mut result, Vec::new()));
            }
            TokenTree::Token(span, token::CloseDelim(delim)) => {
                let tree = TokenTree::Delimited(span, Delimited {
                    delim,
                    tts: result.into_iter().map(TokenStream::from).collect::<TokenStream>().into(),
                });
                result = results.pop().unwrap();
                result.push(tree);
            }
            tree => result.push(tree),
        }
    }
    result
}

// These panicking parsing functions are used by the quote_*!() syntax extensions,
// but shouldn't be used otherwise.
pub fn parse_expr_panic(parser: &mut Parser) -> P<Expr> {
    panictry!(parser.parse_expr())
}

pub fn parse_item_panic(parser: &mut Parser) -> Option<P<Item>> {
    panictry!(parser.parse_item())
}

pub fn parse_pat_panic(parser: &mut Parser) -> P<Pat> {
    panictry!(parser.parse_pat())
}

pub fn parse_arm_panic(parser: &mut Parser) -> Arm {
    panictry!(parser.parse_arm())
}

pub fn parse_ty_panic(parser: &mut Parser) -> P<Ty> {
    panictry!(parser.parse_ty())
}

pub fn parse_stmt_panic(parser: &mut Parser) -> Option<Stmt> {
    panictry!(parser.parse_stmt())
}

pub fn parse_attribute_panic(parser: &mut Parser, permit_inner: bool) -> ast::Attribute {
    panictry!(parser.parse_attribute(permit_inner))
}

pub fn parse_arg_panic(parser: &mut Parser) -> Arg {
    panictry!(parser.parse_arg())
}

pub fn parse_block_panic(parser: &mut Parser) -> P<Block> {
    panictry!(parser.parse_block())
}

pub fn parse_meta_item_panic(parser: &mut Parser) -> ast::MetaItem {
    panictry!(parser.parse_meta_item())
}

pub fn parse_path_panic(parser: &mut Parser, mode: PathStyle) -> ast::Path {
    panictry!(parser.parse_path(mode))
}

pub fn expand_quote_tokens<'cx>(cx: &'cx mut ExtCtxt,
                                sp: Span,
                                tts: &[TokenTree])
                                -> Box<dyn base::MacResult+'cx> {
    let (cx_expr, expr) = expand_tts(cx, sp, tts);
    let expanded = expand_wrapper(cx, sp, cx_expr, expr, &[&["syntax", "ext", "quote", "rt"]]);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_expr<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[TokenTree])
                              -> Box<dyn base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_expr_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_item<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[TokenTree])
                              -> Box<dyn base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_item_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_pat<'cx>(cx: &'cx mut ExtCtxt,
                             sp: Span,
                             tts: &[TokenTree])
                             -> Box<dyn base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_pat_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_arm(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[TokenTree])
                        -> Box<dyn base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_arm_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_ty(cx: &mut ExtCtxt,
                       sp: Span,
                       tts: &[TokenTree])
                       -> Box<dyn base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_ty_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_stmt(cx: &mut ExtCtxt,
                         sp: Span,
                         tts: &[TokenTree])
                         -> Box<dyn base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_stmt_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_attr(cx: &mut ExtCtxt,
                         sp: Span,
                         tts: &[TokenTree])
                         -> Box<dyn base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_attribute_panic",
                                    vec![cx.expr_bool(sp, true)], tts);

    base::MacEager::expr(expanded)
}

pub fn expand_quote_arg(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[TokenTree])
                        -> Box<dyn base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_arg_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_block(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[TokenTree])
                        -> Box<dyn base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_block_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_meta_item(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[TokenTree])
                        -> Box<dyn base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_meta_item_panic", vec![], tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_path(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[TokenTree])
                        -> Box<dyn base::MacResult+'static> {
    let mode = mk_parser_path(cx, sp, &["PathStyle", "Type"]);
    let expanded = expand_parse_call(cx, sp, "parse_path_panic", vec![mode], tts);
    base::MacEager::expr(expanded)
}

fn ids_ext(strs: Vec<String>) -> Vec<ast::Ident> {
    strs.iter().map(|s| ast::Ident::from_str(s)).collect()
}

fn id_ext(s: &str) -> ast::Ident {
    ast::Ident::from_str(s)
}

// Lift an ident to the expr that evaluates to that ident.
fn mk_ident(cx: &ExtCtxt, sp: Span, ident: ast::Ident) -> P<ast::Expr> {
    let e_str = cx.expr_str(sp, ident.name);
    cx.expr_method_call(sp,
                        cx.expr_ident(sp, id_ext("ext_cx")),
                        id_ext("ident_of"),
                        vec![e_str])
}

// Lift a name to the expr that evaluates to that name
fn mk_name(cx: &ExtCtxt, sp: Span, ident: ast::Ident) -> P<ast::Expr> {
    let e_str = cx.expr_str(sp, ident.name);
    cx.expr_method_call(sp,
                        cx.expr_ident(sp, id_ext("ext_cx")),
                        id_ext("name_of"),
                        vec![e_str])
}

fn mk_tt_path(cx: &ExtCtxt, sp: Span, name: &str) -> P<ast::Expr> {
    let idents = vec![id_ext("syntax"), id_ext("tokenstream"), id_ext("TokenTree"), id_ext(name)];
    cx.expr_path(cx.path_global(sp, idents))
}

fn mk_token_path(cx: &ExtCtxt, sp: Span, name: &str) -> P<ast::Expr> {
    let idents = vec![id_ext("syntax"), id_ext("parse"), id_ext("token"), id_ext(name)];
    cx.expr_path(cx.path_global(sp, idents))
}

fn mk_parser_path(cx: &ExtCtxt, sp: Span, names: &[&str]) -> P<ast::Expr> {
    let mut idents = vec![id_ext("syntax"), id_ext("parse"), id_ext("parser")];
    idents.extend(names.iter().cloned().map(id_ext));
    cx.expr_path(cx.path_global(sp, idents))
}

fn mk_binop(cx: &ExtCtxt, sp: Span, bop: token::BinOpToken) -> P<ast::Expr> {
    let name = match bop {
        token::Plus     => "Plus",
        token::Minus    => "Minus",
        token::Star     => "Star",
        token::Slash    => "Slash",
        token::Percent  => "Percent",
        token::Caret    => "Caret",
        token::And      => "And",
        token::Or       => "Or",
        token::Shl      => "Shl",
        token::Shr      => "Shr"
    };
    mk_token_path(cx, sp, name)
}

fn mk_delim(cx: &ExtCtxt, sp: Span, delim: token::DelimToken) -> P<ast::Expr> {
    let name = match delim {
        token::Paren   => "Paren",
        token::Bracket => "Bracket",
        token::Brace   => "Brace",
        token::NoDelim => "NoDelim",
    };
    mk_token_path(cx, sp, name)
}

#[allow(non_upper_case_globals)]
fn expr_mk_token(cx: &ExtCtxt, sp: Span, tok: &token::Token) -> P<ast::Expr> {
    macro_rules! mk_lit {
        ($name: expr, $suffix: expr, $content: expr $(, $count: expr)*) => {{
            let name = mk_name(cx, sp, ast::Ident::with_empty_ctxt($content));
            let inner = cx.expr_call(sp, mk_token_path(cx, sp, $name), vec![
                name $(, cx.expr_u16(sp, $count))*
            ]);
            let suffix = match $suffix {
                Some(name) => cx.expr_some(sp, mk_name(cx, sp, ast::Ident::with_empty_ctxt(name))),
                None => cx.expr_none(sp)
            };
            cx.expr_call(sp, mk_token_path(cx, sp, "Literal"), vec![inner, suffix])
        }}
    }

    let name = match *tok {
        token::BinOp(binop) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "BinOp"), vec![mk_binop(cx, sp, binop)]);
        }
        token::BinOpEq(binop) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "BinOpEq"),
                                vec![mk_binop(cx, sp, binop)]);
        }

        token::OpenDelim(delim) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "OpenDelim"),
                                vec![mk_delim(cx, sp, delim)]);
        }
        token::CloseDelim(delim) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "CloseDelim"),
                                vec![mk_delim(cx, sp, delim)]);
        }

        token::Literal(token::Byte(i), suf) => return mk_lit!("Byte", suf, i),
        token::Literal(token::Char(i), suf) => return mk_lit!("Char", suf, i),
        token::Literal(token::Integer(i), suf) => return mk_lit!("Integer", suf, i),
        token::Literal(token::Float(i), suf) => return mk_lit!("Float", suf, i),
        token::Literal(token::Str_(i), suf) => return mk_lit!("Str_", suf, i),
        token::Literal(token::StrRaw(i, n), suf) => return mk_lit!("StrRaw", suf, i, n),
        token::Literal(token::ByteStr(i), suf) => return mk_lit!("ByteStr", suf, i),
        token::Literal(token::ByteStrRaw(i, n), suf) => return mk_lit!("ByteStrRaw", suf, i, n),

        token::Ident(ident, is_raw) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "Ident"),
                                vec![mk_ident(cx, sp, ident), cx.expr_bool(sp, is_raw)]);
        }

        token::Lifetime(ident) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "Lifetime"),
                                vec![mk_ident(cx, sp, ident)]);
        }

        token::DocComment(ident) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "DocComment"),
                                vec![mk_name(cx, sp, ast::Ident::with_empty_ctxt(ident))]);
        }

        token::Interpolated(_) => {
            cx.span_err(sp, "quote! with interpolated token");
            // Use dummy name.
            "Interpolated"
        }

        token::Eq           => "Eq",
        token::Lt           => "Lt",
        token::Le           => "Le",
        token::EqEq         => "EqEq",
        token::Ne           => "Ne",
        token::Ge           => "Ge",
        token::Gt           => "Gt",
        token::AndAnd       => "AndAnd",
        token::OrOr         => "OrOr",
        token::Not          => "Not",
        token::Tilde        => "Tilde",
        token::At           => "At",
        token::Dot          => "Dot",
        token::DotDot       => "DotDot",
        token::DotEq        => "DotEq",
        token::DotDotDot    => "DotDotDot",
        token::DotDotEq     => "DotDotEq",
        token::Comma        => "Comma",
        token::Semi         => "Semi",
        token::Colon        => "Colon",
        token::ModSep       => "ModSep",
        token::RArrow       => "RArrow",
        token::LArrow       => "LArrow",
        token::FatArrow     => "FatArrow",
        token::Pound        => "Pound",
        token::Dollar       => "Dollar",
        token::Question     => "Question",
        token::SingleQuote  => "SingleQuote",
        token::Eof          => "Eof",

        token::Whitespace | token::Comment | token::Shebang(_) => {
            panic!("unhandled token in quote!");
        }
    };
    mk_token_path(cx, sp, name)
}

fn statements_mk_tt(cx: &ExtCtxt, tt: &TokenTree, quoted: bool) -> Vec<ast::Stmt> {
    match *tt {
        TokenTree::Token(sp, token::Ident(ident, _)) if quoted => {
            // tt.extend($ident.to_tokens(ext_cx))

            let e_to_toks =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, ident),
                                    id_ext("to_tokens"),
                                    vec![cx.expr_ident(sp, id_ext("ext_cx"))]);
            let e_to_toks =
                cx.expr_method_call(sp, e_to_toks, id_ext("into_iter"), vec![]);

            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("extend"),
                                    vec![e_to_toks]);

            vec![cx.stmt_expr(e_push)]
        }
        TokenTree::Token(sp, ref tok) => {
            let e_sp = cx.expr_ident(sp, id_ext("_sp"));
            let e_tok = cx.expr_call(sp,
                                     mk_tt_path(cx, sp, "Token"),
                                     vec![e_sp, expr_mk_token(cx, sp, tok)]);
            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("push"),
                                    vec![e_tok]);
            vec![cx.stmt_expr(e_push)]
        },
        TokenTree::Delimited(span, ref delimed) => {
            let mut stmts = statements_mk_tt(cx, &delimed.open_tt(span), false);
            stmts.extend(statements_mk_tts(cx, delimed.stream()));
            stmts.extend(statements_mk_tt(cx, &delimed.close_tt(span), false));
            stmts
        }
    }
}

fn parse_arguments_to_quote(cx: &ExtCtxt, tts: &[TokenTree])
                            -> (P<ast::Expr>, Vec<TokenTree>) {
    let mut p = cx.new_parser_from_tts(tts);

    let cx_expr = panictry!(p.parse_expr());
    if !p.eat(&token::Comma) {
        let _ = p.diagnostic().fatal("expected token `,`");
    }

    let tts = panictry!(p.parse_all_token_trees());
    p.abort_if_errors();

    (cx_expr, tts)
}

fn mk_stmts_let(cx: &ExtCtxt, sp: Span) -> Vec<ast::Stmt> {
    // We also bind a single value, sp, to ext_cx.call_site()
    //
    // This causes every span in a token-tree quote to be attributed to the
    // call site of the extension using the quote. We can't really do much
    // better since the source of the quote may well be in a library that
    // was not even parsed by this compilation run, that the user has no
    // source code for (eg. in libsyntax, which they're just _using_).
    //
    // The old quasiquoter had an elaborate mechanism for denoting input
    // file locations from which quotes originated; unfortunately this
    // relied on feeding the source string of the quote back into the
    // compiler (which we don't really want to do) and, in any case, only
    // pushed the problem a very small step further back: an error
    // resulting from a parse of the resulting quote is still attributed to
    // the site the string literal occurred, which was in a source file
    // _other_ than the one the user has control over. For example, an
    // error in a quote from the protocol compiler, invoked in user code
    // using macro_rules! for example, will be attributed to the macro_rules.rs
    // file in libsyntax, which the user might not even have source to (unless
    // they happen to have a compiler on hand). Over all, the phase distinction
    // just makes quotes "hard to attribute". Possibly this could be fixed
    // by recreating some of the original qq machinery in the tt regime
    // (pushing fake SourceFiles onto the parser to account for original sites
    // of quotes, for example) but at this point it seems not likely to be
    // worth the hassle.

    let e_sp = cx.expr_method_call(sp,
                                   cx.expr_ident(sp, id_ext("ext_cx")),
                                   id_ext("call_site"),
                                   Vec::new());

    let stmt_let_sp = cx.stmt_let(sp, false,
                                  id_ext("_sp"),
                                  e_sp);

    let stmt_let_tt = cx.stmt_let(sp, true, id_ext("tt"), cx.expr_vec_ng(sp));

    vec![stmt_let_sp, stmt_let_tt]
}

fn statements_mk_tts(cx: &ExtCtxt, tts: TokenStream) -> Vec<ast::Stmt> {
    let mut ss = Vec::new();
    let mut quoted = false;
    for tt in tts.into_trees() {
        quoted = match tt {
            TokenTree::Token(_, token::Dollar) if !quoted => true,
            _ => {
                ss.extend(statements_mk_tt(cx, &tt, quoted));
                false
            }
        }
    }
    ss
}

fn expand_tts(cx: &ExtCtxt, sp: Span, tts: &[TokenTree]) -> (P<ast::Expr>, P<ast::Expr>) {
    let (cx_expr, tts) = parse_arguments_to_quote(cx, tts);

    let mut vector = mk_stmts_let(cx, sp);
    vector.extend(statements_mk_tts(cx, tts.iter().cloned().collect()));
    vector.push(cx.stmt_expr(cx.expr_ident(sp, id_ext("tt"))));
    let block = cx.expr_block(cx.block(sp, vector));
    let unflatten = vec![id_ext("syntax"), id_ext("ext"), id_ext("quote"), id_ext("unflatten")];

    (cx_expr, cx.expr_call_global(sp, unflatten, vec![block]))
}

fn expand_wrapper(cx: &ExtCtxt,
                  sp: Span,
                  cx_expr: P<ast::Expr>,
                  expr: P<ast::Expr>,
                  imports: &[&[&str]]) -> P<ast::Expr> {
    // Explicitly borrow to avoid moving from the invoker (#16992)
    let cx_expr_borrow = cx.expr_addr_of(sp, cx.expr_deref(sp, cx_expr));
    let stmt_let_ext_cx = cx.stmt_let(sp, false, id_ext("ext_cx"), cx_expr_borrow);

    let mut stmts = imports.iter().map(|path| {
        // make item: `use ...;`
        let path = path.iter().map(|s| s.to_string()).collect();
        let use_item = cx.item_use_glob(
            sp,
            respan(sp.shrink_to_lo(), ast::VisibilityKind::Inherited),
            ids_ext(path),
        );
        cx.stmt_item(sp, use_item)
    }).chain(Some(stmt_let_ext_cx)).collect::<Vec<_>>();
    stmts.push(cx.stmt_expr(expr));

    cx.expr_block(cx.block(sp, stmts))
}

fn expand_parse_call(cx: &ExtCtxt,
                     sp: Span,
                     parse_method: &str,
                     arg_exprs: Vec<P<ast::Expr>> ,
                     tts: &[TokenTree]) -> P<ast::Expr> {
    let (cx_expr, tts_expr) = expand_tts(cx, sp, tts);

    let parse_sess_call = || cx.expr_method_call(
        sp, cx.expr_ident(sp, id_ext("ext_cx")),
        id_ext("parse_sess"), Vec::new());

    let new_parser_call =
        cx.expr_call(sp,
                     cx.expr_ident(sp, id_ext("new_parser_from_tts")),
                     vec![parse_sess_call(), tts_expr]);

    let path = vec![id_ext("syntax"), id_ext("ext"), id_ext("quote"), id_ext(parse_method)];
    let mut args = vec![cx.expr_mut_addr_of(sp, new_parser_call)];
    args.extend(arg_exprs);
    let expr = cx.expr_call_global(sp, path, args);

    if parse_method == "parse_attribute" {
        expand_wrapper(cx, sp, cx_expr, expr, &[&["syntax", "ext", "quote", "rt"],
                                                &["syntax", "parse", "attr"]])
    } else {
        expand_wrapper(cx, sp, cx_expr, expr, &[&["syntax", "ext", "quote", "rt"]])
    }
}

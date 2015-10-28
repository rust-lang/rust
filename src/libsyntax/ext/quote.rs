// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::Span;
use ext::base::ExtCtxt;
use ext::base;
use ext::build::AstBuilder;
use parse::token::*;
use parse::token;
use ptr::P;

///  Quasiquoting works via token trees.
///
///  This is registered as a set of expression syntax extension called quote!
///  that lifts its argument token-tree to an AST representing the
///  construction of the same token tree, with token::SubstNt interpreted
///  as antiquotes (splices).

pub mod rt {
    use ast;
    use codemap::Spanned;
    use ext::base::ExtCtxt;
    use parse::{self, token, classify};
    use ptr::P;
    use std::rc::Rc;

    use ast::{TokenTree, Expr};

    pub use parse::new_parser_from_tts;
    pub use codemap::{BytePos, Span, dummy_spanned, DUMMY_SP};

    pub trait ToTokens {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree>;
    }

    impl ToTokens for TokenTree {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec!(self.clone())
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
            match self {
                &Some(ref t) => t.to_tokens(cx),
                &None => Vec::new(),
            }
        }
    }

    impl ToTokens for ast::Ident {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(DUMMY_SP, token::Ident(*self, token::Plain))]
        }
    }

    impl ToTokens for ast::Path {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(DUMMY_SP, token::Interpolated(token::NtPath(Box::new(self.clone()))))]
        }
    }

    impl ToTokens for ast::Ty {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(self.span, token::Interpolated(token::NtTy(P(self.clone()))))]
        }
    }

    impl ToTokens for ast::Block {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(self.span, token::Interpolated(token::NtBlock(P(self.clone()))))]
        }
    }

    impl ToTokens for ast::Generics {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(DUMMY_SP, token::Interpolated(token::NtGenerics(self.clone())))]
        }
    }

    impl ToTokens for ast::WhereClause {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(DUMMY_SP, token::Interpolated(token::NtWhereClause(self.clone())))]
        }
    }

    impl ToTokens for P<ast::Item> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(self.span, token::Interpolated(token::NtItem(self.clone())))]
        }
    }

    impl ToTokens for P<ast::ImplItem> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(self.span, token::Interpolated(token::NtImplItem(self.clone())))]
        }
    }

    impl ToTokens for P<ast::TraitItem> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(self.span, token::Interpolated(token::NtTraitItem(self.clone())))]
        }
    }

    impl ToTokens for P<ast::Stmt> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            let mut tts = vec![
                ast::TtToken(self.span, token::Interpolated(token::NtStmt(self.clone())))
            ];

            // Some statements require a trailing semicolon.
            if classify::stmt_ends_with_semi(&self.node) {
                tts.push(ast::TtToken(self.span, token::Semi));
            }

            tts
        }
    }

    impl ToTokens for P<ast::Expr> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(self.span, token::Interpolated(token::NtExpr(self.clone())))]
        }
    }

    impl ToTokens for P<ast::Pat> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(self.span, token::Interpolated(token::NtPat(self.clone())))]
        }
    }

    impl ToTokens for ast::Arm {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(DUMMY_SP, token::Interpolated(token::NtArm(self.clone())))]
        }
    }

    macro_rules! impl_to_tokens_slice {
        ($t: ty, $sep: expr) => {
            impl ToTokens for [$t] {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    let mut v = vec![];
                    for (i, x) in self.iter().enumerate() {
                        if i > 0 {
                            v.push_all(&$sep);
                        }
                        v.extend(x.to_tokens(cx));
                    }
                    v
                }
            }
        };
    }

    impl_to_tokens_slice! { ast::Ty, [ast::TtToken(DUMMY_SP, token::Comma)] }
    impl_to_tokens_slice! { P<ast::Item>, [] }

    impl ToTokens for P<ast::MetaItem> {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtToken(DUMMY_SP, token::Interpolated(token::NtMeta(self.clone())))]
        }
    }

    impl ToTokens for ast::Attribute {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            let mut r = vec![];
            // FIXME: The spans could be better
            r.push(ast::TtToken(self.span, token::Pound));
            if self.node.style == ast::AttrStyle::Inner {
                r.push(ast::TtToken(self.span, token::Not));
            }
            r.push(ast::TtDelimited(self.span, Rc::new(ast::Delimited {
                delim: token::Bracket,
                open_span: self.span,
                tts: self.node.value.to_tokens(cx),
                close_span: self.span,
            })));
            r
        }
    }

    impl ToTokens for str {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            let lit = ast::LitStr(
                token::intern_and_get_ident(self), ast::CookedStr);
            dummy_spanned(lit).to_tokens(cx)
        }
    }

    impl ToTokens for () {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec![ast::TtDelimited(DUMMY_SP, Rc::new(ast::Delimited {
                delim: token::Paren,
                open_span: DUMMY_SP,
                tts: vec![],
                close_span: DUMMY_SP,
            }))]
        }
    }

    impl ToTokens for ast::Lit {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            // FIXME: This is wrong
            P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: ast::ExprLit(P(self.clone())),
                span: DUMMY_SP,
            }).to_tokens(cx)
        }
    }

    impl ToTokens for bool {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            dummy_spanned(ast::LitBool(*self)).to_tokens(cx)
        }
    }

    impl ToTokens for char {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            dummy_spanned(ast::LitChar(*self)).to_tokens(cx)
        }
    }

    macro_rules! impl_to_tokens_int {
        (signed, $t:ty, $tag:expr) => (
            impl ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    let lit = ast::LitInt(*self as u64, ast::SignedIntLit($tag,
                                                                          ast::Sign::new(*self)));
                    dummy_spanned(lit).to_tokens(cx)
                }
            }
        );
        (unsigned, $t:ty, $tag:expr) => (
            impl ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    let lit = ast::LitInt(*self as u64, ast::UnsignedIntLit($tag));
                    dummy_spanned(lit).to_tokens(cx)
                }
            }
        );
    }

    impl_to_tokens_int! { signed, isize, ast::TyIs }
    impl_to_tokens_int! { signed, i8,  ast::TyI8 }
    impl_to_tokens_int! { signed, i16, ast::TyI16 }
    impl_to_tokens_int! { signed, i32, ast::TyI32 }
    impl_to_tokens_int! { signed, i64, ast::TyI64 }

    impl_to_tokens_int! { unsigned, usize, ast::TyUs }
    impl_to_tokens_int! { unsigned, u8,   ast::TyU8 }
    impl_to_tokens_int! { unsigned, u16,  ast::TyU16 }
    impl_to_tokens_int! { unsigned, u32,  ast::TyU32 }
    impl_to_tokens_int! { unsigned, u64,  ast::TyU64 }

    pub trait ExtParseUtils {
        fn parse_item(&self, s: String) -> P<ast::Item>;
        fn parse_expr(&self, s: String) -> P<ast::Expr>;
        fn parse_stmt(&self, s: String) -> P<ast::Stmt>;
        fn parse_tts(&self, s: String) -> Vec<ast::TokenTree>;
    }

    impl<'a> ExtParseUtils for ExtCtxt<'a> {

        fn parse_item(&self, s: String) -> P<ast::Item> {
            parse::parse_item_from_source_str(
                "<quote expansion>".to_string(),
                s,
                self.cfg(),
                self.parse_sess()).expect("parse error")
        }

        fn parse_stmt(&self, s: String) -> P<ast::Stmt> {
            parse::parse_stmt_from_source_str("<quote expansion>".to_string(),
                                              s,
                                              self.cfg(),
                                              self.parse_sess()).expect("parse error")
        }

        fn parse_expr(&self, s: String) -> P<ast::Expr> {
            parse::parse_expr_from_source_str("<quote expansion>".to_string(),
                                              s,
                                              self.cfg(),
                                              self.parse_sess())
        }

        fn parse_tts(&self, s: String) -> Vec<ast::TokenTree> {
            parse::parse_tts_from_source_str("<quote expansion>".to_string(),
                                             s,
                                             self.cfg(),
                                             self.parse_sess())
        }
    }
}

pub fn expand_quote_tokens<'cx>(cx: &'cx mut ExtCtxt,
                                sp: Span,
                                tts: &[ast::TokenTree])
                                -> Box<base::MacResult+'cx> {
    let (cx_expr, expr) = expand_tts(cx, sp, tts);
    let expanded = expand_wrapper(cx, sp, cx_expr, expr, &[&["syntax", "ext", "quote", "rt"]]);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_expr<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_expr_panic", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_item<'cx>(cx: &mut ExtCtxt,
                              sp: Span,
                              tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_item_panic", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_pat<'cx>(cx: &'cx mut ExtCtxt,
                             sp: Span,
                             tts: &[ast::TokenTree])
                             -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_pat_panic", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_arm(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[ast::TokenTree])
                        -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_arm_panic", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_ty(cx: &mut ExtCtxt,
                       sp: Span,
                       tts: &[ast::TokenTree])
                       -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_ty_panic", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_stmt(cx: &mut ExtCtxt,
                         sp: Span,
                         tts: &[ast::TokenTree])
                         -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_stmt_panic", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_attr(cx: &mut ExtCtxt,
                         sp: Span,
                         tts: &[ast::TokenTree])
                         -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_attribute_panic",
                                    vec!(cx.expr_bool(sp, true)), tts);

    base::MacEager::expr(expanded)
}

pub fn expand_quote_matcher(cx: &mut ExtCtxt,
                            sp: Span,
                            tts: &[ast::TokenTree])
                            -> Box<base::MacResult+'static> {
    let (cx_expr, tts) = parse_arguments_to_quote(cx, tts);
    let mut vector = mk_stmts_let(cx, sp);
    vector.extend(statements_mk_tts(cx, &tts[..], true));
    let block = cx.expr_block(
        cx.block_all(sp,
                     vector,
                     Some(cx.expr_ident(sp, id_ext("tt")))));

    let expanded = expand_wrapper(cx, sp, cx_expr, block, &[&["syntax", "ext", "quote", "rt"]]);
    base::MacEager::expr(expanded)
}

fn ids_ext(strs: Vec<String> ) -> Vec<ast::Ident> {
    strs.iter().map(|str| str_to_ident(&(*str))).collect()
}

fn id_ext(str: &str) -> ast::Ident {
    str_to_ident(str)
}

// Lift an ident to the expr that evaluates to that ident.
fn mk_ident(cx: &ExtCtxt, sp: Span, ident: ast::Ident) -> P<ast::Expr> {
    let e_str = cx.expr_str(sp, ident.name.as_str());
    cx.expr_method_call(sp,
                        cx.expr_ident(sp, id_ext("ext_cx")),
                        id_ext("ident_of"),
                        vec!(e_str))
}

// Lift a name to the expr that evaluates to that name
fn mk_name(cx: &ExtCtxt, sp: Span, ident: ast::Ident) -> P<ast::Expr> {
    let e_str = cx.expr_str(sp, ident.name.as_str());
    cx.expr_method_call(sp,
                        cx.expr_ident(sp, id_ext("ext_cx")),
                        id_ext("name_of"),
                        vec!(e_str))
}

fn mk_ast_path(cx: &ExtCtxt, sp: Span, name: &str) -> P<ast::Expr> {
    let idents = vec!(id_ext("syntax"), id_ext("ast"), id_ext(name));
    cx.expr_path(cx.path_global(sp, idents))
}

fn mk_token_path(cx: &ExtCtxt, sp: Span, name: &str) -> P<ast::Expr> {
    let idents = vec!(id_ext("syntax"), id_ext("parse"), id_ext("token"), id_ext(name));
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
        token::Paren     => "Paren",
        token::Bracket   => "Bracket",
        token::Brace     => "Brace",
    };
    mk_token_path(cx, sp, name)
}

#[allow(non_upper_case_globals)]
fn expr_mk_token(cx: &ExtCtxt, sp: Span, tok: &token::Token) -> P<ast::Expr> {
    macro_rules! mk_lit {
        ($name: expr, $suffix: expr, $($args: expr),*) => {{
            let inner = cx.expr_call(sp, mk_token_path(cx, sp, $name), vec![$($args),*]);
            let suffix = match $suffix {
                Some(name) => cx.expr_some(sp, mk_name(cx, sp, ast::Ident::with_empty_ctxt(name))),
                None => cx.expr_none(sp)
            };
            cx.expr_call(sp, mk_token_path(cx, sp, "Literal"), vec![inner, suffix])
        }}
    }
    match *tok {
        token::BinOp(binop) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "BinOp"), vec!(mk_binop(cx, sp, binop)));
        }
        token::BinOpEq(binop) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "BinOpEq"),
                                vec!(mk_binop(cx, sp, binop)));
        }

        token::OpenDelim(delim) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "OpenDelim"),
                                vec![mk_delim(cx, sp, delim)]);
        }
        token::CloseDelim(delim) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "CloseDelim"),
                                vec![mk_delim(cx, sp, delim)]);
        }

        token::Literal(token::Byte(i), suf) => {
            let e_byte = mk_name(cx, sp, ast::Ident::with_empty_ctxt(i));
            return mk_lit!("Byte", suf, e_byte);
        }

        token::Literal(token::Char(i), suf) => {
            let e_char = mk_name(cx, sp, ast::Ident::with_empty_ctxt(i));
            return mk_lit!("Char", suf, e_char);
        }

        token::Literal(token::Integer(i), suf) => {
            let e_int = mk_name(cx, sp, ast::Ident::with_empty_ctxt(i));
            return mk_lit!("Integer", suf, e_int);
        }

        token::Literal(token::Float(fident), suf) => {
            let e_fident = mk_name(cx, sp, ast::Ident::with_empty_ctxt(fident));
            return mk_lit!("Float", suf, e_fident);
        }

        token::Literal(token::Str_(ident), suf) => {
            return mk_lit!("Str_", suf, mk_name(cx, sp, ast::Ident::with_empty_ctxt(ident)))
        }

        token::Literal(token::StrRaw(ident, n), suf) => {
            return mk_lit!("StrRaw", suf, mk_name(cx, sp, ast::Ident::with_empty_ctxt(ident)),
                           cx.expr_usize(sp, n))
        }

        token::Ident(ident, style) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "Ident"),
                                vec![mk_ident(cx, sp, ident),
                                     match style {
                                        ModName => mk_token_path(cx, sp, "ModName"),
                                        Plain   => mk_token_path(cx, sp, "Plain"),
                                     }]);
        }

        token::Lifetime(ident) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "Lifetime"),
                                vec!(mk_ident(cx, sp, ident)));
        }

        token::DocComment(ident) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "DocComment"),
                                vec!(mk_name(cx, sp, ast::Ident::with_empty_ctxt(ident))));
        }

        token::MatchNt(name, kind, namep, kindp) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "MatchNt"),
                                vec!(mk_ident(cx, sp, name),
                                     mk_ident(cx, sp, kind),
                                     match namep {
                                        ModName => mk_token_path(cx, sp, "ModName"),
                                        Plain   => mk_token_path(cx, sp, "Plain"),
                                     },
                                     match kindp {
                                        ModName => mk_token_path(cx, sp, "ModName"),
                                        Plain   => mk_token_path(cx, sp, "Plain"),
                                     }));
        }

        token::Interpolated(_) => panic!("quote! with interpolated token"),

        _ => ()
    }

    let name = match *tok {
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
        token::Underscore   => "Underscore",
        token::Eof          => "Eof",
        _                   => panic!("unhandled token in quote!"),
    };
    mk_token_path(cx, sp, name)
}

fn statements_mk_tt(cx: &ExtCtxt, tt: &ast::TokenTree, matcher: bool) -> Vec<P<ast::Stmt>> {
    match *tt {
        ast::TtToken(sp, SubstNt(ident, _)) => {
            // tt.extend($ident.to_tokens(ext_cx))

            let e_to_toks =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, ident),
                                    id_ext("to_tokens"),
                                    vec!(cx.expr_ident(sp, id_ext("ext_cx"))));
            let e_to_toks =
                cx.expr_method_call(sp, e_to_toks, id_ext("into_iter"), vec![]);

            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("extend"),
                                    vec!(e_to_toks));

            vec!(cx.stmt_expr(e_push))
        }
        ref tt @ ast::TtToken(_, MatchNt(..)) if !matcher => {
            let mut seq = vec![];
            for i in 0..tt.len() {
                seq.push(tt.get_tt(i));
            }
            statements_mk_tts(cx, &seq[..], matcher)
        }
        ast::TtToken(sp, ref tok) => {
            let e_sp = cx.expr_ident(sp, id_ext("_sp"));
            let e_tok = cx.expr_call(sp,
                                     mk_ast_path(cx, sp, "TtToken"),
                                     vec!(e_sp, expr_mk_token(cx, sp, tok)));
            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("push"),
                                    vec!(e_tok));
            vec!(cx.stmt_expr(e_push))
        },
        ast::TtDelimited(_, ref delimed) => {
            statements_mk_tt(cx, &delimed.open_tt(), matcher).into_iter()
                .chain(delimed.tts.iter()
                                  .flat_map(|tt| statements_mk_tt(cx, tt, matcher)))
                .chain(statements_mk_tt(cx, &delimed.close_tt(), matcher))
                .collect()
        },
        ast::TtSequence(sp, ref seq) => {
            if !matcher {
                panic!("TtSequence in quote!");
            }

            let e_sp = cx.expr_ident(sp, id_ext("_sp"));

            let stmt_let_tt = cx.stmt_let(sp, true, id_ext("tt"), cx.expr_vec_ng(sp));
            let mut tts_stmts = vec![stmt_let_tt];
            tts_stmts.extend(statements_mk_tts(cx, &seq.tts[..], matcher));
            let e_tts = cx.expr_block(cx.block(sp, tts_stmts,
                                                   Some(cx.expr_ident(sp, id_ext("tt")))));
            let e_separator = match seq.separator {
                Some(ref sep) => cx.expr_some(sp, expr_mk_token(cx, sp, sep)),
                None => cx.expr_none(sp),
            };
            let e_op = match seq.op {
                ast::ZeroOrMore => mk_ast_path(cx, sp, "ZeroOrMore"),
                ast::OneOrMore => mk_ast_path(cx, sp, "OneOrMore"),
            };
            let fields = vec![cx.field_imm(sp, id_ext("tts"), e_tts),
                              cx.field_imm(sp, id_ext("separator"), e_separator),
                              cx.field_imm(sp, id_ext("op"), e_op),
                              cx.field_imm(sp, id_ext("num_captures"),
                                               cx.expr_usize(sp, seq.num_captures))];
            let seq_path = vec![id_ext("syntax"), id_ext("ast"), id_ext("SequenceRepetition")];
            let e_seq_struct = cx.expr_struct(sp, cx.path_global(sp, seq_path), fields);
            let e_rc_new = cx.expr_call_global(sp, vec![id_ext("std"),
                                                        id_ext("rc"),
                                                        id_ext("Rc"),
                                                        id_ext("new")],
                                                   vec![e_seq_struct]);
            let e_tok = cx.expr_call(sp,
                                     mk_ast_path(cx, sp, "TtSequence"),
                                     vec!(e_sp, e_rc_new));
            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("push"),
                                    vec!(e_tok));
            vec!(cx.stmt_expr(e_push))
        }
    }
}

fn parse_arguments_to_quote(cx: &ExtCtxt, tts: &[ast::TokenTree])
                            -> (P<ast::Expr>, Vec<ast::TokenTree>) {
    // NB: It appears that the main parser loses its mind if we consider
    // $foo as a SubstNt during the main parse, so we have to re-parse
    // under quote_depth > 0. This is silly and should go away; the _guess_ is
    // it has to do with transition away from supporting old-style macros, so
    // try removing it when enough of them are gone.

    let mut p = cx.new_parser_from_tts(tts);
    p.quote_depth += 1;

    let cx_expr = panictry!(p.parse_expr_nopanic());
    if !panictry!(p.eat(&token::Comma)) {
        panic!(p.fatal("expected token `,`"));
    }

    let tts = panictry!(p.parse_all_token_trees());
    p.abort_if_errors();

    (cx_expr, tts)
}

fn mk_stmts_let(cx: &ExtCtxt, sp: Span) -> Vec<P<ast::Stmt>> {
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
    // (pushing fake FileMaps onto the parser to account for original sites
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

    vec!(stmt_let_sp, stmt_let_tt)
}

fn statements_mk_tts(cx: &ExtCtxt, tts: &[ast::TokenTree], matcher: bool) -> Vec<P<ast::Stmt>> {
    let mut ss = Vec::new();
    for tt in tts {
        ss.extend(statements_mk_tt(cx, tt, matcher));
    }
    ss
}

fn expand_tts(cx: &ExtCtxt, sp: Span, tts: &[ast::TokenTree])
              -> (P<ast::Expr>, P<ast::Expr>) {
    let (cx_expr, tts) = parse_arguments_to_quote(cx, tts);

    let mut vector = mk_stmts_let(cx, sp);
    vector.extend(statements_mk_tts(cx, &tts[..], false));
    let block = cx.expr_block(
        cx.block_all(sp,
                     vector,
                     Some(cx.expr_ident(sp, id_ext("tt")))));

    (cx_expr, block)
}

fn expand_wrapper(cx: &ExtCtxt,
                  sp: Span,
                  cx_expr: P<ast::Expr>,
                  expr: P<ast::Expr>,
                  imports: &[&[&str]]) -> P<ast::Expr> {
    // Explicitly borrow to avoid moving from the invoker (#16992)
    let cx_expr_borrow = cx.expr_addr_of(sp, cx.expr_deref(sp, cx_expr));
    let stmt_let_ext_cx = cx.stmt_let(sp, false, id_ext("ext_cx"), cx_expr_borrow);

    let stmts = imports.iter().map(|path| {
        // make item: `use ...;`
        let path = path.iter().map(|s| s.to_string()).collect();
        cx.stmt_item(sp, cx.item_use_glob(sp, ast::Inherited, ids_ext(path)))
    }).chain(Some(stmt_let_ext_cx)).collect();

    cx.expr_block(cx.block_all(sp, stmts, Some(expr)))
}

fn expand_parse_call(cx: &ExtCtxt,
                     sp: Span,
                     parse_method: &str,
                     arg_exprs: Vec<P<ast::Expr>> ,
                     tts: &[ast::TokenTree]) -> P<ast::Expr> {
    let (cx_expr, tts_expr) = expand_tts(cx, sp, tts);

    let cfg_call = || cx.expr_method_call(
        sp, cx.expr_ident(sp, id_ext("ext_cx")),
        id_ext("cfg"), Vec::new());

    let parse_sess_call = || cx.expr_method_call(
        sp, cx.expr_ident(sp, id_ext("ext_cx")),
        id_ext("parse_sess"), Vec::new());

    let new_parser_call =
        cx.expr_call(sp,
                     cx.expr_ident(sp, id_ext("new_parser_from_tts")),
                     vec!(parse_sess_call(), cfg_call(), tts_expr));

    let expr = cx.expr_method_call(sp, new_parser_call, id_ext(parse_method),
                                   arg_exprs);

    if parse_method == "parse_attribute" {
        expand_wrapper(cx, sp, cx_expr, expr, &[&["syntax", "ext", "quote", "rt"],
                                                &["syntax", "parse", "attr"]])
    } else {
        expand_wrapper(cx, sp, cx_expr, expr, &[&["syntax", "ext", "quote", "rt"]])
    }
}

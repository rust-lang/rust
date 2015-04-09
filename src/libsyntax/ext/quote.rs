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
    use parse::token;
    use parse;
    use print::pprust;
    use ptr::P;

    use ast::{TokenTree, Generics, Expr};

    use std::iter;

    pub use parse::new_parser_from_tts;
    pub use codemap::{BytePos, Span, dummy_spanned};
    pub use std::iter::IntoIterator;
    pub use parse::attr::ParserAttr;

    pub trait ToTokens {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> ;
    }

    impl ToTokens for TokenTree {
        fn to_tokens(&self, _cx: &ExtCtxt) -> Vec<TokenTree> {
            vec!(self.clone())
        }
    }

    impl<T: ToTokens> ToTokens for Vec<T> {
        fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
            self.iter().flat_map(|t| t.to_tokens(cx).into_iter()).collect()
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

    /* Should be (when bugs in default methods are fixed):

    trait ToSource : ToTokens {
        // Takes a thing and generates a string containing rust code for it.
        pub fn to_source() -> String;

        // If you can make source, you can definitely make tokens.
        pub fn to_tokens(cx: &ExtCtxt) -> ~[TokenTree] {
            cx.parse_tts(self.to_source())
        }
    }

    */

    // FIXME: Move this trait to pprust and get rid of *_to_str?
    pub trait ToSource {
        // Takes a thing and generates a string containing rust code for it.
        fn to_source(&self) -> String;
    }

    // FIXME (Issue #16472): This should go away after ToToken impls
    // are revised to go directly to token-trees.
    trait ToSourceWithHygiene : ToSource {
        // Takes a thing and generates a string containing rust code
        // for it, encoding Idents as special byte sequences to
        // maintain hygiene across serialization and deserialization.
        fn to_source_with_hygiene(&self) -> String;
    }

    macro_rules! impl_to_source {
        (P<$t:ty>, $pp:ident) => (
            impl ToSource for P<$t> {
                fn to_source(&self) -> String {
                    pprust::$pp(&**self)
                }
            }
            impl ToSourceWithHygiene for P<$t> {
                fn to_source_with_hygiene(&self) -> String {
                    pprust::with_hygiene::$pp(&**self)
                }
            }
        );
        ($t:ty, $pp:ident) => (
            impl ToSource for $t {
                fn to_source(&self) -> String {
                    pprust::$pp(self)
                }
            }
            impl ToSourceWithHygiene for $t {
                fn to_source_with_hygiene(&self) -> String {
                    pprust::with_hygiene::$pp(self)
                }
            }
        );
    }

    fn slice_to_source<'a, T: ToSource>(sep: &'static str, xs: &'a [T]) -> String {
        xs.iter()
            .map(|i| i.to_source())
            .collect::<Vec<String>>()
            .connect(sep)
            .to_string()
    }

    fn slice_to_source_with_hygiene<'a, T: ToSourceWithHygiene>(
        sep: &'static str, xs: &'a [T]) -> String {
        xs.iter()
            .map(|i| i.to_source_with_hygiene())
            .collect::<Vec<String>>()
            .connect(sep)
            .to_string()
    }

    macro_rules! impl_to_source_slice {
        ($t:ty, $sep:expr) => (
            impl ToSource for [$t] {
                fn to_source(&self) -> String {
                    slice_to_source($sep, self)
                }
            }

            impl ToSourceWithHygiene for [$t] {
                fn to_source_with_hygiene(&self) -> String {
                    slice_to_source_with_hygiene($sep, self)
                }
            }
        )
    }

    impl ToSource for ast::Ident {
        fn to_source(&self) -> String {
            token::get_ident(*self).to_string()
        }
    }

    impl ToSourceWithHygiene for ast::Ident {
        fn to_source_with_hygiene(&self) -> String {
            self.encode_with_hygiene()
        }
    }

    impl_to_source! { ast::Path, path_to_string }
    impl_to_source! { ast::Ty, ty_to_string }
    impl_to_source! { ast::Block, block_to_string }
    impl_to_source! { ast::Arg, arg_to_string }
    impl_to_source! { Generics, generics_to_string }
    impl_to_source! { ast::WhereClause, where_clause_to_string }
    impl_to_source! { P<ast::Item>, item_to_string }
    impl_to_source! { P<ast::ImplItem>, impl_item_to_string }
    impl_to_source! { P<ast::TraitItem>, trait_item_to_string }
    impl_to_source! { P<ast::Stmt>, stmt_to_string }
    impl_to_source! { P<ast::Expr>, expr_to_string }
    impl_to_source! { P<ast::Pat>, pat_to_string }
    impl_to_source! { ast::Arm, arm_to_string }
    impl_to_source_slice! { ast::Ty, ", " }
    impl_to_source_slice! { P<ast::Item>, "\n\n" }

    impl ToSource for ast::Attribute_ {
        fn to_source(&self) -> String {
            pprust::attribute_to_string(&dummy_spanned(self.clone()))
        }
    }
    impl ToSourceWithHygiene for ast::Attribute_ {
        fn to_source_with_hygiene(&self) -> String {
            self.to_source()
        }
    }

    impl ToSource for str {
        fn to_source(&self) -> String {
            let lit = dummy_spanned(ast::LitStr(
                    token::intern_and_get_ident(self), ast::CookedStr));
            pprust::lit_to_string(&lit)
        }
    }
    impl ToSourceWithHygiene for str {
        fn to_source_with_hygiene(&self) -> String {
            self.to_source()
        }
    }

    impl ToSource for () {
        fn to_source(&self) -> String {
            "()".to_string()
        }
    }
    impl ToSourceWithHygiene for () {
        fn to_source_with_hygiene(&self) -> String {
            self.to_source()
        }
    }

    impl ToSource for bool {
        fn to_source(&self) -> String {
            let lit = dummy_spanned(ast::LitBool(*self));
            pprust::lit_to_string(&lit)
        }
    }
    impl ToSourceWithHygiene for bool {
        fn to_source_with_hygiene(&self) -> String {
            self.to_source()
        }
    }

    impl ToSource for char {
        fn to_source(&self) -> String {
            let lit = dummy_spanned(ast::LitChar(*self));
            pprust::lit_to_string(&lit)
        }
    }
    impl ToSourceWithHygiene for char {
        fn to_source_with_hygiene(&self) -> String {
            self.to_source()
        }
    }

    macro_rules! impl_to_source_int {
        (signed, $t:ty, $tag:expr) => (
            impl ToSource for $t {
                fn to_source(&self) -> String {
                    let lit = ast::LitInt(*self as u64, ast::SignedIntLit($tag,
                                                                          ast::Sign::new(*self)));
                    pprust::lit_to_string(&dummy_spanned(lit))
                }
            }
            impl ToSourceWithHygiene for $t {
                fn to_source_with_hygiene(&self) -> String {
                    self.to_source()
                }
            }
        );
        (unsigned, $t:ty, $tag:expr) => (
            impl ToSource for $t {
                fn to_source(&self) -> String {
                    let lit = ast::LitInt(*self as u64, ast::UnsignedIntLit($tag));
                    pprust::lit_to_string(&dummy_spanned(lit))
                }
            }
            impl ToSourceWithHygiene for $t {
                fn to_source_with_hygiene(&self) -> String {
                    self.to_source()
                }
            }
        );
    }

    impl_to_source_int! { signed, isize, ast::TyIs }
    impl_to_source_int! { signed, i8,  ast::TyI8 }
    impl_to_source_int! { signed, i16, ast::TyI16 }
    impl_to_source_int! { signed, i32, ast::TyI32 }
    impl_to_source_int! { signed, i64, ast::TyI64 }

    impl_to_source_int! { unsigned, usize, ast::TyUs }
    impl_to_source_int! { unsigned, u8,   ast::TyU8 }
    impl_to_source_int! { unsigned, u16,  ast::TyU16 }
    impl_to_source_int! { unsigned, u32,  ast::TyU32 }
    impl_to_source_int! { unsigned, u64,  ast::TyU64 }

    // Alas ... we write these out instead. All redundant.

    macro_rules! impl_to_tokens {
        ($t:ty) => (
            impl ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    cx.parse_tts_with_hygiene(self.to_source_with_hygiene())
                }
            }
        )
    }

    macro_rules! impl_to_tokens_lifetime {
        ($t:ty) => (
            impl<'a> ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    cx.parse_tts_with_hygiene(self.to_source_with_hygiene())
                }
            }
        )
    }

    impl_to_tokens_lifetime! { &'a str }
    impl_to_tokens! { () }
    impl_to_tokens! { char }
    impl_to_tokens! { bool }
    impl_to_tokens! { isize }
    impl_to_tokens! { i8 }
    impl_to_tokens! { i16 }
    impl_to_tokens! { i32 }
    impl_to_tokens! { i64 }
    impl_to_tokens! { usize }
    impl_to_tokens! { u8 }
    impl_to_tokens! { u16 }
    impl_to_tokens! { u32 }
    impl_to_tokens! { u64 }

    pub trait ExtParseUtils {
        fn parse_item(&self, s: String) -> P<ast::Item>;
        fn parse_expr(&self, s: String) -> P<ast::Expr>;
        fn parse_stmt(&self, s: String) -> P<ast::Stmt>;
        fn parse_tts(&self, s: String) -> Vec<ast::TokenTree>;
    }

    trait ExtParseUtilsWithHygiene {
        // FIXME (Issue #16472): This should go away after ToToken impls
        // are revised to go directly to token-trees.
        fn parse_tts_with_hygiene(&self, s: String) -> Vec<ast::TokenTree>;
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

    impl<'a> ExtParseUtilsWithHygiene for ExtCtxt<'a> {

        fn parse_tts_with_hygiene(&self, s: String) -> Vec<ast::TokenTree> {
            use parse::with_hygiene::parse_tts_from_source_str;
            parse_tts_from_source_str("<quote expansion>".to_string(),
                                      s,
                                      self.cfg(),
                                      self.parse_sess())
        }

    }

    pub struct IterWrapper<I> {
        inner: I,
        empty: bool,
    }

    impl<T, I: Iterator<Item=T>> Iterator for IterWrapper<I> {
        type Item = T;
        fn next(&mut self) -> Option<T> {
            match self.inner.next() {
                Some(elem) => {
                    self.empty = false;
                    Some(elem)
                }
                None => {
                    assert!(!self.empty, "a fragment must repeat at least once in quasiquotation");
                    None
                }
            }
        }
    }

    pub trait IntoWrappedIter {
        type Item;
        type IntoIter: Iterator<Item=Self::Item>;
        fn into_wrapped_iter(self, one_or_more: bool) -> IterWrapper<Self::IntoIter>;
    }

    impl<I: IntoIterator> IntoWrappedIter for I {
        type Item = I::Item;
        type IntoIter = I::IntoIter;
        fn into_wrapped_iter(self, one_or_more: bool) -> IterWrapper<I::IntoIter> {
            IterWrapper { empty: one_or_more, inner: self.into_iter() }
        }
    }

    impl<T: Clone> IntoWrappedIter for Spanned<T> {
        type Item = Spanned<T>;
        type IntoIter = iter::Repeat<Spanned<T>>;
        fn into_wrapped_iter(self, one_or_more: bool) -> IterWrapper<iter::Repeat<Spanned<T>>> {
            IterWrapper { empty: one_or_more, inner: iter::repeat(self) }
        }
    }

    impl IntoWrappedIter for TokenTree {
        type Item = TokenTree;
        type IntoIter = iter::Repeat<TokenTree>;
        fn into_wrapped_iter(self, one_or_more: bool) -> IterWrapper<iter::Repeat<TokenTree>> {
            IterWrapper { empty: one_or_more, inner: iter::repeat(self) }
        }
    }

    macro_rules! impl_to_tokens_local {
        ($t:ty) => (
            impl_to_tokens!($t);

            impl IntoWrappedIter for $t {
                type Item = $t;
                type IntoIter = iter::Repeat<$t>;
                fn into_wrapped_iter(self, one_or_more: bool) -> IterWrapper<iter::Repeat<$t>> {
                    IterWrapper { empty: one_or_more, inner: iter::repeat(self) }
                }
            }
        )
    }

    impl_to_tokens_local! { ast::Ident }
    impl_to_tokens_local! { ast::Path }
    impl_to_tokens_local! { P<ast::Item> }
    impl_to_tokens_local! { P<ast::ImplItem> }
    impl_to_tokens_local! { P<ast::TraitItem> }
    impl_to_tokens_local! { P<ast::Pat> }
    impl_to_tokens_local! { ast::Arm }
    impl_to_tokens_local! { ast::Ty }
    impl_to_tokens_local! { Generics }
    impl_to_tokens_local! { ast::WhereClause }
    impl_to_tokens_local! { P<ast::Stmt> }
    impl_to_tokens_local! { P<ast::Expr> }
    impl_to_tokens_local! { ast::Block }
    impl_to_tokens_local! { ast::Arg }
    impl_to_tokens_local! { ast::Attribute_ }
}

pub fn expand_quote_tokens<'cx>(cx: &'cx mut ExtCtxt,
                                sp: Span,
                                tts: &[ast::TokenTree])
                                -> Box<base::MacResult+'cx> {
    let (cx_expr, expr) = expand_tts(cx, sp, tts);
    let expanded = expand_wrapper(cx, sp, cx_expr, expr);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_expr<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_expr", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_item<'cx>(cx: &mut ExtCtxt,
                              sp: Span,
                              tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_item", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_pat<'cx>(cx: &'cx mut ExtCtxt,
                             sp: Span,
                             tts: &[ast::TokenTree])
                             -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_pat", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_arm(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[ast::TokenTree])
                        -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_arm", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_ty(cx: &mut ExtCtxt,
                       sp: Span,
                       tts: &[ast::TokenTree])
                       -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_ty", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_stmt(cx: &mut ExtCtxt,
                         sp: Span,
                         tts: &[ast::TokenTree])
                         -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_stmt", vec!(), tts);
    base::MacEager::expr(expanded)
}

pub fn expand_quote_attr(cx: &mut ExtCtxt,
                         sp: Span,
                         tts: &[ast::TokenTree])
                         -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_attribute",
                                    vec!(cx.expr_bool(sp, true)), tts);

    base::MacEager::expr(expanded)
}

pub fn expand_quote_matcher(cx: &mut ExtCtxt,
                            sp: Span,
                            tts: &[ast::TokenTree])
                            -> Box<base::MacResult+'static> {
    let (cx_expr, tts) = parse_arguments_to_quote(cx, tts);
    let mut vector = mk_stmts_let(cx, sp);
    vector.extend(statements_mk_tts(cx, &tts[..], true).0.into_iter());
    let block = cx.expr_block(
        cx.block_all(sp,
                     vector,
                     Some(cx.expr_ident(sp, id_ext("tt")))));

    let expanded = expand_wrapper(cx, sp, cx_expr, block);
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
    let e_str = cx.expr_str(sp, token::get_ident(ident));
    cx.expr_method_call(sp,
                        cx.expr_ident(sp, id_ext("ext_cx")),
                        id_ext("ident_of"),
                        vec!(e_str))
}

// Lift a name to the expr that evaluates to that name
fn mk_name(cx: &ExtCtxt, sp: Span, ident: ast::Ident) -> P<ast::Expr> {
    let e_str = cx.expr_str(sp, token::get_ident(ident));
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
                Some(name) => cx.expr_some(sp, mk_name(cx, sp, ast::Ident::new(name))),
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
            let e_byte = mk_name(cx, sp, i.ident());
            return mk_lit!("Byte", suf, e_byte);
        }

        token::Literal(token::Char(i), suf) => {
            let e_char = mk_name(cx, sp, i.ident());
            return mk_lit!("Char", suf, e_char);
        }

        token::Literal(token::Integer(i), suf) => {
            let e_int = mk_name(cx, sp, i.ident());
            return mk_lit!("Integer", suf, e_int);
        }

        token::Literal(token::Float(fident), suf) => {
            let e_fident = mk_name(cx, sp, fident.ident());
            return mk_lit!("Float", suf, e_fident);
        }

        token::Literal(token::Str_(ident), suf) => {
            return mk_lit!("Str_", suf, mk_name(cx, sp, ident.ident()))
        }

        token::Literal(token::StrRaw(ident, n), suf) => {
            return mk_lit!("StrRaw", suf, mk_name(cx, sp, ident.ident()), cx.expr_usize(sp, n))
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
                                vec!(mk_name(cx, sp, ident.ident())));
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

fn statements_mk_tt(cx: &ExtCtxt, tt: &ast::TokenTree, matcher: bool) -> (Vec<P<ast::Stmt>>,
                                                                          Vec<ast::Ident>) {
    match *tt {
        ast::TtToken(sp, SubstNt(ident, _)) => {
            // tt.extend($ident.to_tokens(ext_cx).into_iter())

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

            (vec![cx.stmt_expr(e_push)], vec![ident])
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
            (vec![cx.stmt_expr(e_push)], vec![])
        },
        ast::TtDelimited(_, ref delimed) => {
            let (stmts, idents) = statements_mk_tts(cx, &delimed.tts[..], matcher);
            (statements_mk_tt(cx, &delimed.open_tt(), matcher).0.into_iter()
                .chain(stmts.into_iter())
                .chain(statements_mk_tt(cx, &delimed.close_tt(), matcher).0.into_iter())
                .collect(),
             idents)
        },
        ast::TtSequence(sp, ref seq) => {
            if matcher {
                let e_sp = cx.expr_ident(sp, id_ext("_sp"));

                let stmt_let_tt = cx.stmt_let(sp, true, id_ext("tt"), cx.expr_vec_ng(sp));
                let mut tts_stmts = vec![stmt_let_tt];
                tts_stmts.extend(statements_mk_tts(cx, &seq.tts[..], matcher).0.into_iter());
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
                (vec![cx.stmt_expr(e_push)], vec![])
            } else {
                // Repeating fragments in a loop:
                // for (...(a, b), ...) in a.into_wrapped_iter().zip(b.into_wrapped_iter())... {
                //     // (quasiquotation with $a, $b, ...)
                //}
                let (mut stmts, idents) = statements_mk_tts(cx, &seq.tts[..], matcher);
                if idents.is_empty() {
                    cx.span_fatal(sp, "attempted to repeat an expression containing \
                                       no syntax variables matched as repeating at this depth");
                }

                let mut iter = idents.clone().into_iter();
                let first = iter.next().unwrap();
                let mut zipped = cx.expr_ident(sp, first);
                let one_or_more = cx.expr_bool(sp, seq.op == ast::OneOrMore);
                zipped = cx.expr_method_call(sp, zipped, id_ext("into_wrapped_iter"),
                                                         vec![one_or_more.clone()]);
                let mut pat = cx.pat_ident(sp, first);
                for ident in iter {
                    // Assertion: zipped iterators must have at least one element
                    // if one_or_more == `true`.
                    let expr_ident = cx.expr_ident(sp, ident);
                    let expr = cx.expr_method_call(sp, expr_ident,
                                                       id_ext("into_wrapped_iter"),
                                                       vec![one_or_more.clone()]);
                    zipped = cx.expr_method_call(sp, zipped, id_ext("zip"), vec![expr]);
                    pat = cx.pat_tuple(sp, vec!(pat, cx.pat_ident(sp, ident)));
                }

                match seq.separator {
                    Some(ref tok) => {
                        // Intersperse the separator
                        stmts.extend(statements_mk_tt(cx, &ast::TtToken(sp, tok.clone()),
                                                          false).0.into_iter());
                    }
                    None => {}
                }

                let stmt_for = cx.stmt_expr(P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    node: ast::ExprForLoop(pat, zipped, cx.block(sp, stmts, None), None),
                    span: sp,
                }));

                let stmts_for = if seq.separator.is_some() {
                    // Pop the last occurence of the separator
                    let tt = cx.expr_ident(sp, id_ext("tt"));
                    let len = cx.expr_method_call(sp, tt.clone(), id_ext("len"), vec!());
                    let _len_ident = gensym_ident("_len");
                    let _len = cx.expr_ident(sp, _len_ident);
                    let cond = cx.expr_binary(sp, ast::BiNe, len.clone(), _len);
                    let then = cx.expr_method_call(sp, tt, id_ext("pop"), vec!());
                    let then = cx.expr_block(cx.block(sp, vec![cx.stmt_expr(then)], None));
                    let if_len_eq = cx.expr_if(sp, cond, then, None);
                    vec![cx.stmt_let(sp, false, _len_ident, len),
                         stmt_for,
                         cx.stmt_expr(if_len_eq)]
                } else {
                    vec![stmt_for]
                };

                (stmts_for, idents)
            }
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

    let cx_expr = p.parse_expr();
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

fn statements_mk_tts(cx: &ExtCtxt, tts: &[ast::TokenTree], matcher: bool) -> (Vec<P<ast::Stmt>>,
                                                                              Vec<ast::Ident>) {
    let mut stmts = Vec::new();
    let mut idents = Vec::new();
    for tt in tts {
        let (ss, is) = statements_mk_tt(cx, tt, matcher);
        stmts.extend(ss.into_iter());
        idents.extend(is.into_iter());
    }
    (stmts, idents)
}

fn expand_tts(cx: &ExtCtxt, sp: Span, tts: &[ast::TokenTree])
              -> (P<ast::Expr>, P<ast::Expr>) {
    let (cx_expr, tts) = parse_arguments_to_quote(cx, tts);

    let mut vector = mk_stmts_let(cx, sp);
    vector.extend(statements_mk_tts(cx, &tts[..], false).0.into_iter());
    let block = cx.expr_block(
        cx.block_all(sp,
                     vector,
                     Some(cx.expr_ident(sp, id_ext("tt")))));

    (cx_expr, block)
}

fn expand_wrapper(cx: &ExtCtxt,
                  sp: Span,
                  cx_expr: P<ast::Expr>,
                  expr: P<ast::Expr>) -> P<ast::Expr> {
    // Explicitly borrow to avoid moving from the invoker (#16992)
    let cx_expr_borrow = cx.expr_addr_of(sp, cx.expr_deref(sp, cx_expr));
    let stmt_let_ext_cx = cx.stmt_let(sp, false, id_ext("ext_cx"), cx_expr_borrow);

    let stmts = [
        &["syntax", "ext", "quote", "rt"]
    ].iter().map(|path| {
        // make item: `use ...;`
        let path = path.iter().map(|s| s.to_string()).collect();
        cx.stmt_item(sp, cx.item_use_glob(sp, ast::Inherited, ids_ext(path)))
    }).chain(Some(stmt_let_ext_cx).into_iter()).collect();

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

    expand_wrapper(cx, sp, cx_expr, expr)
}

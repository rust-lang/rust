// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use parse;

/**
*
* Quasiquoting works via token trees.
*
* This is registered as a set of expression syntax extension called quote!
* that lifts its argument token-tree to an AST representing the
* construction of the same token tree, with ast::tt_nonterminal nodes
* interpreted as antiquotes (splices).
*
*/

pub mod rt {
    use ast;
    use ext::base::ExtCtxt;
    use parse;
    use print::pprust;

    pub use ast::*;
    pub use parse::token::*;
    pub use parse::new_parser_from_tts;
    pub use codemap::{BytePos, Span, dummy_spanned};

    pub trait ToTokens {
        fn to_tokens(&self, _cx: @ExtCtxt) -> ~[token_tree];
    }

    impl ToTokens for ~[token_tree] {
        fn to_tokens(&self, _cx: @ExtCtxt) -> ~[token_tree] {
            (*self).clone()
        }
    }

    /* Should be (when bugs in default methods are fixed):

    trait ToSource : ToTokens {
        // Takes a thing and generates a string containing rust code for it.
        pub fn to_source() -> ~str;

        // If you can make source, you can definitely make tokens.
        pub fn to_tokens(cx: @ExtCtxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source())
        }
    }

    */

    pub trait ToSource {
        // Takes a thing and generates a string containing rust code for it.
        fn to_source(&self) -> @str;
    }

    impl ToSource for ast::Ident {
        fn to_source(&self) -> @str {
            ident_to_str(self)
        }
    }

    impl ToSource for @ast::item {
        fn to_source(&self) -> @str {
            pprust::item_to_str(*self, get_ident_interner()).to_managed()
        }
    }

    impl<'a> ToSource for &'a [@ast::item] {
        fn to_source(&self) -> @str {
            self.map(|i| i.to_source()).connect("\n\n").to_managed()
        }
    }

    impl ToSource for ast::Ty {
        fn to_source(&self) -> @str {
            pprust::ty_to_str(self, get_ident_interner()).to_managed()
        }
    }

    impl<'a> ToSource for &'a [ast::Ty] {
        fn to_source(&self) -> @str {
            self.map(|i| i.to_source()).connect(", ").to_managed()
        }
    }

    impl ToSource for Generics {
        fn to_source(&self) -> @str {
            pprust::generics_to_str(self, get_ident_interner()).to_managed()
        }
    }

    impl ToSource for @ast::Expr {
        fn to_source(&self) -> @str {
            pprust::expr_to_str(*self, get_ident_interner()).to_managed()
        }
    }

    impl ToSource for ast::Block {
        fn to_source(&self) -> @str {
            pprust::block_to_str(self, get_ident_interner()).to_managed()
        }
    }

    impl<'a> ToSource for &'a str {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_str(self.to_managed(), ast::CookedStr));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for int {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for i8 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i8));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for i16 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i16));
            pprust::lit_to_str(&lit).to_managed()
        }
    }


    impl ToSource for i32 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i32));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for i64 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i64));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for uint {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for u8 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u8));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for u16 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u16));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for u32 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u32));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    impl ToSource for u64 {
        fn to_source(&self) -> @str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u64));
            pprust::lit_to_str(&lit).to_managed()
        }
    }

    // Alas ... we write these out instead. All redundant.

    macro_rules! impl_to_tokens(
        ($t:ty) => (
            impl ToTokens for $t {
                fn to_tokens(&self, cx: @ExtCtxt) -> ~[token_tree] {
                    cx.parse_tts(self.to_source())
                }
            }
        )
    )

    macro_rules! impl_to_tokens_self(
        ($t:ty) => (
            impl<'a> ToTokens for $t {
                fn to_tokens(&self, cx: @ExtCtxt) -> ~[token_tree] {
                    cx.parse_tts(self.to_source())
                }
            }
        )
    )

    impl_to_tokens!(ast::Ident)
    impl_to_tokens!(@ast::item)
    impl_to_tokens_self!(&'a [@ast::item])
    impl_to_tokens!(ast::Ty)
    impl_to_tokens_self!(&'a [ast::Ty])
    impl_to_tokens!(Generics)
    impl_to_tokens!(@ast::Expr)
    impl_to_tokens!(ast::Block)
    impl_to_tokens_self!(&'a str)
    impl_to_tokens!(int)
    impl_to_tokens!(i8)
    impl_to_tokens!(i16)
    impl_to_tokens!(i32)
    impl_to_tokens!(i64)
    impl_to_tokens!(uint)
    impl_to_tokens!(u8)
    impl_to_tokens!(u16)
    impl_to_tokens!(u32)
    impl_to_tokens!(u64)

    pub trait ExtParseUtils {
        fn parse_item(&self, s: @str) -> @ast::item;
        fn parse_expr(&self, s: @str) -> @ast::Expr;
        fn parse_stmt(&self, s: @str) -> @ast::Stmt;
        fn parse_tts(&self, s: @str) -> ~[ast::token_tree];
    }

    impl ExtParseUtils for ExtCtxt {

        fn parse_item(&self, s: @str) -> @ast::item {
            let res = parse::parse_item_from_source_str(
                @"<quote expansion>",
                s,
                self.cfg(),
                ~[],
                self.parse_sess());
            match res {
                Some(ast) => ast,
                None => {
                    error!("Parse error with ```\n{}\n```", s);
                    fail!()
                }
            }
        }

        fn parse_stmt(&self, s: @str) -> @ast::Stmt {
            parse::parse_stmt_from_source_str(
                @"<quote expansion>",
                s,
                self.cfg(),
                ~[],
                self.parse_sess())
        }

        fn parse_expr(&self, s: @str) -> @ast::Expr {
            parse::parse_expr_from_source_str(
                @"<quote expansion>",
                s,
                self.cfg(),
                self.parse_sess())
        }

        fn parse_tts(&self, s: @str) -> ~[ast::token_tree] {
            parse::parse_tts_from_source_str(
                @"<quote expansion>",
                s,
                self.cfg(),
                self.parse_sess())
        }
    }

}

pub fn expand_quote_tokens(cx: @ExtCtxt,
                           sp: Span,
                           tts: &[ast::token_tree]) -> base::MacResult {
    let (cx_expr, expr) = expand_tts(cx, sp, tts);
    let expanded = expand_wrapper(cx, sp, cx_expr, expr);
    base::MRExpr(expanded)
}

pub fn expand_quote_expr(cx: @ExtCtxt,
                         sp: Span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    let expanded = expand_parse_call(cx, sp, "parse_expr", ~[], tts);
    base::MRExpr(expanded)
}

pub fn expand_quote_item(cx: @ExtCtxt,
                         sp: Span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    let e_attrs = cx.expr_vec_uniq(sp, ~[]);
    let expanded = expand_parse_call(cx, sp, "parse_item",
                                    ~[e_attrs], tts);
    base::MRExpr(expanded)
}

pub fn expand_quote_pat(cx: @ExtCtxt,
                        sp: Span,
                        tts: &[ast::token_tree]) -> base::MacResult {
    let e_refutable = cx.expr_lit(sp, ast::lit_bool(true));
    let expanded = expand_parse_call(cx, sp, "parse_pat",
                                    ~[e_refutable], tts);
    base::MRExpr(expanded)
}

pub fn expand_quote_ty(cx: @ExtCtxt,
                       sp: Span,
                       tts: &[ast::token_tree]) -> base::MacResult {
    let e_param_colons = cx.expr_lit(sp, ast::lit_bool(false));
    let expanded = expand_parse_call(cx, sp, "parse_ty",
                                     ~[e_param_colons], tts);
    base::MRExpr(expanded)
}

pub fn expand_quote_stmt(cx: @ExtCtxt,
                         sp: Span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    let e_attrs = cx.expr_vec_uniq(sp, ~[]);
    let expanded = expand_parse_call(cx, sp, "parse_stmt",
                                    ~[e_attrs], tts);
    base::MRExpr(expanded)
}

fn ids_ext(strs: ~[~str]) -> ~[ast::Ident] {
    strs.map(|str| str_to_ident(*str))
}

fn id_ext(str: &str) -> ast::Ident {
    str_to_ident(str)
}

// Lift an ident to the expr that evaluates to that ident.
fn mk_ident(cx: @ExtCtxt, sp: Span, ident: ast::Ident) -> @ast::Expr {
    let e_str = cx.expr_str(sp, cx.str_of(ident));
    cx.expr_method_call(sp,
                        cx.expr_ident(sp, id_ext("ext_cx")),
                        id_ext("ident_of"),
                        ~[e_str])
}

fn mk_binop(cx: @ExtCtxt, sp: Span, bop: token::binop) -> @ast::Expr {
    let name = match bop {
        PLUS => "PLUS",
        MINUS => "MINUS",
        STAR => "STAR",
        SLASH => "SLASH",
        PERCENT => "PERCENT",
        CARET => "CARET",
        AND => "AND",
        OR => "OR",
        SHL => "SHL",
        SHR => "SHR"
    };
    cx.expr_ident(sp, id_ext(name))
}

fn mk_token(cx: @ExtCtxt, sp: Span, tok: &token::Token) -> @ast::Expr {

    match *tok {
        BINOP(binop) => {
            return cx.expr_call_ident(sp,
                                      id_ext("BINOP"),
                                      ~[mk_binop(cx, sp, binop)]);
        }
        BINOPEQ(binop) => {
            return cx.expr_call_ident(sp,
                                      id_ext("BINOPEQ"),
                                      ~[mk_binop(cx, sp, binop)]);
        }

        LIT_CHAR(i) => {
            let s_ity = ~"ty_char";
            let e_ity = cx.expr_ident(sp, id_ext(s_ity));

            let e_char = cx.expr_lit(sp, ast::lit_char(i));

            return cx.expr_call_ident(sp, id_ext("LIT_CHAR"), ~[e_char, e_ity]);
        }

        LIT_INT(i, ity) => {
            let s_ity = match ity {
                ast::ty_i => ~"ty_i",
                ast::ty_i8 => ~"ty_i8",
                ast::ty_i16 => ~"ty_i16",
                ast::ty_i32 => ~"ty_i32",
                ast::ty_i64 => ~"ty_i64"
            };
            let e_ity = cx.expr_ident(sp, id_ext(s_ity));

            let e_i64 = cx.expr_lit(sp, ast::lit_int(i, ast::ty_i64));

            return cx.expr_call_ident(sp,
                                      id_ext("LIT_INT"),
                                      ~[e_i64, e_ity]);
        }

        LIT_UINT(u, uty) => {
            let s_uty = match uty {
                ast::ty_u => ~"ty_u",
                ast::ty_u8 => ~"ty_u8",
                ast::ty_u16 => ~"ty_u16",
                ast::ty_u32 => ~"ty_u32",
                ast::ty_u64 => ~"ty_u64"
            };
            let e_uty = cx.expr_ident(sp, id_ext(s_uty));

            let e_u64 = cx.expr_lit(sp, ast::lit_uint(u, ast::ty_u64));

            return cx.expr_call_ident(sp,
                                      id_ext("LIT_UINT"),
                                      ~[e_u64, e_uty]);
        }

        LIT_INT_UNSUFFIXED(i) => {
            let e_i64 = cx.expr_lit(sp, ast::lit_int(i, ast::ty_i64));

            return cx.expr_call_ident(sp,
                                      id_ext("LIT_INT_UNSUFFIXED"),
                                      ~[e_i64]);
        }

        LIT_FLOAT(fident, fty) => {
            let s_fty = match fty {
                ast::ty_f32 => ~"ty_f32",
                ast::ty_f64 => ~"ty_f64"
            };
            let e_fty = cx.expr_ident(sp, id_ext(s_fty));

            let e_fident = mk_ident(cx, sp, fident);

            return cx.expr_call_ident(sp,
                                      id_ext("LIT_FLOAT"),
                                      ~[e_fident, e_fty]);
        }

        LIT_STR(ident) => {
            return cx.expr_call_ident(sp,
                                      id_ext("LIT_STR"),
                                      ~[mk_ident(cx, sp, ident)]);
        }

        LIT_STR_RAW(ident, n) => {
            return cx.expr_call_ident(sp,
                                      id_ext("LIT_STR_RAW"),
                                      ~[mk_ident(cx, sp, ident),
                                        cx.expr_uint(sp, n)]);
        }

        IDENT(ident, b) => {
            return cx.expr_call_ident(sp,
                                      id_ext("IDENT"),
                                      ~[mk_ident(cx, sp, ident),
                                        cx.expr_bool(sp, b)]);
        }

        LIFETIME(ident) => {
            return cx.expr_call_ident(sp,
                                      id_ext("LIFETIME"),
                                      ~[mk_ident(cx, sp, ident)]);
        }

        DOC_COMMENT(ident) => {
            return cx.expr_call_ident(sp,
                                      id_ext("DOC_COMMENT"),
                                      ~[mk_ident(cx, sp, ident)]);
        }

        INTERPOLATED(_) => fail!("quote! with interpolated token"),

        _ => ()
    }

    let name = match *tok {
        EQ => "EQ",
        LT => "LT",
        LE => "LE",
        EQEQ => "EQEQ",
        NE => "NE",
        GE => "GE",
        GT => "GT",
        ANDAND => "ANDAND",
        OROR => "OROR",
        NOT => "NOT",
        TILDE => "TILDE",
        AT => "AT",
        DOT => "DOT",
        DOTDOT => "DOTDOT",
        COMMA => "COMMA",
        SEMI => "SEMI",
        COLON => "COLON",
        MOD_SEP => "MOD_SEP",
        RARROW => "RARROW",
        LARROW => "LARROW",
        DARROW => "DARROW",
        FAT_ARROW => "FAT_ARROW",
        LPAREN => "LPAREN",
        RPAREN => "RPAREN",
        LBRACKET => "LBRACKET",
        RBRACKET => "RBRACKET",
        LBRACE => "LBRACE",
        RBRACE => "RBRACE",
        POUND => "POUND",
        DOLLAR => "DOLLAR",
        UNDERSCORE => "UNDERSCORE",
        EOF => "EOF",
        _ => fail!()
    };
    cx.expr_ident(sp, id_ext(name))
}


fn mk_tt(cx: @ExtCtxt, sp: Span, tt: &ast::token_tree)
    -> ~[@ast::Stmt] {

    match *tt {

        ast::tt_tok(sp, ref tok) => {
            let e_sp = cx.expr_ident(sp, id_ext("sp"));
            let e_tok = cx.expr_call_ident(sp,
                                           id_ext("tt_tok"),
                                           ~[e_sp, mk_token(cx, sp, tok)]);
            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("push"),
                                    ~[e_tok]);
            ~[cx.stmt_expr(e_push)]
        }

        ast::tt_delim(ref tts) => mk_tts(cx, sp, **tts),
        ast::tt_seq(..) => fail!("tt_seq in quote!"),

        ast::tt_nonterminal(sp, ident) => {

            // tt.push_all_move($ident.to_tokens(ext_cx))

            let e_to_toks =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, ident),
                                    id_ext("to_tokens"),
                                    ~[cx.expr_ident(sp, id_ext("ext_cx"))]);

            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("push_all_move"),
                                    ~[e_to_toks]);

            ~[cx.stmt_expr(e_push)]
        }
    }
}

fn mk_tts(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> ~[@ast::Stmt] {
    let mut ss = ~[];
    for tt in tts.iter() {
        ss.push_all_move(mk_tt(cx, sp, tt));
    }
    ss
}

fn expand_tts(cx: @ExtCtxt,
              sp: Span,
              tts: &[ast::token_tree]) -> (@ast::Expr, @ast::Expr) {

    // NB: It appears that the main parser loses its mind if we consider
    // $foo as a tt_nonterminal during the main parse, so we have to re-parse
    // under quote_depth > 0. This is silly and should go away; the _guess_ is
    // it has to do with transition away from supporting old-style macros, so
    // try removing it when enough of them are gone.

    let p = parse::new_parser_from_tts(
        cx.parse_sess(),
        cx.cfg(),
        tts.to_owned()
    );
    *p.quote_depth += 1u;

    let cx_expr = p.parse_expr();
    if !p.eat(&token::COMMA) {
        p.fatal("Expected token `,`");
    }

    let tts = p.parse_all_token_trees();
    p.abort_if_errors();

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
                                   ~[]);

    let stmt_let_sp = cx.stmt_let(sp, false,
                                  id_ext("sp"),
                                  e_sp);

    let stmt_let_tt = cx.stmt_let(sp, true,
                                  id_ext("tt"),
                                  cx.expr_vec_uniq(sp, ~[]));

    let block = cx.expr_block(
        cx.block_all(sp,
                     ~[],
                     ~[stmt_let_sp, stmt_let_tt] + mk_tts(cx, sp, tts),
                     Some(cx.expr_ident(sp, id_ext("tt")))));

    (cx_expr, block)
}

fn expand_wrapper(cx: @ExtCtxt,
                  sp: Span,
                  cx_expr: @ast::Expr,
                  expr: @ast::Expr) -> @ast::Expr {
    let uses = ~[ cx.view_use_glob(sp, ast::public,
                                   ids_ext(~[~"syntax",
                                             ~"ext",
                                             ~"quote",
                                             ~"rt"])) ];

    let stmt_let_ext_cx = cx.stmt_let(sp, false, id_ext("ext_cx"), cx_expr);

    cx.expr_block(cx.block_all(sp, uses, ~[stmt_let_ext_cx], Some(expr)))
}

fn expand_parse_call(cx: @ExtCtxt,
                     sp: Span,
                     parse_method: &str,
                     arg_exprs: ~[@ast::Expr],
                     tts: &[ast::token_tree]) -> @ast::Expr {
    let (cx_expr, tts_expr) = expand_tts(cx, sp, tts);

    let cfg_call = || cx.expr_method_call(
        sp, cx.expr_ident(sp, id_ext("ext_cx")),
        id_ext("cfg"), ~[]);

    let parse_sess_call = || cx.expr_method_call(
        sp, cx.expr_ident(sp, id_ext("ext_cx")),
        id_ext("parse_sess"), ~[]);

    let new_parser_call =
        cx.expr_call(sp,
                     cx.expr_ident(sp, id_ext("new_parser_from_tts")),
                     ~[parse_sess_call(), cfg_call(), tts_expr]);

    let expr = cx.expr_method_call(sp, new_parser_call, id_ext(parse_method),
                                   arg_exprs);

    expand_wrapper(cx, sp, cx_expr, expr)
}

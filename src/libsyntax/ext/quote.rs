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
use codemap::{BytePos, Pos, span};
use ext::base::ext_ctxt;
use ext::base;
use ext::build;
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
    use ext::base::ext_ctxt;
    use parse;
    use print::pprust;

    use core::str;

    pub use ast::*;
    pub use parse::token::*;
    pub use parse::new_parser_from_tts;
    pub use codemap::{BytePos, span, dummy_spanned};

    use print::pprust;
    use print::pprust::{item_to_str, ty_to_str};

    pub trait ToTokens {
        pub fn to_tokens(&self, _cx: @ext_ctxt) -> ~[token_tree];
    }

    impl ToTokens for ~[token_tree] {
        pub fn to_tokens(&self, _cx: @ext_ctxt) -> ~[token_tree] {
            copy *self
        }
    }

    /* Should be (when bugs in default methods are fixed):

    trait ToSource : ToTokens {
        // Takes a thing and generates a string containing rust code for it.
        pub fn to_source(cx: @ext_ctxt) -> ~str;

        // If you can make source, you can definitely make tokens.
        pub fn to_tokens(cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    */

    pub trait ToSource {
        // Takes a thing and generates a string containing rust code for it.
        pub fn to_source(&self, cx: @ext_ctxt) -> ~str;
    }

    impl ToSource for ast::ident {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            copy *cx.parse_sess().interner.get(*self)
        }
    }

    impl ToSource for @ast::item {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            item_to_str(*self, cx.parse_sess().interner)
        }
    }

    impl<'self> ToSource for &'self [@ast::item] {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            str::connect(self.map(|i| i.to_source(cx)), ~"\n\n")
        }
    }

    impl ToSource for @ast::Ty {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            ty_to_str(*self, cx.parse_sess().interner)
        }
    }

    impl<'self> ToSource for &'self [@ast::Ty] {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            str::connect(self.map(|i| i.to_source(cx)), ~", ")
        }
    }

    impl ToSource for Generics {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            pprust::generics_to_str(self, cx.parse_sess().interner)
        }
    }

    impl ToSource for @ast::expr {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            pprust::expr_to_str(*self, cx.parse_sess().interner)
        }
    }

    impl ToSource for ast::blk {
        fn to_source(&self, cx: @ext_ctxt) -> ~str {
            pprust::block_to_str(self, cx.parse_sess().interner)
        }
    }

    impl<'self> ToSource for &'self str {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_str(@str::from_slice(*self)));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for int {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for i8 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i8));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for i16 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i16));
            pprust::lit_to_str(@lit)
        }
    }


    impl ToSource for i32 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i32));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for i64 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_int(*self as i64, ast::ty_i64));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for uint {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for u8 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u8));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for u16 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u16));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for u32 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u32));
            pprust::lit_to_str(@lit)
        }
    }

    impl ToSource for u64 {
        fn to_source(&self, _cx: @ext_ctxt) -> ~str {
            let lit = dummy_spanned(ast::lit_uint(*self as u64, ast::ty_u64));
            pprust::lit_to_str(@lit)
        }
    }

    // Alas ... we write these out instead. All redundant.

    impl ToTokens for ast::ident {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for @ast::item {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl<'self> ToTokens for &'self [@ast::item] {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for @ast::Ty {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl<'self> ToTokens for &'self [@ast::Ty] {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for Generics {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for @ast::expr {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for ast::blk {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl<'self> ToTokens for &'self str {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for int {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for i8 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for i16 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for i32 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for i64 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for uint {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for u8 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for u16 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for u32 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    impl ToTokens for u64 {
        fn to_tokens(&self, cx: @ext_ctxt) -> ~[token_tree] {
            cx.parse_tts(self.to_source(cx))
        }
    }

    pub trait ExtParseUtils {
        fn parse_item(&self, s: ~str) -> @ast::item;
        fn parse_expr(&self, s: ~str) -> @ast::expr;
        fn parse_stmt(&self, s: ~str) -> @ast::stmt;
        fn parse_tts(&self, s: ~str) -> ~[ast::token_tree];
    }

    impl ExtParseUtils for @ext_ctxt {

        fn parse_item(&self, s: ~str) -> @ast::item {
            let res = parse::parse_item_from_source_str(
                ~"<quote expansion>",
                @(copy s),
                self.cfg(),
                ~[],
                self.parse_sess());
            match res {
                Some(ast) => ast,
                None => {
                    error!("Parse error with ```\n%s\n```", s);
                    fail!()
                }
            }
        }

        fn parse_stmt(&self, s: ~str) -> @ast::stmt {
            parse::parse_stmt_from_source_str(
                ~"<quote expansion>",
                @(copy s),
                self.cfg(),
                ~[],
                self.parse_sess())
        }

        fn parse_expr(&self, s: ~str) -> @ast::expr {
            parse::parse_expr_from_source_str(
                ~"<quote expansion>",
                @(copy s),
                self.cfg(),
                self.parse_sess())
        }

        fn parse_tts(&self, s: ~str) -> ~[ast::token_tree] {
            parse::parse_tts_from_source_str(
                ~"<quote expansion>",
                @(copy s),
                self.cfg(),
                self.parse_sess())
        }
    }

}

pub fn expand_quote_tokens(cx: @ext_ctxt,
                           sp: span,
                           tts: &[ast::token_tree]) -> base::MacResult {
    base::MRExpr(expand_tts(cx, sp, tts))
}

pub fn expand_quote_expr(cx: @ext_ctxt,
                         sp: span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    base::MRExpr(expand_parse_call(cx, sp, "parse_expr", ~[], tts))
}

pub fn expand_quote_item(cx: @ext_ctxt,
                         sp: span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    let e_attrs = build::mk_uniq_vec_e(cx, sp, ~[]);
    base::MRExpr(expand_parse_call(cx, sp, "parse_item",
                                    ~[e_attrs], tts))
}

pub fn expand_quote_pat(cx: @ext_ctxt,
                        sp: span,
                        tts: &[ast::token_tree]) -> base::MacResult {
    let e_refutable = build::mk_lit(cx, sp, ast::lit_bool(true));
    base::MRExpr(expand_parse_call(cx, sp, "parse_pat",
                                    ~[e_refutable], tts))
}

pub fn expand_quote_ty(cx: @ext_ctxt,
                       sp: span,
                       tts: &[ast::token_tree]) -> base::MacResult {
    let e_param_colons = build::mk_lit(cx, sp, ast::lit_bool(false));
    base::MRExpr(expand_parse_call(cx, sp, "parse_ty",
                                    ~[e_param_colons], tts))
}

pub fn expand_quote_stmt(cx: @ext_ctxt,
                         sp: span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    let e_attrs = build::mk_uniq_vec_e(cx, sp, ~[]);
    base::MRExpr(expand_parse_call(cx, sp, "parse_stmt",
                                    ~[e_attrs], tts))
}

fn ids_ext(cx: @ext_ctxt, strs: ~[~str]) -> ~[ast::ident] {
    strs.map(|str| cx.parse_sess().interner.intern(*str))
}

fn id_ext(cx: @ext_ctxt, str: &str) -> ast::ident {
    cx.parse_sess().interner.intern(str)
}

// Lift an ident to the expr that evaluates to that ident.
fn mk_ident(cx: @ext_ctxt, sp: span, ident: ast::ident) -> @ast::expr {
    let e_str = build::mk_uniq_str(cx, sp, cx.str_of(ident));
    build::mk_method_call(cx, sp,
                          build::mk_path(cx, sp, ids_ext(cx, ~[~"ext_cx"])),
                          id_ext(cx, "ident_of"),
                          ~[e_str])
}

fn mk_bytepos(cx: @ext_ctxt, sp: span, bpos: BytePos) -> @ast::expr {
    let path = ids_ext(cx, ~[~"BytePos"]);
    let arg = build::mk_uint(cx, sp, bpos.to_uint());
    build::mk_call(cx, sp, path, ~[arg])
}

fn mk_binop(cx: @ext_ctxt, sp: span, bop: token::binop) -> @ast::expr {
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
    build::mk_path(cx, sp,
                   ids_ext(cx, ~[name.to_owned()]))
}

fn mk_token(cx: @ext_ctxt, sp: span, tok: token::Token) -> @ast::expr {

    match tok {
        BINOP(binop) => {
            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"BINOP"]),
                                  ~[mk_binop(cx, sp, binop)]);
        }
        BINOPEQ(binop) => {
            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"BINOPEQ"]),
                                  ~[mk_binop(cx, sp, binop)]);
        }

        LIT_INT(i, ity) => {
            let s_ity = match ity {
                ast::ty_i => ~"ty_i",
                ast::ty_char => ~"ty_char",
                ast::ty_i8 => ~"ty_i8",
                ast::ty_i16 => ~"ty_i16",
                ast::ty_i32 => ~"ty_i32",
                ast::ty_i64 => ~"ty_i64"
            };
            let e_ity =
                build::mk_path(cx, sp,
                               ids_ext(cx, ~[s_ity]));

            let e_i64 = build::mk_lit(cx, sp, ast::lit_int(i, ast::ty_i64));

            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"LIT_INT"]),
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
            let e_uty =
                build::mk_path(cx, sp,
                               ids_ext(cx, ~[s_uty]));

            let e_u64 = build::mk_lit(cx, sp, ast::lit_uint(u, ast::ty_u64));

            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"LIT_UINT"]),
                                  ~[e_u64, e_uty]);
        }

        LIT_INT_UNSUFFIXED(i) => {
            let e_i64 = build::mk_lit(cx, sp,
                                      ast::lit_int(i, ast::ty_i64));

            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"LIT_INT_UNSUFFIXED"]),
                                  ~[e_i64]);
        }

        LIT_FLOAT(fident, fty) => {
            let s_fty = match fty {
                ast::ty_f => ~"ty_f",
                ast::ty_f32 => ~"ty_f32",
                ast::ty_f64 => ~"ty_f64"
            };
            let e_fty =
                build::mk_path(cx, sp,
                               ids_ext(cx, ~[s_fty]));

            let e_fident = mk_ident(cx, sp, fident);

            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"LIT_FLOAT"]),
                                  ~[e_fident, e_fty]);
        }

        LIT_STR(ident) => {
            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"LIT_STR"]),
                                  ~[mk_ident(cx, sp, ident)]);
        }

        IDENT(ident, b) => {
            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"IDENT"]),
                                  ~[mk_ident(cx, sp, ident),
                                    build::mk_lit(cx, sp, ast::lit_bool(b))]);
        }

        LIFETIME(ident) => {
            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"LIFETIME"]),
                                  ~[mk_ident(cx, sp, ident)]);
        }

        DOC_COMMENT(ident) => {
            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"DOC_COMMENT"]),
                                  ~[mk_ident(cx, sp, ident)]);
        }

        INTERPOLATED(_) => fail!(~"quote! with interpolated token"),

        _ => ()
    }

    let name = match tok {
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
    build::mk_path(cx, sp,
                   ids_ext(cx, ~[name.to_owned()]))
}


fn mk_tt(cx: @ext_ctxt, sp: span, tt: &ast::token_tree)
    -> ~[@ast::stmt] {

    match *tt {

        ast::tt_tok(sp, ref tok) => {
            let e_sp = build::mk_path(cx, sp,
                                      ids_ext(cx, ~[~"sp"]));
            let e_tok =
                build::mk_call(cx, sp,
                               ids_ext(cx, ~[~"tt_tok"]),
                               ~[e_sp, mk_token(cx, sp, *tok)]);
            let e_push =
                build::mk_method_call(cx, sp,
                                      build::mk_path(cx, sp, ids_ext(cx, ~[~"tt"])),
                                      id_ext(cx, "push"),
                                      ~[e_tok]);
            ~[build::mk_stmt(cx, sp, e_push)]

        }

        ast::tt_delim(ref tts) => mk_tts(cx, sp, *tts),
        ast::tt_seq(*) => fail!(~"tt_seq in quote!"),

        ast::tt_nonterminal(sp, ident) => {

            // tt.push_all_move($ident.to_tokens(ext_cx))

            let e_to_toks =
                build::mk_method_call(cx, sp,
                                      build::mk_path(cx, sp, ~[ident]),
                                      id_ext(cx, "to_tokens"),
                                      ~[build::mk_path(cx, sp,
                                                       ids_ext(cx, ~[~"ext_cx"]))]);

            let e_push =
                build::mk_method_call(cx, sp,
                                      build::mk_path(cx, sp, ids_ext(cx, ~[~"tt"])),
                                      id_ext(cx, "push_all_move"),
                                      ~[e_to_toks]);

            ~[build::mk_stmt(cx, sp, e_push)]
        }
    }
}

fn mk_tts(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> ~[@ast::stmt] {
    let mut ss = ~[];
    for tts.each |tt| {
        ss.push_all_move(mk_tt(cx, sp, tt));
    }
    ss
}

fn expand_tts(cx: @ext_ctxt,
              sp: span,
              tts: &[ast::token_tree]) -> @ast::expr {

    // NB: It appears that the main parser loses its mind if we consider
    // $foo as a tt_nonterminal during the main parse, so we have to re-parse
    // under quote_depth > 0. This is silly and should go away; the _guess_ is
    // it has to do with transition away from supporting old-style macros, so
    // try removing it when enough of them are gone.

    let p = parse::new_parser_from_tts(
        cx.parse_sess(),
        cx.cfg(),
        vec::from_slice(tts)
    );
    *p.quote_depth += 1u;
    let tts = p.parse_all_token_trees();
    p.abort_if_errors();

    // We want to emit a block expression that does a sequence of 'use's to
    // import the runtime module, followed by a tt-building expression.

    let uses = ~[ build::mk_glob_use(cx, sp, ids_ext(cx, ~[~"syntax",
                                                           ~"ext",
                                                           ~"quote",
                                                           ~"rt"])) ];

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
    // using proto! for example, will be attributed to the pipec.rs file in
    // libsyntax, which the user might not even have source to (unless they
    // happen to have a compiler on hand). Over all, the phase distinction
    // just makes quotes "hard to attribute". Possibly this could be fixed
    // by recreating some of the original qq machinery in the tt regime
    // (pushing fake FileMaps onto the parser to account for original sites
    // of quotes, for example) but at this point it seems not likely to be
    // worth the hassle.

    let e_sp = build::mk_method_call(cx, sp,
                                     build::mk_path(cx, sp, ids_ext(cx, ~[~"ext_cx"])),
                                     id_ext(cx, "call_site"),
                                     ~[]);

    let stmt_let_sp = build::mk_local(cx, sp, false,
                                      id_ext(cx, "sp"),
                                      e_sp);

    let stmt_let_tt = build::mk_local(cx, sp, true,
                                      id_ext(cx, "tt"),
                                      build::mk_uniq_vec_e(cx, sp, ~[]));

    build::mk_block(cx, sp, uses,
                    ~[stmt_let_sp,
                      stmt_let_tt] + mk_tts(cx, sp, tts),
                    Some(build::mk_path(cx, sp,
                                        ids_ext(cx, ~[~"tt"]))))
}

fn expand_parse_call(cx: @ext_ctxt,
                     sp: span,
                     parse_method: &str,
                     arg_exprs: ~[@ast::expr],
                     tts: &[ast::token_tree]) -> @ast::expr {
    let tts_expr = expand_tts(cx, sp, tts);

    let cfg_call = || build::mk_method_call(
        cx, sp, build::mk_path(cx, sp, ids_ext(cx, ~[~"ext_cx"])),
        id_ext(cx, "cfg"), ~[]);

    let parse_sess_call = || build::mk_method_call(
        cx, sp, build::mk_path(cx, sp, ids_ext(cx, ~[~"ext_cx"])),
        id_ext(cx, "parse_sess"), ~[]);

    let new_parser_call =
        build::mk_call_global(cx, sp,
                              ids_ext(cx, ~[~"syntax",
                                            ~"ext",
                                            ~"quote",
                                            ~"rt",
                                            ~"new_parser_from_tts"]),
                              ~[parse_sess_call(),
                                cfg_call(),
                                tts_expr]);

    build::mk_method_call(cx, sp,
                          new_parser_call,
                          id_ext(cx, parse_method),
                          arg_exprs)
}

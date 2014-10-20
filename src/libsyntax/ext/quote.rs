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
use ptr::P;

/**
*
* Quasiquoting works via token trees.
*
* This is registered as a set of expression syntax extension called quote!
* that lifts its argument token-tree to an AST representing the
* construction of the same token tree, with ast::TTNonterminal nodes
* interpreted as antiquotes (splices).
*
*/

pub mod rt {
    use ast;
    use codemap::Spanned;
    use ext::base::ExtCtxt;
    use parse::token;
    use parse;
    use print::pprust;
    use ptr::P;

    use ast::{TokenTree, Generics, Expr};

    pub use parse::new_parser_from_tts;
    pub use codemap::{BytePos, Span, dummy_spanned};

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
            let a = self.iter().flat_map(|t| t.to_tokens(cx).into_iter());
            FromIterator::from_iter(a)
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

    macro_rules! impl_to_source(
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
    )

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

    macro_rules! impl_to_source_slice(
        ($t:ty, $sep:expr) => (
            impl<'a> ToSource for &'a [$t] {
                fn to_source(&self) -> String {
                    slice_to_source($sep, *self)
                }
            }

            impl<'a> ToSourceWithHygiene for &'a [$t] {
                fn to_source_with_hygiene(&self) -> String {
                    slice_to_source_with_hygiene($sep, *self)
                }
            }
        )
    )

    impl ToSource for ast::Ident {
        fn to_source(&self) -> String {
            token::get_ident(*self).get().to_string()
        }
    }

    impl ToSourceWithHygiene for ast::Ident {
        fn to_source_with_hygiene(&self) -> String {
            self.encode_with_hygiene()
        }
    }

    impl_to_source!(ast::Ty, ty_to_string)
    impl_to_source!(ast::Block, block_to_string)
    impl_to_source!(ast::Arg, arg_to_string)
    impl_to_source!(Generics, generics_to_string)
    impl_to_source!(P<ast::Item>, item_to_string)
    impl_to_source!(P<ast::Method>, method_to_string)
    impl_to_source!(P<ast::Stmt>, stmt_to_string)
    impl_to_source!(P<ast::Expr>, expr_to_string)
    impl_to_source!(P<ast::Pat>, pat_to_string)
    impl_to_source!(ast::Arm, arm_to_string)
    impl_to_source_slice!(ast::Ty, ", ")
    impl_to_source_slice!(P<ast::Item>, "\n\n")

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

    impl<'a> ToSource for &'a str {
        fn to_source(&self) -> String {
            let lit = dummy_spanned(ast::LitStr(
                    token::intern_and_get_ident(*self), ast::CookedStr));
            pprust::lit_to_string(&lit)
        }
    }
    impl<'a> ToSourceWithHygiene for &'a str {
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

    macro_rules! impl_to_source_int(
        (signed, $t:ty, $tag:ident) => (
            impl ToSource for $t {
                fn to_source(&self) -> String {
                    let lit = ast::LitInt(*self as u64, ast::SignedIntLit(ast::$tag,
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
        (unsigned, $t:ty, $tag:ident) => (
            impl ToSource for $t {
                fn to_source(&self) -> String {
                    let lit = ast::LitInt(*self as u64, ast::UnsignedIntLit(ast::$tag));
                    pprust::lit_to_string(&dummy_spanned(lit))
                }
            }
            impl ToSourceWithHygiene for $t {
                fn to_source_with_hygiene(&self) -> String {
                    self.to_source()
                }
            }
        );
    )

    impl_to_source_int!(signed, int, TyI)
    impl_to_source_int!(signed, i8,  TyI8)
    impl_to_source_int!(signed, i16, TyI16)
    impl_to_source_int!(signed, i32, TyI32)
    impl_to_source_int!(signed, i64, TyI64)

    impl_to_source_int!(unsigned, uint, TyU)
    impl_to_source_int!(unsigned, u8,   TyU8)
    impl_to_source_int!(unsigned, u16,  TyU16)
    impl_to_source_int!(unsigned, u32,  TyU32)
    impl_to_source_int!(unsigned, u64,  TyU64)

    // Alas ... we write these out instead. All redundant.

    macro_rules! impl_to_tokens(
        ($t:ty) => (
            impl ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    cx.parse_tts_with_hygiene(self.to_source_with_hygiene())
                }
            }
        )
    )

    macro_rules! impl_to_tokens_lifetime(
        ($t:ty) => (
            impl<'a> ToTokens for $t {
                fn to_tokens(&self, cx: &ExtCtxt) -> Vec<TokenTree> {
                    cx.parse_tts_with_hygiene(self.to_source_with_hygiene())
                }
            }
        )
    )

    impl_to_tokens!(ast::Ident)
    impl_to_tokens!(P<ast::Item>)
    impl_to_tokens!(P<ast::Pat>)
    impl_to_tokens!(ast::Arm)
    impl_to_tokens!(P<ast::Method>)
    impl_to_tokens_lifetime!(&'a [P<ast::Item>])
    impl_to_tokens!(ast::Ty)
    impl_to_tokens_lifetime!(&'a [ast::Ty])
    impl_to_tokens!(Generics)
    impl_to_tokens!(P<ast::Stmt>)
    impl_to_tokens!(P<ast::Expr>)
    impl_to_tokens!(ast::Block)
    impl_to_tokens!(ast::Arg)
    impl_to_tokens!(ast::Attribute_)
    impl_to_tokens_lifetime!(&'a str)
    impl_to_tokens!(())
    impl_to_tokens!(char)
    impl_to_tokens!(bool)
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
            let res = parse::parse_item_from_source_str(
                "<quote expansion>".to_string(),
                s,
                self.cfg(),
                self.parse_sess());
            match res {
                Some(ast) => ast,
                None => {
                    error!("parse error");
                    fail!()
                }
            }
        }

        fn parse_stmt(&self, s: String) -> P<ast::Stmt> {
            parse::parse_stmt_from_source_str("<quote expansion>".to_string(),
                                              s,
                                              self.cfg(),
                                              Vec::new(),
                                              self.parse_sess())
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

}

pub fn expand_quote_tokens<'cx>(cx: &'cx mut ExtCtxt,
                                sp: Span,
                                tts: &[ast::TokenTree])
                                -> Box<base::MacResult+'cx> {
    let (cx_expr, expr) = expand_tts(cx, sp, tts);
    let expanded = expand_wrapper(cx, sp, cx_expr, expr);
    base::MacExpr::new(expanded)
}

pub fn expand_quote_expr<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_expr", Vec::new(), tts);
    base::MacExpr::new(expanded)
}

pub fn expand_quote_item<'cx>(cx: &mut ExtCtxt,
                              sp: Span,
                              tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_item_with_outer_attributes",
                                    vec!(), tts);
    base::MacExpr::new(expanded)
}

pub fn expand_quote_pat<'cx>(cx: &'cx mut ExtCtxt,
                             sp: Span,
                             tts: &[ast::TokenTree])
                             -> Box<base::MacResult+'cx> {
    let expanded = expand_parse_call(cx, sp, "parse_pat", vec!(), tts);
    base::MacExpr::new(expanded)
}

pub fn expand_quote_arm(cx: &mut ExtCtxt,
                        sp: Span,
                        tts: &[ast::TokenTree])
                        -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_arm", vec!(), tts);
    base::MacExpr::new(expanded)
}

pub fn expand_quote_ty(cx: &mut ExtCtxt,
                       sp: Span,
                       tts: &[ast::TokenTree])
                       -> Box<base::MacResult+'static> {
    let e_param_colons = cx.expr_lit(sp, ast::LitBool(false));
    let expanded = expand_parse_call(cx, sp, "parse_ty",
                                     vec!(e_param_colons), tts);
    base::MacExpr::new(expanded)
}

pub fn expand_quote_method(cx: &mut ExtCtxt,
                           sp: Span,
                           tts: &[ast::TokenTree])
                           -> Box<base::MacResult+'static> {
    let expanded = expand_parse_call(cx, sp, "parse_method_with_outer_attributes",
                                     vec!(), tts);
    base::MacExpr::new(expanded)
}

pub fn expand_quote_stmt(cx: &mut ExtCtxt,
                         sp: Span,
                         tts: &[ast::TokenTree])
                         -> Box<base::MacResult+'static> {
    let e_attrs = cx.expr_vec_ng(sp);
    let expanded = expand_parse_call(cx, sp, "parse_stmt",
                                    vec!(e_attrs), tts);
    base::MacExpr::new(expanded)
}

fn ids_ext(strs: Vec<String> ) -> Vec<ast::Ident> {
    strs.iter().map(|str| str_to_ident((*str).as_slice())).collect()
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

fn mk_binop(cx: &ExtCtxt, sp: Span, bop: token::BinOp) -> P<ast::Expr> {
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
    mk_token_path(cx, sp, name)
}

fn mk_token(cx: &ExtCtxt, sp: Span, tok: &token::Token) -> P<ast::Expr> {

    match *tok {
        BINOP(binop) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "BINOP"), vec!(mk_binop(cx, sp, binop)));
        }
        BINOPEQ(binop) => {
            return cx.expr_call(sp, mk_token_path(cx, sp, "BINOPEQ"),
                                vec!(mk_binop(cx, sp, binop)));
        }

        LIT_BYTE(i) => {
            let e_byte = mk_name(cx, sp, i.ident());

            return cx.expr_call(sp, mk_token_path(cx, sp, "LIT_BYTE"), vec!(e_byte));
        }

        LIT_CHAR(i) => {
            let e_char = mk_name(cx, sp, i.ident());

            return cx.expr_call(sp, mk_token_path(cx, sp, "LIT_CHAR"), vec!(e_char));
        }

        LIT_INTEGER(i) => {
            let e_int = mk_name(cx, sp, i.ident());
            return cx.expr_call(sp, mk_token_path(cx, sp, "LIT_INTEGER"), vec!(e_int));
        }

        LIT_FLOAT(fident) => {
            let e_fident = mk_name(cx, sp, fident.ident());
            return cx.expr_call(sp, mk_token_path(cx, sp, "LIT_FLOAT"), vec!(e_fident));
        }

        LIT_STR(ident) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "LIT_STR"),
                                vec!(mk_name(cx, sp, ident.ident())));
        }

        LIT_STR_RAW(ident, n) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "LIT_STR_RAW"),
                                vec!(mk_name(cx, sp, ident.ident()), cx.expr_uint(sp, n)));
        }

        IDENT(ident, b) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "IDENT"),
                                vec!(mk_ident(cx, sp, ident), cx.expr_bool(sp, b)));
        }

        LIFETIME(ident) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "LIFETIME"),
                                vec!(mk_ident(cx, sp, ident)));
        }

        DOC_COMMENT(ident) => {
            return cx.expr_call(sp,
                                mk_token_path(cx, sp, "DOC_COMMENT"),
                                vec!(mk_name(cx, sp, ident.ident())));
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
    mk_token_path(cx, sp, name)
}


fn mk_tt(cx: &ExtCtxt, sp: Span, tt: &ast::TokenTree) -> Vec<P<ast::Stmt>> {
    match *tt {
        ast::TTTok(sp, ref tok) => {
            let e_sp = cx.expr_ident(sp, id_ext("_sp"));
            let e_tok = cx.expr_call(sp,
                                     mk_ast_path(cx, sp, "TTTok"),
                                     vec!(e_sp, mk_token(cx, sp, tok)));
            let e_push =
                cx.expr_method_call(sp,
                                    cx.expr_ident(sp, id_ext("tt")),
                                    id_ext("push"),
                                    vec!(e_tok));
            vec!(cx.stmt_expr(e_push))
        }

        ast::TTDelim(ref tts) => mk_tts(cx, sp, tts.as_slice()),
        ast::TTSeq(..) => fail!("TTSeq in quote!"),

        ast::TTNonterminal(sp, ident) => {

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

            vec!(cx.stmt_expr(e_push))
        }
    }
}

fn mk_tts(cx: &ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> Vec<P<ast::Stmt>> {
    let mut ss = Vec::new();
    for tt in tts.iter() {
        ss.extend(mk_tt(cx, sp, tt).into_iter());
    }
    ss
}

fn expand_tts(cx: &ExtCtxt, sp: Span, tts: &[ast::TokenTree])
              -> (P<ast::Expr>, P<ast::Expr>) {
    // NB: It appears that the main parser loses its mind if we consider
    // $foo as a TTNonterminal during the main parse, so we have to re-parse
    // under quote_depth > 0. This is silly and should go away; the _guess_ is
    // it has to do with transition away from supporting old-style macros, so
    // try removing it when enough of them are gone.

    let mut p = cx.new_parser_from_tts(tts);
    p.quote_depth += 1u;

    let cx_expr = p.parse_expr();
    if !p.eat(&token::COMMA) {
        p.fatal("expected token `,`");
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
                                   Vec::new());

    let stmt_let_sp = cx.stmt_let(sp, false,
                                  id_ext("_sp"),
                                  e_sp);

    let stmt_let_tt = cx.stmt_let(sp, true, id_ext("tt"), cx.expr_vec_ng(sp));

    let mut vector = vec!(stmt_let_sp, stmt_let_tt);
    vector.extend(mk_tts(cx, sp, tts.as_slice()).into_iter());
    let block = cx.expr_block(
        cx.block_all(sp,
                     Vec::new(),
                     vector,
                     Some(cx.expr_ident(sp, id_ext("tt")))));

    (cx_expr, block)
}

fn expand_wrapper(cx: &ExtCtxt,
                  sp: Span,
                  cx_expr: P<ast::Expr>,
                  expr: P<ast::Expr>) -> P<ast::Expr> {
    let uses = [
        &["syntax", "ext", "quote", "rt"],
    ].iter().map(|path| {
        let path = path.iter().map(|s| s.to_string()).collect();
        cx.view_use_glob(sp, ast::Inherited, ids_ext(path))
    }).collect();

    // Explicitly borrow to avoid moving from the invoker (#16992)
    let cx_expr_borrow = cx.expr_addr_of(sp, cx.expr_deref(sp, cx_expr));
    let stmt_let_ext_cx = cx.stmt_let(sp, false, id_ext("ext_cx"), cx_expr_borrow);

    cx.expr_block(cx.block_all(sp, uses, vec!(stmt_let_ext_cx), Some(expr)))
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

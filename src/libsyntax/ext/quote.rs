use mod ast;
use mod parse::token;

use codemap::span;
use ext::base::ext_ctxt;
use token::*;

/**
*
* Quasiquoting works via token trees.
*
* This is registered as a expression syntax extension called quote! that lifts
* its argument token-tree to an AST representing the construction of the same
* token tree, with ast::tt_nonterminal nodes interpreted as antiquotes
* (splices).
*
*/

pub fn expand_quote(cx: ext_ctxt,
                    sp: span,
                    tts: ~[ast::token_tree]) -> base::mac_result
{

    // NB: It appears that the main parser loses its mind if we consider
    // $foo as a tt_nonterminal during the main parse, so we have to re-parse
    // under quote_depth > 0. This is silly and should go away; the _guess_ is
    // it has to do with transition away from supporting old-style macros, so
    // try removing it when enough of them are gone.
    let p = parse::new_parser_from_tt(cx.parse_sess(), cx.cfg(), tts);
    p.quote_depth += 1u;
    let tq = dvec::DVec();
    while p.token != token::EOF {
        tq.push(p.parse_token_tree());
    }
    let tts = tq.get();

    // We want to emit a block expression that does a sequence of 'use's to
    // import the AST and token constructors, followed by a tt expression.
    let uses = ~[ build::mk_glob_use(cx, sp, ids_ext(cx, ~[~"syntax",
                                                           ~"ast"])),
                  build::mk_glob_use(cx, sp, ids_ext(cx, ~[~"syntax",
                                                           ~"parse",
                                                           ~"token"])) ];
    base::mr_expr(build::mk_block(cx, sp, uses, ~[],
                                  Some(mk_tt(cx, sp, &ast::tt_delim(tts)))))
}

fn ids_ext(cx: ext_ctxt, strs: ~[~str]) -> ~[ast::ident] {
    strs.map(|str| cx.parse_sess().interner.intern(@*str))
}

fn id_ext(cx: ext_ctxt, str: ~str) -> ast::ident {
    cx.parse_sess().interner.intern(@str)
}

fn mk_option_span(cx: ext_ctxt,
                  qsp: span,
                  sp: Option<span>) -> @ast::expr {
    match sp {
        None => build::mk_path(cx, qsp, ids_ext(cx, ~[~"None"])),
        Some(sp) => {
            build::mk_call(cx, qsp,
                           ids_ext(cx, ~[~"Some"]),
                           ~[build::mk_managed(cx, qsp,
                                               mk_span(cx, qsp, sp))])
        }
    }
}

fn mk_span(cx: ext_ctxt, qsp: span, sp: span) -> @ast::expr {

    let e_expn_info = match sp.expn_info {
        None => build::mk_path(cx, qsp, ids_ext(cx, ~[~"None"])),
        Some(@codemap::expanded_from(cr)) => {
            let e_callee =
                build::mk_rec_e(
                    cx, qsp,
                    ~[{ident: id_ext(cx, ~"name"),
                       ex: build::mk_uniq_str(cx, qsp,
                                              cr.callie.name)},
                      {ident: id_ext(cx, ~"span"),
                       ex: mk_option_span(cx, qsp, cr.callie.span)}]);

            let e_expn_info_ =
                build::mk_call(
                    cx, qsp,
                    ids_ext(cx, ~[~"expanded_from"]),
                    ~[build::mk_rec_e(
                        cx, qsp,
                        ~[{ident: id_ext(cx, ~"call_site"),
                           ex: mk_span(cx, qsp, cr.call_site)},
                          {ident: id_ext(cx, ~"callie"),
                           ex: e_callee}])]);

            build::mk_call(cx, qsp,
                           ids_ext(cx, ~[~"Some"]),
                           ~[build::mk_managed(cx, qsp, e_expn_info_)])
        }
    };

    build::mk_rec_e(cx, qsp,
                    ~[{ident: id_ext(cx, ~"lo"),
                       ex: build::mk_uint(cx, qsp, sp.lo) },

                      {ident: id_ext(cx, ~"hi"),
                       ex: build::mk_uint(cx, qsp, sp.hi) },

                      {ident: id_ext(cx, ~"expn_info"),
                       ex: e_expn_info}])
}

// Lift an ident to the expr that evaluates to that ident.
//
// NB: this identifies the interner used when re-parsing the token tree
// with the interner used during initial parse. This is _wrong_ and we
// should be emitting a &str here and the token type should be ok with
// &static/str or &session/str. Longer-term issue.
fn mk_ident(cx: ext_ctxt, sp: span, ident: ast::ident) -> @ast::expr {
    build::mk_struct_e(cx, sp,
                       ids_ext(cx, ~[~"ident"]),
                       ~[{ident: id_ext(cx, ~"repr"),
                          ex: build::mk_uint(cx, sp, ident.repr) }])
}


fn mk_binop(cx: ext_ctxt, sp: span, bop: token::binop) -> @ast::expr {
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

fn mk_token(cx: ext_ctxt, sp: span, tok: token::Token) -> @ast::expr {

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

        DOC_COMMENT(ident) => {
            return build::mk_call(cx, sp,
                                  ids_ext(cx, ~[~"DOC_COMMENT"]),
                                  ~[mk_ident(cx, sp, ident)]);
        }

        INTERPOLATED(_) => fail ~"quote! with interpolated token",

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
        ELLIPSIS => "ELLIPSIS",
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
        _ => fail
    };
    build::mk_path(cx, sp,
                   ids_ext(cx, ~[name.to_owned()]))
}


fn mk_tt(cx: ext_ctxt, sp: span, tt: &ast::token_tree) -> @ast::expr {
    match *tt {
        ast::tt_tok(sp, tok) =>
        build::mk_call(cx, sp,
                       ids_ext(cx, ~[~"tt_tok"]),
                       ~[mk_span(cx, sp, sp),
                         mk_token(cx, sp, tok)]),

        ast::tt_delim(tts) => {
            let e_tts = tts.map(|tt| mk_tt(cx, sp, tt));
            build::mk_call(cx, sp,
                           ids_ext(cx, ~[~"tt_delim"]),
                           ~[build::mk_uniq_vec_e(cx, sp, e_tts)])
        }

        ast::tt_seq(*) => fail ~"tt_seq in quote!",

        ast::tt_nonterminal(sp, ident) =>
        build::mk_copy(cx, sp, build::mk_path(cx, sp, ~[ident]))
    }
}
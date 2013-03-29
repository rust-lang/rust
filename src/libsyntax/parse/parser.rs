// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast::{Sigil, BorrowedSigil, ManagedSigil, OwnedSigil, RustAbi};
use ast::{CallSugar, NoSugar, DoSugar, ForSugar};
use ast::{TyBareFn, TyClosure};
use ast::{RegionTyParamBound, TraitTyParamBound};
use ast::{provided, public, purity};
use ast::{_mod, add, arg, arm, attribute, bind_by_ref, bind_infer};
use ast::{bind_by_copy, bitand, bitor, bitxor, blk};
use ast::{blk_check_mode, box, by_copy, by_ref};
use ast::{crate, crate_cfg, decl, decl_item};
use ast::{decl_local, default_blk, deref, div, enum_def};
use ast::{expl, expr, expr_, expr_addr_of, expr_match, expr_again};
use ast::{expr_assign, expr_assign_op, expr_binary, expr_block};
use ast::{expr_break, expr_call, expr_cast, expr_copy, expr_do_body};
use ast::{expr_field, expr_fn_block, expr_if, expr_index};
use ast::{expr_lit, expr_log, expr_loop, expr_loop_body, expr_mac};
use ast::{expr_method_call, expr_paren, expr_path, expr_repeat};
use ast::{expr_ret, expr_swap, expr_struct, expr_tup, expr_unary};
use ast::{expr_vec, expr_vstore, expr_vstore_mut_box};
use ast::{expr_vstore_slice, expr_vstore_box};
use ast::{expr_vstore_mut_slice, expr_while, extern_fn, field, fn_decl};
use ast::{expr_vstore_uniq, TyClosure, TyBareFn, Onceness, Once, Many};
use ast::{foreign_item, foreign_item_const, foreign_item_fn, foreign_mod};
use ast::{ident, impure_fn, infer, inherited, item, item_, item_const};
use ast::{item_const, item_enum, item_fn, item_foreign_mod, item_impl};
use ast::{item_mac, item_mod, item_struct, item_trait, item_ty, lit, lit_};
use ast::{lit_bool, lit_float, lit_float_unsuffixed, lit_int};
use ast::{lit_int_unsuffixed, lit_nil, lit_str, lit_uint, local, m_const};
use ast::{m_imm, m_mutbl, mac_, mac_invoc_tt, matcher, match_nonterminal};
use ast::{match_seq, match_tok, method, mode, module_ns, mt, mul, mutability};
use ast::{named_field, neg, node_id, noreturn, not, pat, pat_box, pat_enum};
use ast::{pat_ident, pat_lit, pat_range, pat_region, pat_struct};
use ast::{pat_tup, pat_uniq, pat_wild, private};
use ast::{rem, required};
use ast::{ret_style, return_val, self_ty, shl, shr, stmt, stmt_decl};
use ast::{stmt_expr, stmt_semi, stmt_mac, struct_def, struct_field};
use ast::{struct_immutable, struct_mutable, struct_variant_kind, subtract};
use ast::{sty_box, sty_region, sty_static, sty_uniq, sty_value};
use ast::{token_tree, trait_method, trait_ref, tt_delim, tt_seq, tt_tok};
use ast::{tt_nonterminal, tuple_variant_kind, Ty, ty_, ty_bot, ty_box};
use ast::{ty_field, ty_fixed_length_vec, ty_closure, ty_bare_fn};
use ast::{ty_infer, ty_method};
use ast::{ty_nil, TyParam, TyParamBound, ty_path, ty_ptr, ty_rptr};
use ast::{ty_tup, ty_u32, ty_uniq, ty_vec, type_value_ns, uniq};
use ast::{unnamed_field, unsafe_blk, unsafe_fn, view_item};
use ast::{view_item_, view_item_extern_mod, view_item_use};
use ast::{view_path, view_path_glob, view_path_list, view_path_simple};
use ast::visibility;
use ast;
use ast_util::{ident_to_path, operator_prec};
use ast_util;
use codemap::{span, BytePos, spanned, mk_sp};
use codemap;
use parse::attr::parser_attr;
use parse::classify;
use parse::common::{seq_sep_none, token_to_str};
use parse::common::{seq_sep_trailing_disallowed, seq_sep_trailing_allowed};
use parse::lexer::reader;
use parse::lexer::TokenAndSpan;
use parse::obsolete::{ObsoleteClassTraits};
use parse::obsolete::{ObsoleteLet, ObsoleteFieldTerminator};
use parse::obsolete::{ObsoleteMoveInit, ObsoleteBinaryMove};
use parse::obsolete::{ObsoleteSyntax, ObsoleteLowerCaseKindBounds};
use parse::obsolete::{ObsoleteUnsafeBlock, ObsoleteImplSyntax};
use parse::obsolete::{ObsoleteTraitBoundSeparator, ObsoleteMutOwnedPointer};
use parse::obsolete::{ObsoleteMutVector, ObsoleteTraitImplVisibility};
use parse::obsolete::{ObsoleteRecordType, ObsoleteRecordPattern};
use parse::obsolete::{ObsoletePostFnTySigil};
use parse::obsolete::{ObsoleteBareFnType, ObsoleteNewtypeEnum};
use parse::obsolete::ObsoleteMode;
use parse::obsolete::{ObsoleteLifetimeNotation, ObsoleteConstManagedPointer};
use parse::obsolete::{ObsoletePurity, ObsoleteStaticMethod};
use parse::obsolete::{ObsoleteConstItem, ObsoleteFixedLengthVectorType};
use parse::prec::{as_prec, token_to_binop};
use parse::token::{can_begin_expr, is_ident, is_ident_or_path};
use parse::token::{is_plain_ident, INTERPOLATED, special_idents};
use parse::token;
use parse::{new_sub_parser_from_file, next_node_id, ParseSess};
use opt_vec;
use opt_vec::OptVec;

use core::either::Either;
use core::either;
use core::hashmap::linear::LinearSet;
use core::vec;

#[deriving(Eq)]
enum restriction {
    UNRESTRICTED,
    RESTRICT_STMT_EXPR,
    RESTRICT_NO_CALL_EXPRS,
    RESTRICT_NO_BAR_OP,
    RESTRICT_NO_BAR_OR_DOUBLEBAR_OP,
}

//  So that we can distinguish a class dtor from other class members

enum class_contents { dtor_decl(blk, ~[attribute], codemap::span),
                      members(~[@struct_field]) }

type arg_or_capture_item = Either<arg, ()>;
type item_info = (ident, item_, Option<~[attribute]>);

pub enum item_or_view_item {
    // indicates a failure to parse any kind of item:
    iovi_none,
    iovi_item(@item),
    iovi_foreign_item(@foreign_item),
    iovi_view_item(@view_item)
}

#[deriving(Eq)]
enum view_item_parse_mode {
    VIEW_ITEMS_AND_ITEMS_ALLOWED,
    FOREIGN_ITEMS_ALLOWED,
    IMPORTS_AND_ITEMS_ALLOWED
}

/* The expr situation is not as complex as I thought it would be.
The important thing is to make sure that lookahead doesn't balk
at INTERPOLATED tokens */
macro_rules! maybe_whole_expr (
    ($p:expr) => (
        match *($p).token {
            INTERPOLATED(token::nt_expr(copy e)) => {
                $p.bump();
                return e;
            }
            INTERPOLATED(token::nt_path(copy pt)) => {
                $p.bump();
                return $p.mk_expr(
                    ($p).span.lo,
                    ($p).span.hi,
                    expr_path(pt)
                );
            }
            _ => ()
        }
    )
)

macro_rules! maybe_whole (
    ($p:expr, $constructor:ident) => (
        match *($p).token {
            INTERPOLATED(token::$constructor(copy x)) => {
                $p.bump();
                return x;
            }
            _ => ()
       }
    );
    (deref $p:expr, $constructor:ident) => (
        match *($p).token {
            INTERPOLATED(token::$constructor(copy x)) => {
                $p.bump();
                return copy *x;
            }
            _ => ()
        }
    );
    (Some $p:expr, $constructor:ident) => (
        match *($p).token {
            INTERPOLATED(token::$constructor(copy x)) => {
                $p.bump();
                return Some(x);
            }
            _ => ()
        }
    );
    (iovi $p:expr, $constructor:ident) => (
        match *($p).token {
            INTERPOLATED(token::$constructor(copy x)) => {
                $p.bump();
                return iovi_item(x);
            }
            _ => ()
        }
    );
    (pair_empty $p:expr, $constructor:ident) => (
        match *($p).token {
            INTERPOLATED(token::$constructor(copy x)) => {
                $p.bump();
                return (~[], x);
            }
            _ => ()
        }
    )
)


fn maybe_append(+lhs: ~[attribute], rhs: Option<~[attribute]>)
             -> ~[attribute] {
    match rhs {
        None => lhs,
        Some(ref attrs) => vec::append(lhs, (*attrs))
    }
}


struct ParsedItemsAndViewItems {
    attrs_remaining: ~[attribute],
    view_items: ~[@view_item],
    items: ~[@item],
    foreign_items: ~[@foreign_item]
}

/* ident is handled by common.rs */

pub fn Parser(sess: @mut ParseSess,
              +cfg: ast::crate_cfg,
              +rdr: @reader)
           -> Parser {
    let tok0 = copy rdr.next_token();
    let interner = rdr.interner();

    Parser {
        reader: rdr,
        interner: interner,
        sess: sess,
        cfg: cfg,
        token: @mut copy tok0.tok,
        span: @mut copy tok0.sp,
        last_span: @mut copy tok0.sp,
        buffer: @mut ([copy tok0, .. 4]),
        buffer_start: @mut 0,
        buffer_end: @mut 0,
        tokens_consumed: @mut 0,
        restriction: @mut UNRESTRICTED,
        quote_depth: @mut 0,
        keywords: token::keyword_table(),
        strict_keywords: token::strict_keyword_table(),
        reserved_keywords: token::reserved_keyword_table(),
        obsolete_set: @mut LinearSet::new(),
        mod_path_stack: @mut ~[],
    }
}

pub struct Parser {
    sess: @mut ParseSess,
    cfg: crate_cfg,
    token: @mut token::Token,
    span: @mut span,
    last_span: @mut span,
    buffer: @mut [TokenAndSpan, ..4],
    buffer_start: @mut int,
    buffer_end: @mut int,
    tokens_consumed: @mut uint,
    restriction: @mut restriction,
    quote_depth: @mut uint, // not (yet) related to the quasiquoter
    reader: @reader,
    interner: @token::ident_interner,
    keywords: LinearSet<~str>,
    strict_keywords: LinearSet<~str>,
    reserved_keywords: LinearSet<~str>,
    /// The set of seen errors about obsolete syntax. Used to suppress
    /// extra detail when the same error is seen twice
    obsolete_set: @mut LinearSet<ObsoleteSyntax>,
    /// Used to determine the path to externally loaded source files
    mod_path_stack: @mut ~[~str],

}

#[unsafe_destructor]
impl Drop for Parser {
    /* do not copy the parser; its state is tied to outside state */
    fn finalize(&self) {}
}

pub impl Parser {
    // advance the parser by one token
    fn bump(&self) {
        *self.last_span = copy *self.span;
        let next = if *self.buffer_start == *self.buffer_end {
            self.reader.next_token()
        } else {
            let next = copy self.buffer[*self.buffer_start];
            *self.buffer_start = (*self.buffer_start + 1) & 3;
            next
        };
        *self.token = copy next.tok;
        *self.span = copy next.sp;
        *self.tokens_consumed += 1u;
    }
    // EFFECT: replace the current token and span with the given one
    fn replace_token(&self, +next: token::Token, +lo: BytePos, +hi: BytePos) {
        *self.token = next;
        *self.span = mk_sp(lo, hi);
    }
    fn buffer_length(&self) -> int {
        if *self.buffer_start <= *self.buffer_end {
            return *self.buffer_end - *self.buffer_start;
        }
        return (4 - *self.buffer_start) + *self.buffer_end;
    }
    fn look_ahead(&self, distance: uint) -> token::Token {
        let dist = distance as int;
        while self.buffer_length() < dist {
            self.buffer[*self.buffer_end] = self.reader.next_token();
            *self.buffer_end = (*self.buffer_end + 1) & 3;
        }
        return copy self.buffer[(*self.buffer_start + dist - 1) & 3].tok;
    }
    fn fatal(&self, m: ~str) -> ! {
        self.sess.span_diagnostic.span_fatal(*copy self.span, m)
    }
    fn span_fatal(&self, sp: span, m: ~str) -> ! {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }
    fn span_note(&self, sp: span, m: ~str) {
        self.sess.span_diagnostic.span_note(sp, m)
    }
    fn bug(&self, m: ~str) -> ! {
        self.sess.span_diagnostic.span_bug(*copy self.span, m)
    }
    fn warn(&self, m: ~str) {
        self.sess.span_diagnostic.span_warn(*copy self.span, m)
    }
    fn span_err(&self, sp: span, m: ~str) {
        self.sess.span_diagnostic.span_err(sp, m)
    }
    fn abort_if_errors(&self) {
        self.sess.span_diagnostic.handler().abort_if_errors();
    }
    fn get_id(&self) -> node_id { next_node_id(self.sess) }

    fn id_to_str(&self, id: ident) -> @~str {
        self.sess.interner.get(id)
    }

    fn token_is_closure_keyword(&self, tok: &token::Token) -> bool {
        self.token_is_keyword(&~"pure", tok) ||
            self.token_is_keyword(&~"unsafe", tok) ||
            self.token_is_keyword(&~"once", tok) ||
            self.token_is_keyword(&~"fn", tok)
    }

    fn parse_ty_bare_fn(&self) -> ty_
    {
        /*

        extern "ABI" [pure|unsafe] fn <'lt> (S) -> T
               ^~~~^ ^~~~~~~~~~~~^    ^~~~^ ^~^    ^
                 |     |                |    |     |
                 |     |                |    |   Return type
                 |     |                |  Argument types
                 |     |            Lifetimes
                 |     |
                 |   Purity
                ABI

        */

        let purity = self.parse_purity();
        self.expect_keyword(&~"fn");
        let (decl, lifetimes) = self.parse_ty_fn_decl();
        return ty_bare_fn(@TyBareFn {
            abi: RustAbi,
            purity: purity,
            lifetimes: lifetimes,
            decl: decl
        });
    }

    fn parse_ty_closure(&self,
                        sigil: ast::Sigil,
                        region: Option<@ast::Lifetime>) -> ty_
    {
        /*

        (&|~|@) ['r] [pure|unsafe] [once] fn <'lt> (S) -> T
        ^~~~~~^ ^~~^ ^~~~~~~~~~~~^ ^~~~~^    ^~~~^ ^~^    ^
           |     |     |             |         |    |     |
           |     |     |             |         |    |   Return type
           |     |     |             |         |  Argument types
           |     |     |             |     Lifetimes
           |     |     |          Once-ness (a.k.a., affine)
           |     |   Purity
           | Lifetime bound
        Allocation type

        */

        // At this point, the allocation type and lifetime bound have been
        // parsed.

        let purity = self.parse_purity();
        let onceness = parse_onceness(self);
        self.expect_keyword(&~"fn");

        if self.parse_fn_ty_sigil().is_some() {
            self.obsolete(*self.span,
                          ObsoletePostFnTySigil);
        }

        let (decl, lifetimes) = self.parse_ty_fn_decl();

        return ty_closure(@TyClosure {
            sigil: sigil,
            region: region,
            purity: purity,
            onceness: onceness,
            decl: decl,
            lifetimes: lifetimes,
        });

        fn parse_onceness(self: &Parser) -> Onceness {
            if self.eat_keyword(&~"once") { Once } else { Many }
        }
    }

    fn parse_purity(&self) -> purity {
        if self.eat_keyword(&~"pure") {
            self.obsolete(*self.last_span, ObsoletePurity);
            return impure_fn;
        } else if self.eat_keyword(&~"unsafe") {
            return unsafe_fn;
        } else {
            return impure_fn;
        }
    }

    fn parse_ty_fn_decl(&self) -> (fn_decl, OptVec<ast::Lifetime>) {
        /*

        (fn) <'lt> (S) -> T
             ^~~~^ ^~^    ^
               |    |     |
               |    |   Return type
               |  Argument types
           Lifetimes

        */
        let lifetimes = if self.eat(&token::LT) {
            let lifetimes = self.parse_lifetimes();
            self.expect_gt();
            lifetimes
        } else {
            opt_vec::Empty
        };

        let inputs = self.parse_unspanned_seq(
            &token::LPAREN,
            &token::RPAREN,
            seq_sep_trailing_disallowed(token::COMMA),
            |p| p.parse_arg_general(false)
        );
        let (ret_style, ret_ty) = self.parse_ret_ty();
        let decl = ast::fn_decl {
            inputs: inputs,
            output: ret_ty,
            cf: ret_style
        };
        (decl, lifetimes)
    }

    fn parse_trait_methods(&self) -> ~[trait_method] {
        do self.parse_unspanned_seq(
            &token::LBRACE,
            &token::RBRACE,
            seq_sep_none()
        ) |p| {
            let attrs = p.parse_outer_attributes();
            let lo = p.span.lo;

            let vis = p.parse_visibility();
            let pur = p.parse_fn_purity();
            // NB: at the moment, trait methods are public by default; this
            // could change.
            let ident = p.parse_ident();

            let generics = p.parse_generics();

            let (self_ty, d) = do self.parse_fn_decl_with_self() |p| {
                // This is somewhat dubious; We don't want to allow argument
                // names to be left off if there is a definition...
                either::Left(p.parse_arg_general(false))
            };

            let hi = p.last_span.hi;
            debug!("parse_trait_methods(): trait method signature ends in \
                    `%s`",
                   token_to_str(p.reader, &copy *p.token));
            match *p.token {
              token::SEMI => {
                p.bump();
                debug!("parse_trait_methods(): parsing required method");
                // NB: at the moment, visibility annotations on required
                // methods are ignored; this could change.
                required(ty_method {
                    ident: ident,
                    attrs: attrs,
                    purity: pur,
                    decl: d,
                    generics: generics,
                    self_ty: self_ty,
                    id: p.get_id(),
                    span: mk_sp(lo, hi)
                })
              }
              token::LBRACE => {
                debug!("parse_trait_methods(): parsing provided method");
                let (inner_attrs, body) =
                    p.parse_inner_attrs_and_block(true);
                let attrs = vec::append(attrs, inner_attrs);
                provided(@ast::method {
                    ident: ident,
                    attrs: attrs,
                    generics: generics,
                    self_ty: self_ty,
                    purity: pur,
                    decl: d,
                    body: body,
                    id: p.get_id(),
                    span: mk_sp(lo, hi),
                    self_id: p.get_id(),
                    vis: vis,
                })
              }

              _ => {
                    p.fatal(
                        fmt!(
                            "expected `;` or `}` but found `%s`",
                            token_to_str(p.reader, &copy *p.token)
                        )
                    );
                }
            }
        }
    }


    fn parse_mt(&self) -> mt {
        let mutbl = self.parse_mutability();
        let t = self.parse_ty(false);
        mt { ty: t, mutbl: mutbl }
    }

    fn parse_ty_field(&self) -> ty_field {
        let lo = self.span.lo;
        let mutbl = self.parse_mutability();
        let id = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        spanned(
            lo,
            ty.span.hi,
            ast::ty_field_ {
                ident: id,
                mt: ast::mt { ty: ty, mutbl: mutbl },
            }
        )
    }

    fn parse_ret_ty(&self) -> (ret_style, @Ty) {
        return if self.eat(&token::RARROW) {
            let lo = self.span.lo;
            if self.eat(&token::NOT) {
                (
                    noreturn,
                    @Ty {
                        id: self.get_id(),
                        node: ty_bot,
                        span: mk_sp(lo, self.last_span.hi)
                    }
                )
            } else {
                (return_val, self.parse_ty(false))
            }
        } else {
            let pos = self.span.lo;
            (
                return_val,
                @Ty {
                    id: self.get_id(),
                    node: ty_nil,
                    span: mk_sp(pos, pos),
                }
            )
        }
    }

    // Useless second parameter for compatibility with quasiquote macros.
    // Bleh!
    fn parse_ty(&self, _: bool) -> @Ty {
        maybe_whole!(self, nt_ty);

        let lo = self.span.lo;

        let t = if *self.token == token::LPAREN {
            self.bump();
            if *self.token == token::RPAREN {
                self.bump();
                ty_nil
            } else {
                // (t) is a parenthesized ty
                // (t,) is the type of a tuple with only one field,
                // of type t
                let mut ts = ~[self.parse_ty(false)];
                let mut one_tuple = false;
                while *self.token == token::COMMA {
                    self.bump();
                    if *self.token != token::RPAREN {
                        ts.push(self.parse_ty(false));
                    }
                    else {
                        one_tuple = true;
                    }
                }
                let t = if ts.len() == 1 && !one_tuple {
                    copy ts[0].node
                } else {
                    ty_tup(ts)
                };
                self.expect(&token::RPAREN);
                t
            }
        } else if *self.token == token::AT {
            self.bump();
            self.parse_box_or_uniq_pointee(ManagedSigil, ty_box)
        } else if *self.token == token::TILDE {
            self.bump();
            self.parse_box_or_uniq_pointee(OwnedSigil, ty_uniq)
        } else if *self.token == token::BINOP(token::STAR) {
            self.bump();
            ty_ptr(self.parse_mt())
        } else if *self.token == token::LBRACE {
            let elems = self.parse_unspanned_seq(
                &token::LBRACE,
                &token::RBRACE,
                seq_sep_trailing_allowed(token::COMMA),
                |p| p.parse_ty_field()
            );
            if elems.len() == 0 {
                self.unexpected_last(&token::RBRACE);
            }
            self.obsolete(*self.last_span, ObsoleteRecordType);
            ty_nil
        } else if *self.token == token::LBRACKET {
            self.expect(&token::LBRACKET);
            let mt = self.parse_mt();
            if mt.mutbl == m_mutbl {    // `m_const` too after snapshot
                self.obsolete(*self.last_span, ObsoleteMutVector);
            }

            // Parse the `, ..e` in `[ int, ..e ]`
            // where `e` is a const expression
            let t = match self.maybe_parse_fixed_vstore() {
                None => ty_vec(mt),
                Some(suffix) => ty_fixed_length_vec(mt, suffix)
            };
            self.expect(&token::RBRACKET);
            t
        } else if *self.token == token::BINOP(token::AND) {
            self.bump();
            self.parse_borrowed_pointee()
        } else if self.eat_keyword(&~"extern") {
            self.parse_ty_bare_fn()
        } else if self.token_is_closure_keyword(&copy *self.token) {
            let result = self.parse_ty_closure(ast::BorrowedSigil, None);
            self.obsolete(*self.last_span, ObsoleteBareFnType);
            result
        } else if *self.token == token::MOD_SEP
            || is_ident_or_path(&*self.token) {
            let path = self.parse_path_with_tps(false);
            ty_path(path, self.get_id())
        } else {
            self.fatal(~"expected type");
        };

        let sp = mk_sp(lo, self.last_span.hi);
        @Ty {id: self.get_id(), node: t, span: sp}
    }

    fn parse_box_or_uniq_pointee(
        &self,
        sigil: ast::Sigil,
        ctor: &fn(+v: mt) -> ty_) -> ty_
    {
        // @'foo fn() or @foo/fn() or @fn() are parsed directly as fn types:
        match *self.token {
            token::LIFETIME(*) => {
                let lifetime = @self.parse_lifetime();
                self.bump();
                return self.parse_ty_closure(sigil, Some(lifetime));
            }

            token::IDENT(*) => {
                if self.look_ahead(1u) == token::BINOP(token::SLASH) &&
                    self.token_is_closure_keyword(&self.look_ahead(2u))
                {
                    let lifetime = @self.parse_lifetime();
                    self.obsolete(*self.last_span, ObsoleteLifetimeNotation);
                    return self.parse_ty_closure(sigil, Some(lifetime));
                } else if self.token_is_closure_keyword(&copy *self.token) {
                    return self.parse_ty_closure(sigil, None);
                }
            }
            _ => {}
        }

        // other things are parsed as @ + a type.  Note that constructs like
        // @[] and @str will be resolved during typeck to slices and so forth,
        // rather than boxed ptrs.  But the special casing of str/vec is not
        // reflected in the AST type.
        let mt = self.parse_mt();

        if mt.mutbl != m_imm && sigil == OwnedSigil {
            self.obsolete(*self.last_span, ObsoleteMutOwnedPointer);
        }
        if mt.mutbl == m_const && sigil == ManagedSigil {
            self.obsolete(*self.last_span, ObsoleteConstManagedPointer);
        }

        ctor(mt)
    }

    fn parse_borrowed_pointee(&self) -> ty_ {
        // look for `&'lt` or `&'foo ` and interpret `foo` as the region name:
        let opt_lifetime = self.parse_opt_lifetime();

        if self.token_is_closure_keyword(&copy *self.token) {
            return self.parse_ty_closure(BorrowedSigil, opt_lifetime);
        }

        let mt = self.parse_mt();
        return ty_rptr(opt_lifetime, mt);
    }

    fn parse_arg_mode(&self) -> mode {
        if self.eat(&token::BINOP(token::MINUS)) {
            self.obsolete(*self.span, ObsoleteMode);
            expl(by_copy)
        } else if self.eat(&token::ANDAND) {
            expl(by_ref)
        } else if self.eat(&token::BINOP(token::PLUS)) {
            if self.eat(&token::BINOP(token::PLUS)) {
                // ++ mode is obsolete, but we need a snapshot
                // to stop parsing it.
                expl(by_copy)
            } else {
                expl(by_copy)
            }
        } else {
            infer(self.get_id())
        }
    }

    fn is_named_argument(&self) -> bool {
        let offset = if *self.token == token::BINOP(token::AND) {
            1
        } else if *self.token == token::BINOP(token::MINUS) {
            1
        } else if *self.token == token::ANDAND {
            1
        } else if *self.token == token::BINOP(token::PLUS) {
            if self.look_ahead(1) == token::BINOP(token::PLUS) {
                2
            } else {
                1
            }
        } else { 0 };
        if offset == 0 {
            is_plain_ident(&*self.token)
                && self.look_ahead(1) == token::COLON
        } else {
            is_plain_ident(&self.look_ahead(offset))
                && self.look_ahead(offset + 1) == token::COLON
        }
    }

    // This version of parse arg doesn't necessarily require
    // identifier names.
    fn parse_arg_general(&self, require_name: bool) -> arg {
        let mut m;
        let mut is_mutbl = false;
        let pat = if require_name || self.is_named_argument() {
            m = self.parse_arg_mode();
            is_mutbl = self.eat_keyword(&~"mut");
            let pat = self.parse_pat(false);
            self.expect(&token::COLON);
            pat
        } else {
            m = infer(self.get_id());
            ast_util::ident_to_pat(self.get_id(),
                                   *self.last_span,
                                   special_idents::invalid)
        };

        let t = self.parse_ty(false);

        ast::arg { mode: m, is_mutbl: is_mutbl,
                  ty: t, pat: pat, id: self.get_id() }
    }

    fn parse_arg(&self) -> arg_or_capture_item {
        either::Left(self.parse_arg_general(true))
    }

    fn parse_fn_block_arg(&self) -> arg_or_capture_item {
        let m = self.parse_arg_mode();
        let is_mutbl = self.eat_keyword(&~"mut");
        let pat = self.parse_pat(false);
        let t = if self.eat(&token::COLON) {
            self.parse_ty(false)
        } else {
            @Ty {
                id: self.get_id(),
                node: ty_infer,
                span: mk_sp(self.span.lo, self.span.hi),
            }
        };
        either::Left(ast::arg {
            mode: m,
            is_mutbl: is_mutbl,
            ty: t,
            pat: pat,
            id: self.get_id()
        })
    }

    fn maybe_parse_fixed_vstore(&self) -> Option<@ast::expr> {
        if self.eat(&token::BINOP(token::STAR)) {
            self.obsolete(*self.last_span, ObsoleteFixedLengthVectorType);
            Some(self.parse_expr())
        } else if *self.token == token::COMMA &&
                self.look_ahead(1) == token::DOTDOT {
            self.bump();
            self.bump();
            Some(self.parse_expr())
        } else {
            None
        }
    }

    fn lit_from_token(&self, tok: &token::Token) -> lit_ {
        match *tok {
            token::LIT_INT(i, it) => lit_int(i, it),
            token::LIT_UINT(u, ut) => lit_uint(u, ut),
            token::LIT_INT_UNSUFFIXED(i) => lit_int_unsuffixed(i),
            token::LIT_FLOAT(s, ft) => lit_float(self.id_to_str(s), ft),
            token::LIT_FLOAT_UNSUFFIXED(s) =>
                lit_float_unsuffixed(self.id_to_str(s)),
            token::LIT_STR(s) => lit_str(self.id_to_str(s)),
            token::LPAREN => { self.expect(&token::RPAREN); lit_nil },
            _ => { self.unexpected_last(tok); }
        }
    }

    fn parse_lit(&self) -> lit {
        let lo = self.span.lo;
        let lit = if self.eat_keyword(&~"true") {
            lit_bool(true)
        } else if self.eat_keyword(&~"false") {
            lit_bool(false)
        } else {
            // XXX: This is a really bad copy!
            let tok = copy *self.token;
            self.bump();
            self.lit_from_token(&tok)
        };
        codemap::spanned { node: lit, span: mk_sp(lo, self.last_span.hi) }
    }

    // parse a path that doesn't have type parameters attached
    fn parse_path_without_tps(&self)
        -> @ast::path {
        maybe_whole!(self, nt_path);
        let lo = self.span.lo;
        let global = self.eat(&token::MOD_SEP);
        let mut ids = ~[];
        loop {
            let is_not_last =
                self.look_ahead(2u) != token::LT
                && self.look_ahead(1u) == token::MOD_SEP;

            if is_not_last {
                ids.push(self.parse_ident());
                self.expect(&token::MOD_SEP);
            } else {
                ids.push(self.parse_ident());
                break;
            }
        }
        @ast::path { span: mk_sp(lo, self.last_span.hi),
                     global: global,
                     idents: ids,
                     rp: None,
                     types: ~[] }
    }

    fn parse_path_with_tps(&self, colons: bool) -> @ast::path {
        debug!("parse_path_with_tps(colons=%b)", colons);

        maybe_whole!(self, nt_path);
        let lo = self.span.lo;
        let path = self.parse_path_without_tps();
        if colons && !self.eat(&token::MOD_SEP) {
            return path;
        }

        // Parse the (obsolete) trailing region parameter, if any, which will
        // be written "foo/&x"
        let rp_slash = {
            if *self.token == token::BINOP(token::SLASH)
                && self.look_ahead(1u) == token::BINOP(token::AND)
            {
                self.bump(); self.bump();
                self.obsolete(*self.last_span, ObsoleteLifetimeNotation);
                match *self.token {
                    token::IDENT(sid, _) => {
                        let span = copy self.span;
                        self.bump();
                        Some(@ast::Lifetime {
                            id: self.get_id(),
                            span: *span,
                            ident: sid
                        })
                    }
                    _ => {
                        self.fatal(fmt!("Expected a lifetime name"));
                    }
                }
            } else {
                None
            }
        };

        // Parse any lifetime or type parameters which may appear:
        let (lifetimes, tps) = self.parse_generic_values();
        let hi = self.span.lo;

        let rp = match (&rp_slash, &lifetimes) {
            (&Some(_), _) => rp_slash,
            (&None, v) => {
                if v.len() == 0 {
                    None
                } else if v.len() == 1 {
                    Some(@*v.get(0))
                } else {
                    self.fatal(fmt!("Expected at most one \
                                     lifetime name (for now)"));
                }
            }
        };

        @ast::path { span: mk_sp(lo, hi),
                     rp: rp,
                     types: tps,
                     .. copy *path }
    }

    fn parse_opt_lifetime(&self) -> Option<@ast::Lifetime> {
        /*!
         *
         * Parses 0 or 1 lifetime.
         */

        match *self.token {
            token::LIFETIME(*) => {
                Some(@self.parse_lifetime())
            }

            // Also accept the (obsolete) syntax `foo/`
            token::IDENT(*) => {
                if self.look_ahead(1u) == token::BINOP(token::SLASH) {
                    self.obsolete(*self.last_span, ObsoleteLifetimeNotation);
                    Some(@self.parse_lifetime())
                } else {
                    None
                }
            }

            _ => {
                None
            }
        }
    }

    fn token_is_lifetime(&self, tok: &token::Token) -> bool {
        match *tok {
            token::LIFETIME(_) => true,
            _ => false
        }
    }

    fn parse_lifetime(&self) -> ast::Lifetime {
        /*!
         *
         * Parses a single lifetime.
         */

        match *self.token {
            token::LIFETIME(i) => {
                let span = copy self.span;
                self.bump();
                return ast::Lifetime {
                    id: self.get_id(),
                    span: *span,
                    ident: i
                };
            }

            // Also accept the (obsolete) syntax `foo/`
            token::IDENT(i, _) => {
                let span = copy self.span;
                self.bump();
                self.expect(&token::BINOP(token::SLASH));
                self.obsolete(*self.last_span, ObsoleteLifetimeNotation);
                return ast::Lifetime {
                    id: self.get_id(),
                    span: *span,
                    ident: i
                };
            }

            _ => {
                self.fatal(fmt!("Expected a lifetime name"));
            }
        }
    }

    fn parse_lifetimes(&self) -> OptVec<ast::Lifetime> {
        /*!
         *
         * Parses zero or more comma separated lifetimes.
         * Expects each lifetime to be followed by either
         * a comma or `>`.  Used when parsing type parameter
         * lists, where we expect something like `<'a, 'b, T>`.
         */

        let mut res = opt_vec::Empty;
        loop {
            match *self.token {
                token::LIFETIME(_) => {
                    res.push(self.parse_lifetime());
                }
                _ => {
                    return res;
                }
            }

            match *self.token {
                token::COMMA => { self.bump();}
                token::GT => { return res; }
                token::BINOP(token::SHR) => { return res; }
                _ => {
                    self.fatal(~"expected `,` or `>` after lifetime name");
                }
            }
        }
    }

    fn token_is_mutability(&self, tok: &token::Token) -> bool {
        self.token_is_keyword(&~"mut", tok) ||
        self.token_is_keyword(&~"const", tok)
    }

    fn parse_mutability(&self) -> mutability {
        if self.eat_keyword(&~"mut") {
            m_mutbl
        } else if self.eat_keyword(&~"const") {
            m_const
        } else {
            m_imm
        }
    }

    fn parse_field(&self, sep: token::Token) -> field {
        let lo = self.span.lo;
        let m = self.parse_mutability();
        let i = self.parse_ident();
        self.expect(&sep);
        let e = self.parse_expr();
        spanned(lo, e.span.hi, ast::field_ { mutbl: m, ident: i, expr: e })
    }

    fn mk_expr(&self, +lo: BytePos, +hi: BytePos, +node: expr_) -> @expr {
        @expr {
            id: self.get_id(),
            callee_id: self.get_id(),
            node: node,
            span: mk_sp(lo, hi),
        }
    }

    fn mk_mac_expr(&self, +lo: BytePos, +hi: BytePos, +m: mac_) -> @expr {
        @expr {
            id: self.get_id(),
            callee_id: self.get_id(),
            node: expr_mac(codemap::spanned {node: m, span: mk_sp(lo, hi)}),
            span: mk_sp(lo, hi),
        }
    }

    fn mk_lit_u32(&self, i: u32) -> @expr {
        let span = self.span;
        let lv_lit = @codemap::spanned {
            node: lit_uint(i as u64, ty_u32),
            span: *span
        };

        @expr {
            id: self.get_id(),
            callee_id: self.get_id(),
            node: expr_lit(lv_lit),
            span: *span,
        }
    }

    fn parse_bottom_expr(&self) -> @expr {
        maybe_whole_expr!(self);

        let lo = self.span.lo;
        let mut hi = self.span.hi;

        let mut ex: expr_;

        if *self.token == token::LPAREN {
            self.bump();
            // (e) is parenthesized e
            // (e,) is a tuple with only one field, e
            let mut one_tuple = false;
            if *self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = @spanned(lo, hi, lit_nil);
                return self.mk_expr(lo, hi, expr_lit(lit));
            }
            let mut es = ~[self.parse_expr()];
            while *self.token == token::COMMA {
                self.bump();
                if *self.token != token::RPAREN {
                    es.push(self.parse_expr());
                }
                else {
                    one_tuple = true;
                }
            }
            hi = self.span.hi;
            self.expect(&token::RPAREN);

            return if es.len() == 1 && !one_tuple {
                self.mk_expr(lo, self.span.hi, expr_paren(es[0]))
            }
            else {
                self.mk_expr(lo, hi, expr_tup(es))
            }
        } else if *self.token == token::LBRACE {
            self.bump();
            let blk = self.parse_block_tail(lo, default_blk);
            return self.mk_expr(blk.span.lo, blk.span.hi,
                                 expr_block(blk));
        } else if token::is_bar(&*self.token) {
            return self.parse_lambda_expr();
        } else if self.eat_keyword(&~"if") {
            return self.parse_if_expr();
        } else if self.eat_keyword(&~"for") {
            return self.parse_sugary_call_expr(~"for", ForSugar,
                                               expr_loop_body);
        } else if self.eat_keyword(&~"do") {
            return self.parse_sugary_call_expr(~"do", DoSugar,
                                               expr_do_body);
        } else if self.eat_keyword(&~"while") {
            return self.parse_while_expr();
        } else if self.eat_keyword(&~"loop") {
            return self.parse_loop_expr();
        } else if self.eat_keyword(&~"match") {
            return self.parse_match_expr();
        } else if self.eat_keyword(&~"unsafe") {
            return self.parse_block_expr(lo, unsafe_blk);
        } else if *self.token == token::LBRACKET {
            self.bump();
            let mutbl = self.parse_mutability();
            if mutbl == m_mutbl || mutbl == m_const {
                self.obsolete(*self.last_span, ObsoleteMutVector);
            }

            if *self.token == token::RBRACKET {
                // Empty vector.
                self.bump();
                ex = expr_vec(~[], mutbl);
            } else {
                // Nonempty vector.
                let first_expr = self.parse_expr();
                if *self.token == token::COMMA &&
                        self.look_ahead(1) == token::DOTDOT {
                    // Repeating vector syntax: [ 0, ..512 ]
                    self.bump();
                    self.bump();
                    let count = self.parse_expr();
                    self.expect(&token::RBRACKET);
                    ex = expr_repeat(first_expr, count, mutbl);
                } else if *self.token == token::COMMA {
                    // Vector with two or more elements.
                    self.bump();
                    let remaining_exprs = self.parse_seq_to_end(
                        &token::RBRACKET,
                        seq_sep_trailing_allowed(token::COMMA),
                        |p| p.parse_expr()
                    );
                    ex = expr_vec(~[first_expr] + remaining_exprs, mutbl);
                } else {
                    // Vector with one element.
                    self.expect(&token::RBRACKET);
                    ex = expr_vec(~[first_expr], mutbl);
                }
            }
            hi = self.span.hi;
        } else if self.eat_keyword(&~"__log") {
            self.expect(&token::LPAREN);
            let lvl = self.parse_expr();
            self.expect(&token::COMMA);
            let e = self.parse_expr();
            ex = expr_log(lvl, e);
            hi = self.span.hi;
            self.expect(&token::RPAREN);
        } else if self.eat_keyword(&~"return") {
            if can_begin_expr(&*self.token) {
                let e = self.parse_expr();
                hi = e.span.hi;
                ex = expr_ret(Some(e));
            } else { ex = expr_ret(None); }
        } else if self.eat_keyword(&~"break") {
            if is_ident(&*self.token) {
                ex = expr_break(Some(self.parse_ident()));
            } else {
                ex = expr_break(None);
            }
            hi = self.span.hi;
        } else if self.eat_keyword(&~"copy") {
            let e = self.parse_expr();
            ex = expr_copy(e);
            hi = e.span.hi;
        } else if *self.token == token::MOD_SEP ||
                is_ident(&*self.token) && !self.is_keyword(&~"true") &&
                !self.is_keyword(&~"false") {
            let pth = self.parse_path_with_tps(true);

            // `!`, as an operator, is prefix, so we know this isn't that
            if *self.token == token::NOT {
                self.bump();
                match *self.token {
                    token::LPAREN | token::LBRACE => {}
                    _ => self.fatal(~"expected open delimiter")
                };

                let ket = token::flip_delimiter(&*self.token);
                let tts = self.parse_unspanned_seq(
                    &copy *self.token,
                    &ket,
                    seq_sep_none(),
                    |p| p.parse_token_tree()
                );
                let hi = self.span.hi;

                return self.mk_mac_expr(lo, hi, mac_invoc_tt(pth, tts));
            } else if *self.token == token::LBRACE {
                // This might be a struct literal.
                if self.looking_at_record_literal() {
                    // It's a struct literal.
                    self.bump();
                    let mut fields = ~[];
                    let mut base = None;

                    fields.push(self.parse_field(token::COLON));
                    while *self.token != token::RBRACE {
                        if self.try_parse_obsolete_with() {
                            break;
                        }

                        self.expect(&token::COMMA);

                        if self.eat(&token::DOTDOT) {
                            base = Some(self.parse_expr());
                            break;
                        }

                        if *self.token == token::RBRACE {
                            // Accept an optional trailing comma.
                            break;
                        }
                        fields.push(self.parse_field(token::COLON));
                    }

                    hi = pth.span.hi;
                    self.expect(&token::RBRACE);
                    ex = expr_struct(pth, fields, base);
                    return self.mk_expr(lo, hi, ex);
                }
            }

            hi = pth.span.hi;
            ex = expr_path(pth);
        } else {
            let lit = self.parse_lit();
            hi = lit.span.hi;
            ex = expr_lit(@lit);
        }

        return self.mk_expr(lo, hi, ex);
    }

    fn parse_block_expr(
        &self,
        lo: BytePos,
        blk_mode: blk_check_mode
    ) -> @expr {
        self.expect(&token::LBRACE);
        let blk = self.parse_block_tail(lo, blk_mode);
        return self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk));
    }

    fn parse_dot_or_call_expr(&self) -> @expr {
        let b = self.parse_bottom_expr();
        self.parse_dot_or_call_expr_with(b)
    }

    fn permits_call(&self) -> bool {
        return *self.restriction != RESTRICT_NO_CALL_EXPRS;
    }

    fn parse_dot_or_call_expr_with(&self, e0: @expr) -> @expr {
        let mut e = e0;
        let lo = e.span.lo;
        let mut hi;
        loop {
            // expr.f
            if self.eat(&token::DOT) {
                match *self.token {
                  token::IDENT(i, _) => {
                    hi = self.span.hi;
                    self.bump();
                    let (_, tys) = if self.eat(&token::MOD_SEP) {
                        self.expect(&token::LT);
                        self.parse_generic_values_after_lt()
                    } else {
                        (opt_vec::Empty, ~[])
                    };

                    // expr.f() method call
                    match *self.token {
                        token::LPAREN if self.permits_call() => {
                            let es = self.parse_unspanned_seq(
                                &token::LPAREN,
                                &token::RPAREN,
                                seq_sep_trailing_disallowed(token::COMMA),
                                |p| p.parse_expr()
                            );
                            hi = self.span.hi;

                            let nd = expr_method_call(e, i, tys, es, NoSugar);
                            e = self.mk_expr(lo, hi, nd);
                        }
                        _ => {
                            e = self.mk_expr(lo, hi, expr_field(e, i, tys));
                        }
                    }
                  }
                  _ => self.unexpected()
                }
                loop;
            }
            if self.expr_is_complete(e) { break; }
            match *self.token {
              // expr(...)
              token::LPAREN if self.permits_call() => {
                let es = self.parse_unspanned_seq(
                    &token::LPAREN,
                    &token::RPAREN,
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_expr()
                );
                hi = self.span.hi;

                let nd = expr_call(e, es, NoSugar);
                e = self.mk_expr(lo, hi, nd);
              }

              // expr[...]
              token::LBRACKET => {
                self.bump();
                let ix = self.parse_expr();
                hi = ix.span.hi;
                self.expect(&token::RBRACKET);
                e = self.mk_expr(lo, hi, expr_index(e, ix));
              }

              _ => return e
            }
        }
        return e;
    }

    // parse an optional separator followed by a kleene-style
    // repetition token (+ or *).
    fn parse_sep_and_zerok(&self) -> (Option<token::Token>, bool) {
        if *self.token == token::BINOP(token::STAR)
            || *self.token == token::BINOP(token::PLUS) {
            let zerok = *self.token == token::BINOP(token::STAR);
            self.bump();
            (None, zerok)
        } else {
            let sep = copy *self.token;
            self.bump();
            if *self.token == token::BINOP(token::STAR)
                || *self.token == token::BINOP(token::PLUS) {
                let zerok = *self.token == token::BINOP(token::STAR);
                self.bump();
                (Some(sep), zerok)
            } else {
                self.fatal(~"expected `*` or `+`");
            }
        }
    }

    // parse a single token tree from the input.
    fn parse_token_tree(&self) -> token_tree {
        maybe_whole!(deref self, nt_tt);

        fn parse_non_delim_tt_tok(p: &Parser) -> token_tree {
            maybe_whole!(deref p, nt_tt);
            match *p.token {
              token::RPAREN | token::RBRACE | token::RBRACKET
              => {
                p.fatal(
                    fmt!(
                        "incorrect close delimiter: `%s`",
                        token_to_str(p.reader, &copy *p.token)
                    )
                );
              }
              /* we ought to allow different depths of unquotation */
              token::DOLLAR if *p.quote_depth > 0u => {
                p.bump();
                let sp = *p.span;

                if *p.token == token::LPAREN {
                    let seq = p.parse_seq(
                        &token::LPAREN,
                        &token::RPAREN,
                        seq_sep_none(),
                        |p| p.parse_token_tree()
                    );
                    let (s, z) = p.parse_sep_and_zerok();
                    tt_seq(
                        mk_sp(sp.lo ,p.span.hi),
                        seq.node,
                        s,
                        z
                    )
                } else {
                    tt_nonterminal(sp, p.parse_ident())
                }
              }
              _ => {
                  parse_any_tt_tok(p)
              }
            }
        }

        // turn the next token into a tt_tok:
        fn parse_any_tt_tok(p: &Parser) -> token_tree{
            let res = tt_tok(*p.span, copy *p.token);
            p.bump();
            res
        }

        match *self.token {
            token::EOF => {
                self.fatal(~"file ended in the middle of a macro invocation");
            }
            token::LPAREN | token::LBRACE | token::LBRACKET => {
                // tjc: ??????
                let ket = token::flip_delimiter(&*self.token);
                tt_delim(
                    vec::append(
                        // the open delimiter:
                        ~[parse_any_tt_tok(self)],
                        vec::append(
                            self.parse_seq_to_before_end(
                                &ket,
                                seq_sep_none(),
                                |p| p.parse_token_tree()
                            ),
                            // the close delimiter:
                            ~[parse_any_tt_tok(self)]
                        )
                    )
                )
            }
            _ => parse_non_delim_tt_tok(self)
        }
    }

    fn parse_all_token_trees(&self) -> ~[token_tree] {
        let mut tts = ~[];
        while *self.token != token::EOF {
            tts.push(self.parse_token_tree());
        }
        tts
    }

    fn parse_matchers(&self) -> ~[matcher] {
        // unification of matchers and token_trees would vastly improve
        // the interpolation of matchers
        maybe_whole!(self, nt_matchers);
        let name_idx = @mut 0u;
        match *self.token {
            token::LBRACE | token::LPAREN | token::LBRACKET => {
                self.parse_matcher_subseq(
                    name_idx,
                    &*self.token,
                    // tjc: not sure why we need a copy
                    &token::flip_delimiter(&*self.token)
                )
            }
            _ => self.fatal(~"expected open delimiter")
        }
    }


    // This goofy function is necessary to correctly match parens in matchers.
    // Otherwise, `$( ( )` would be a valid matcher, and `$( () )` would be
    // invalid. It's similar to common::parse_seq.
    fn parse_matcher_subseq(
        &self,
        name_idx: @mut uint,
        bra: &token::Token,
        ket: &token::Token
    ) -> ~[matcher] {
        let mut ret_val = ~[];
        let mut lparens = 0u;

        self.expect(bra);

        while *self.token != *ket || lparens > 0u {
            if *self.token == token::LPAREN { lparens += 1u; }
            if *self.token == token::RPAREN { lparens -= 1u; }
            ret_val.push(self.parse_matcher(name_idx));
        }

        self.bump();

        return ret_val;
    }

    fn parse_matcher(&self, name_idx: @mut uint) -> matcher {
        let lo = self.span.lo;

        let m = if *self.token == token::DOLLAR {
            self.bump();
            if *self.token == token::LPAREN {
                let name_idx_lo = *name_idx;
                let ms = self.parse_matcher_subseq(
                    name_idx,
                    &token::LPAREN,
                    &token::RPAREN
                );
                if ms.len() == 0u {
                    self.fatal(~"repetition body must be nonempty");
                }
                let (sep, zerok) = self.parse_sep_and_zerok();
                match_seq(ms, sep, zerok, name_idx_lo, *name_idx)
            } else {
                let bound_to = self.parse_ident();
                self.expect(&token::COLON);
                let nt_name = self.parse_ident();
                let m = match_nonterminal(bound_to, nt_name, *name_idx);
                *name_idx += 1u;
                m
            }
        } else {
            let m = match_tok(copy *self.token);
            self.bump();
            m
        };

        return spanned(lo, self.span.hi, m);
    }


    fn parse_prefix_expr(&self) -> @expr {
        let lo = self.span.lo;
        let mut hi;

        let mut ex;
        match *self.token {
          token::NOT => {
            self.bump();
            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            self.get_id(); // see ast_util::op_expr_callee_id
            ex = expr_unary(not, e);
          }
          token::BINOP(b) => {
            match b {
              token::MINUS => {
                self.bump();
                let e = self.parse_prefix_expr();
                hi = e.span.hi;
                self.get_id(); // see ast_util::op_expr_callee_id
                ex = expr_unary(neg, e);
              }
              token::STAR => {
                self.bump();
                let e = self.parse_prefix_expr();
                hi = e.span.hi;
                ex = expr_unary(deref, e);
              }
              token::AND => {
                self.bump();
                let _lt = self.parse_opt_lifetime();
                let m = self.parse_mutability();
                let e = self.parse_prefix_expr();
                hi = e.span.hi;
                // HACK: turn &[...] into a &-evec
                ex = match e.node {
                  expr_vec(*) | expr_lit(@codemap::spanned {
                    node: lit_str(_), span: _
                  })
                  if m == m_imm => {
                    expr_vstore(e, expr_vstore_slice)
                  }
                  expr_vec(*) if m == m_mutbl => {
                    expr_vstore(e, expr_vstore_mut_slice)
                  }
                  _ => expr_addr_of(m, e)
                };
              }
              _ => return self.parse_dot_or_call_expr()
            }
          }
          token::AT => {
            self.bump();
            let m = self.parse_mutability();
            if m == m_const {
                self.obsolete(*self.last_span, ObsoleteConstManagedPointer);
            }

            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            // HACK: turn @[...] into a @-evec
            ex = match e.node {
              expr_vec(*) | expr_repeat(*) if m == m_mutbl =>
                expr_vstore(e, expr_vstore_mut_box),
              expr_vec(*) |
              expr_lit(@codemap::spanned { node: lit_str(_), span: _}) |
              expr_repeat(*) if m == m_imm => expr_vstore(e, expr_vstore_box),
              _ => expr_unary(box(m), e)
            };
          }
          token::TILDE => {
            self.bump();
            let m = self.parse_mutability();
            if m != m_imm {
                self.obsolete(*self.last_span, ObsoleteMutOwnedPointer);
            }

            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            // HACK: turn ~[...] into a ~-evec
            ex = match e.node {
              expr_vec(*) |
              expr_lit(@codemap::spanned { node: lit_str(_), span: _}) |
              expr_repeat(*)
              if m == m_imm => expr_vstore(e, expr_vstore_uniq),
              _ => expr_unary(uniq(m), e)
            };
          }
          _ => return self.parse_dot_or_call_expr()
        }
        return self.mk_expr(lo, hi, ex);
    }

    // parse an expression of binops
    fn parse_binops(&self) -> @expr {
        self.parse_more_binops(self.parse_prefix_expr(), 0)
    }

    // parse an expression of binops of at least min_prec precedence
    fn parse_more_binops(&self, lhs: @expr, min_prec: uint) ->
        @expr {
        if self.expr_is_complete(lhs) { return lhs; }
        let peeked = copy *self.token;
        if peeked == token::BINOP(token::OR) &&
            (*self.restriction == RESTRICT_NO_BAR_OP ||
             *self.restriction == RESTRICT_NO_BAR_OR_DOUBLEBAR_OP) {
            lhs
        } else if peeked == token::OROR &&
            *self.restriction == RESTRICT_NO_BAR_OR_DOUBLEBAR_OP {
            lhs
        } else {
            let cur_opt = token_to_binop(peeked);
            match cur_opt {
                Some(cur_op) => {
                    let cur_prec = operator_prec(cur_op);
                    if cur_prec > min_prec {
                        self.bump();
                        let expr = self.parse_prefix_expr();
                        let rhs = self.parse_more_binops(expr, cur_prec);
                        self.get_id(); // see ast_util::op_expr_callee_id
                        let bin = self.mk_expr(lhs.span.lo, rhs.span.hi,
                                               expr_binary(cur_op, lhs, rhs));
                        self.parse_more_binops(bin, min_prec)
                    } else {
                        lhs
                    }
                }
                None => {
                    if as_prec > min_prec && self.eat_keyword(&~"as") {
                        let rhs = self.parse_ty(true);
                        let _as = self.mk_expr(lhs.span.lo,
                                               rhs.span.hi,
                                               expr_cast(lhs, rhs));
                        self.parse_more_binops(_as, min_prec)
                    } else {
                        lhs
                    }
                }
            }
        }
    }

    // parse an assignment expression....
    // actually, this seems to be the main entry point for
    // parsing an arbitrary expression.
    fn parse_assign_expr(&self) -> @expr {
        let lo = self.span.lo;
        let lhs = self.parse_binops();
        match *self.token {
            token::EQ => {
                self.bump();
                let rhs = self.parse_expr();
                self.mk_expr(lo, rhs.span.hi, expr_assign(lhs, rhs))
          }
          token::BINOPEQ(op) => {
              self.bump();
              let rhs = self.parse_expr();
              let mut aop;
              match op {
                  token::PLUS => aop = add,
                  token::MINUS => aop = subtract,
                  token::STAR => aop = mul,
                  token::SLASH => aop = div,
                  token::PERCENT => aop = rem,
                  token::CARET => aop = bitxor,
                  token::AND => aop = bitand,
                  token::OR => aop = bitor,
                  token::SHL => aop = shl,
                  token::SHR => aop = shr
              }
              self.get_id(); // see ast_util::op_expr_callee_id
              self.mk_expr(lo, rhs.span.hi,
                           expr_assign_op(aop, lhs, rhs))
          }
          token::LARROW => {
              self.obsolete(*self.span, ObsoleteBinaryMove);
              // Bogus value (but it's an error)
              self.bump(); // <-
              self.bump(); // rhs
              self.bump(); // ;
              self.mk_expr(lo, self.span.hi,
                           expr_break(None))
          }
          token::DARROW => {
            self.bump();
            let rhs = self.parse_expr();
            self.mk_expr(lo, rhs.span.hi, expr_swap(lhs, rhs))
          }
          _ => {
              lhs
          }
        }
    }

    fn parse_if_expr(&self) -> @expr {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let thn = self.parse_block();
        let mut els: Option<@expr> = None;
        let mut hi = thn.span.hi;
        if self.eat_keyword(&~"else") {
            let elexpr = self.parse_else_expr();
            els = Some(elexpr);
            hi = elexpr.span.hi;
        }
        self.mk_expr(lo, hi, expr_if(cond, thn, els))
    }

    // `|args| { ... }` like in `do` expressions
    fn parse_lambda_block_expr(&self) -> @expr {
        self.parse_lambda_expr_(
            || {
                match *self.token {
                  token::BINOP(token::OR) | token::OROR => {
                    self.parse_fn_block_decl()
                  }
                  _ => {
                    // No argument list - `do foo {`
                      ast::fn_decl {
                          inputs: ~[],
                          output: @Ty {
                              id: self.get_id(),
                              node: ty_infer,
                              span: *self.span
                          },
                          cf: return_val
                      }
                  }
                }
            },
            || {
                let blk = self.parse_block();
                self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk))
            })
    }

    // `|args| expr`
    fn parse_lambda_expr(&self) -> @expr {
        self.parse_lambda_expr_(|| self.parse_fn_block_decl(),
                                || self.parse_expr())
    }

    fn parse_lambda_expr_(
        &self,
        parse_decl: &fn() -> fn_decl,
        parse_body: &fn() -> @expr
    ) -> @expr {
        let lo = self.last_span.lo;
        let decl = parse_decl();
        let body = parse_body();
        let fakeblock = ast::blk_ {
            view_items: ~[],
            stmts: ~[],
            expr: Some(body),
            id: self.get_id(),
            rules: default_blk,
        };
        let fakeblock = spanned(body.span.lo, body.span.hi,
                                fakeblock);
        return self.mk_expr(lo, body.span.hi,
                            expr_fn_block(decl, fakeblock));
    }

    fn parse_else_expr(&self) -> @expr {
        if self.eat_keyword(&~"if") {
            return self.parse_if_expr();
        } else {
            let blk = self.parse_block();
            return self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk));
        }
    }

    fn parse_sugary_call_expr(&self, keyword: ~str,
                              sugar: CallSugar,
                              ctor: &fn(+v: @expr) -> expr_) -> @expr {
        let lo = self.last_span;
        // Parse the callee `foo` in
        //    for foo || {
        //    for foo.bar || {
        // etc, or the portion of the call expression before the lambda in
        //    for foo() || {
        // or
        //    for foo.bar(a) || {
        // Turn on the restriction to stop at | or || so we can parse
        // them as the lambda arguments
        let e = self.parse_expr_res(RESTRICT_NO_BAR_OR_DOUBLEBAR_OP);
        match e.node {
            expr_call(f, args, NoSugar) => {
                let block = self.parse_lambda_block_expr();
                let last_arg = self.mk_expr(block.span.lo, block.span.hi,
                                            ctor(block));
                let args = vec::append(args, ~[last_arg]);
                self.mk_expr(lo.lo, block.span.hi, expr_call(f, args, sugar))
            }
            expr_method_call(f, i, tps, args, NoSugar) => {
                let block = self.parse_lambda_block_expr();
                let last_arg = self.mk_expr(block.span.lo, block.span.hi,
                                            ctor(block));
                let args = vec::append(args, ~[last_arg]);
                self.mk_expr(lo.lo, block.span.hi,
                             expr_method_call(f, i, tps, args, sugar))
            }
            expr_field(f, i, tps) => {
                let block = self.parse_lambda_block_expr();
                let last_arg = self.mk_expr(block.span.lo, block.span.hi,
                                            ctor(block));
                self.mk_expr(lo.lo, block.span.hi,
                             expr_method_call(f, i, tps, ~[last_arg], sugar))
            }
            expr_path(*) | expr_call(*) | expr_method_call(*) |
                expr_paren(*) => {
                let block = self.parse_lambda_block_expr();
                let last_arg = self.mk_expr(block.span.lo, block.span.hi,
                                            ctor(block));
                self.mk_expr(lo.lo, last_arg.span.hi,
                             expr_call(e, ~[last_arg], sugar))
            }
            _ => {
                // There may be other types of expressions that can
                // represent the callee in `for` and `do` expressions
                // but they aren't represented by tests
                debug!("sugary call on %?", e.node);
                self.span_fatal(
                    *lo,
                    fmt!("`%s` must be followed by a block call", keyword));
            }
        }
    }

    fn parse_while_expr(&self) -> @expr {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let body = self.parse_block_no_value();
        let mut hi = body.span.hi;
        return self.mk_expr(lo, hi, expr_while(cond, body));
    }

    fn parse_loop_expr(&self) -> @expr {
        // loop headers look like 'loop {' or 'loop unsafe {'
        let is_loop_header =
            *self.token == token::LBRACE
            || (is_ident(&*self.token)
                && self.look_ahead(1) == token::LBRACE);
        // labeled loop headers look like 'loop foo: {'
        let is_labeled_loop_header =
            is_ident(&*self.token)
            && !self.is_any_keyword(&copy *self.token)
            && self.look_ahead(1) == token::COLON;

        if is_loop_header || is_labeled_loop_header {
            // This is a loop body
            let opt_ident;
            if is_labeled_loop_header {
                opt_ident = Some(self.parse_ident());
                self.expect(&token::COLON);
            } else {
                opt_ident = None;
            }

            let lo = self.last_span.lo;
            let body = self.parse_block_no_value();
            let mut hi = body.span.hi;
            return self.mk_expr(lo, hi, expr_loop(body, opt_ident));
        } else {
            // This is a 'continue' expression
            let lo = self.span.lo;
            let ex = if is_ident(&*self.token) {
                expr_again(Some(self.parse_ident()))
            } else {
                expr_again(None)
            };
            let hi = self.span.hi;
            return self.mk_expr(lo, hi, ex);
        }
    }

    // For distingishing between record literals and blocks
    fn looking_at_record_literal(&self) -> bool {
        let lookahead = self.look_ahead(1);
        *self.token == token::LBRACE &&
            (self.token_is_keyword(&~"mut", &lookahead) ||
             (is_plain_ident(&lookahead) &&
              self.look_ahead(2) == token::COLON))
    }

    fn parse_match_expr(&self) -> @expr {
        let lo = self.last_span.lo;
        let discriminant = self.parse_expr();
        self.expect(&token::LBRACE);
        let mut arms: ~[arm] = ~[];
        while *self.token != token::RBRACE {
            let pats = self.parse_pats();
            let mut guard = None;
            if self.eat_keyword(&~"if") { guard = Some(self.parse_expr()); }
            self.expect(&token::FAT_ARROW);
            let expr = self.parse_expr_res(RESTRICT_STMT_EXPR);

            let require_comma =
                !classify::expr_is_simple_block(expr)
                && *self.token != token::RBRACE;

            if require_comma {
                self.expect(&token::COMMA);
            } else {
                self.eat(&token::COMMA);
            }

            let blk = codemap::spanned {
                node: ast::blk_ {
                    view_items: ~[],
                    stmts: ~[],
                    expr: Some(expr),
                    id: self.get_id(),
                    rules: default_blk,
                },
                span: expr.span,
            };

            arms.push(ast::arm { pats: pats, guard: guard, body: blk });
        }
        let mut hi = self.span.hi;
        self.bump();
        return self.mk_expr(lo, hi, expr_match(discriminant, arms));
    }

    // parse an expression
    fn parse_expr(&self) -> @expr {
        return self.parse_expr_res(UNRESTRICTED);
    }

    // parse an expression, subject to the given restriction
    fn parse_expr_res(&self, r: restriction) -> @expr {
        let old = *self.restriction;
        *self.restriction = r;
        let e = self.parse_assign_expr();
        *self.restriction = old;
        return e;
    }

    fn parse_initializer(&self) -> Option<@expr> {
        match *self.token {
          token::EQ => {
            self.bump();
            return Some(self.parse_expr());
          }
          token::LARROW => {
              self.obsolete(*self.span, ObsoleteMoveInit);
              self.bump();
              self.bump();
              return None;
          }
          _ => {
            return None;
          }
        }
    }

    fn parse_pats(&self) -> ~[@pat] {
        let mut pats = ~[];
        loop {
            pats.push(self.parse_pat(true));
            if *self.token == token::BINOP(token::OR) { self.bump(); }
            else { return pats; }
        };
    }

    fn parse_pat_vec_elements(
        &self,
        refutable: bool
    ) -> (~[@pat], Option<@pat>, ~[@pat]) {
        let mut before = ~[];
        let mut slice = None;
        let mut after = ~[];
        let mut first = true;
        let mut before_slice = true;

        while *self.token != token::RBRACKET {
            if first { first = false; }
            else { self.expect(&token::COMMA); }

            let mut is_slice = false;
            if before_slice {
                if *self.token == token::DOTDOT {
                    self.bump();
                    is_slice = true;
                    before_slice = false;
                }
            }

            let subpat = self.parse_pat(refutable);
            if is_slice {
                match subpat {
                    @ast::pat { node: pat_wild, _ } => (),
                    @ast::pat { node: pat_ident(_, _, _), _ } => (),
                    @ast::pat { span, _ } => self.span_fatal(
                        span, ~"expected an identifier or `_`"
                    )
                }
                slice = Some(subpat);
            } else {
                if before_slice {
                    before.push(subpat);
                } else {
                    after.push(subpat);
                }
            }
        }

        (before, slice, after)
    }

    fn parse_pat_fields(&self, refutable: bool) -> (~[ast::field_pat], bool) {
        let mut fields = ~[];
        let mut etc = false;
        let mut first = true;
        while *self.token != token::RBRACE {
            if first { first = false; }
            else { self.expect(&token::COMMA); }

            if *self.token == token::UNDERSCORE {
                self.bump();
                if *self.token != token::RBRACE {
                    self.fatal(
                        fmt!(
                            "expected `}`, found `%s`",
                            token_to_str(self.reader, &copy *self.token)
                        )
                    );
                }
                etc = true;
                break;
            }

            let lo1 = self.last_span.lo;
            let fieldname = self.parse_ident();
            let hi1 = self.last_span.lo;
            let fieldpath = ast_util::ident_to_path(mk_sp(lo1, hi1),
                                                    fieldname);
            let mut subpat;
            if *self.token == token::COLON {
                self.bump();
                subpat = self.parse_pat(refutable);
            } else {
                subpat = @ast::pat {
                    id: self.get_id(),
                    node: pat_ident(bind_infer, fieldpath, None),
                    span: *self.last_span
                };
            }
            fields.push(ast::field_pat { ident: fieldname, pat: subpat });
        }
        return (fields, etc);
    }

    fn parse_pat(&self, refutable: bool) -> @pat {
        maybe_whole!(self, nt_pat);

        let lo = self.span.lo;
        let mut hi = self.span.hi;
        let mut pat;
        match *self.token {
          token::UNDERSCORE => { self.bump(); pat = pat_wild; }
          token::AT => {
            self.bump();
            let sub = self.parse_pat(refutable);
            hi = sub.span.hi;
            // HACK: parse @"..." as a literal of a vstore @str
            pat = match sub.node {
              pat_lit(e@@expr {
                node: expr_lit(@codemap::spanned {
                    node: lit_str(_),
                    span: _}), _
              }) => {
                let vst = @expr {
                    id: self.get_id(),
                    callee_id: self.get_id(),
                    node: expr_vstore(e, expr_vstore_box),
                    span: mk_sp(lo, hi),
                };
                pat_lit(vst)
              }
              _ => pat_box(sub)
            };
          }
          token::TILDE => {
            self.bump();
            let sub = self.parse_pat(refutable);
            hi = sub.span.hi;
            // HACK: parse ~"..." as a literal of a vstore ~str
            pat = match sub.node {
              pat_lit(e@@expr {
                node: expr_lit(@codemap::spanned {
                    node: lit_str(_),
                    span: _}), _
              }) => {
                let vst = @expr {
                    id: self.get_id(),
                    callee_id: self.get_id(),
                    node: expr_vstore(e, expr_vstore_uniq),
                    span: mk_sp(lo, hi),
                };
                pat_lit(vst)
              }
              _ => pat_uniq(sub)
            };
          }
          token::BINOP(token::AND) => {
              let lo = self.span.lo;
              self.bump();
              let sub = self.parse_pat(refutable);
              hi = sub.span.hi;
              // HACK: parse &"..." as a literal of a borrowed str
              pat = match sub.node {
                  pat_lit(e@@expr {
                      node: expr_lit(@codemap::spanned {
                            node: lit_str(_), span: _}), _
                  }) => {
                      let vst = @expr {
                          id: self.get_id(),
                          callee_id: self.get_id(),
                          node: expr_vstore(e, expr_vstore_slice),
                          span: mk_sp(lo, hi)
                      };
                      pat_lit(vst)
                  }
              _ => pat_region(sub)
              };
          }
          token::LBRACE => {
            self.bump();
            let (_, _) = self.parse_pat_fields(refutable);
            hi = self.span.hi;
            self.bump();
            self.obsolete(*self.span, ObsoleteRecordPattern);
            pat = pat_wild;
          }
          token::LPAREN => {
            self.bump();
            if *self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = @codemap::spanned {
                    node: lit_nil,
                    span: mk_sp(lo, hi)};
                let expr = self.mk_expr(lo, hi, expr_lit(lit));
                pat = pat_lit(expr);
            } else {
                let mut fields = ~[self.parse_pat(refutable)];
                if self.look_ahead(1) != token::RPAREN {
                    while *self.token == token::COMMA {
                        self.bump();
                        fields.push(self.parse_pat(refutable));
                    }
                }
                if fields.len() == 1 { self.expect(&token::COMMA); }
                hi = self.span.hi;
                self.expect(&token::RPAREN);
                pat = pat_tup(fields);
            }
          }
          token::LBRACKET => {
            self.bump();
            let (before, slice, after) =
                self.parse_pat_vec_elements(refutable);
            hi = self.span.hi;
            self.expect(&token::RBRACKET);
            pat = ast::pat_vec(before, slice, after);
          }
          tok => {
            if !is_ident_or_path(&tok)
                || self.is_keyword(&~"true")
                || self.is_keyword(&~"false")
            {
                let val = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                if self.eat(&token::DOTDOT) {
                    let end = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                    pat = pat_range(val, end);
                } else {
                    pat = pat_lit(val);
                }
            } else if self.eat_keyword(&~"ref") {
                let mutbl = self.parse_mutability();
                pat = self.parse_pat_ident(refutable, bind_by_ref(mutbl));
            } else if self.eat_keyword(&~"copy") {
                pat = self.parse_pat_ident(refutable, bind_by_copy);
            } else {
                // XXX---refutable match bindings should work same as let
                let binding_mode =
                    if refutable {bind_infer} else {bind_by_copy};

                let cannot_be_enum_or_struct;
                match self.look_ahead(1) {
                    token::LPAREN | token::LBRACKET | token::LT |
                    token::LBRACE | token::MOD_SEP =>
                        cannot_be_enum_or_struct = false,
                    _ =>
                        cannot_be_enum_or_struct = true
                }

                if is_plain_ident(&*self.token) && cannot_be_enum_or_struct {
                    let name = self.parse_path_without_tps();
                    let sub;
                    if self.eat(&token::AT) {
                        sub = Some(self.parse_pat(refutable));
                    } else {
                        sub = None;
                    };
                    pat = pat_ident(binding_mode, name, sub);
                } else {
                    let enum_path = self.parse_path_with_tps(true);
                    match *self.token {
                        token::LBRACE => {
                            self.bump();
                            let (fields, etc) =
                                self.parse_pat_fields(refutable);
                            self.bump();
                            pat = pat_struct(enum_path, fields, etc);
                        }
                        _ => {
                            let mut args: ~[@pat] = ~[];
                            let mut star_pat = false;
                            match *self.token {
                              token::LPAREN => match self.look_ahead(1u) {
                                token::BINOP(token::STAR) => {
                                    // This is a "top constructor only" pat
                                      self.bump(); self.bump();
                                      star_pat = true;
                                      self.expect(&token::RPAREN);
                                  }
                                _ => {
                                    args = self.parse_unspanned_seq(
                                        &token::LPAREN,
                                        &token::RPAREN,
                                        seq_sep_trailing_disallowed(
                                            token::COMMA
                                        ),
                                        |p| p.parse_pat(refutable)
                                    );
                                  }
                              },
                              _ => ()
                            }
                            // at this point, we're not sure whether it's a
                            // enum or a bind
                            if star_pat {
                                pat = pat_enum(enum_path, None);
                            }
                            else if vec::is_empty(args) &&
                                vec::len(enum_path.idents) == 1u {
                                pat = pat_ident(binding_mode,
                                                enum_path,
                                                None);
                            }
                            else {
                                pat = pat_enum(enum_path, Some(args));
                            }
                        }
                    }
                }
            }
            hi = self.span.hi;
          }
        }
        @ast::pat { id: self.get_id(), node: pat, span: mk_sp(lo, hi) }
    }

    fn parse_pat_ident(&self, refutable: bool,
                       binding_mode: ast::binding_mode) -> ast::pat_ {
        if !is_plain_ident(&*self.token) {
            self.span_fatal(
                *self.last_span,
                ~"expected identifier, found path");
        }
        let name = self.parse_path_without_tps();
        let sub = if self.eat(&token::AT) {
            Some(self.parse_pat(refutable))
        } else { None };

        // just to be friendly, if they write something like
        //   ref Some(i)
        // we end up here with ( as the current token.  This shortly
        // leads to a parse error.  Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to parse_enum_variant()
        if *self.token == token::LPAREN {
            self.span_fatal(
                *self.last_span,
                ~"expected identifier, found enum pattern");
        }

        pat_ident(binding_mode, name, sub)
    }

    fn parse_local(&self, is_mutbl: bool,
                   allow_init: bool) -> @local {
        let lo = self.span.lo;
        let pat = self.parse_pat(false);
        let mut ty = @Ty {
            id: self.get_id(),
            node: ty_infer,
            span: mk_sp(lo, lo),
        };
        if self.eat(&token::COLON) { ty = self.parse_ty(false); }
        let init = if allow_init { self.parse_initializer() } else { None };
        @spanned(
            lo,
            self.last_span.hi,
            ast::local_ {
                is_mutbl: is_mutbl,
                ty: ty,
                pat: pat,
                init: init,
                id: self.get_id(),
            }
        )
    }

    fn parse_let(&self) -> @decl {
        let is_mutbl = self.eat_keyword(&~"mut");
        let lo = self.span.lo;
        let mut locals = ~[self.parse_local(is_mutbl, true)];
        while self.eat(&token::COMMA) {
            locals.push(self.parse_local(is_mutbl, true));
        }
        return @spanned(lo, self.last_span.hi, decl_local(locals));
    }

    /* assumes "let" token has already been consumed */
    fn parse_instance_var(&self, pr: visibility) -> @struct_field {
        let mut is_mutbl = struct_immutable;
        let lo = self.span.lo;
        if self.eat_keyword(&~"mut") {
            is_mutbl = struct_mutable;
        }
        if !is_plain_ident(&*self.token) {
            self.fatal(~"expected ident");
        }
        let name = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        @spanned(lo, self.last_span.hi, ast::struct_field_ {
            kind: named_field(name, is_mutbl, pr),
            id: self.get_id(),
            ty: ty
        })
    }

    fn parse_stmt(&self, +first_item_attrs: ~[attribute]) -> @stmt {
        maybe_whole!(self, nt_stmt);

        fn check_expected_item(p: &Parser, current_attrs: &[attribute]) {
            // If we have attributes then we should have an item
            if !current_attrs.is_empty() {
                p.fatal(~"expected item after attrs");
            }
        }

        let lo = self.span.lo;
        if self.is_keyword(&~"let") {
            check_expected_item(self, first_item_attrs);
            self.expect_keyword(&~"let");
            let decl = self.parse_let();
            return @spanned(lo, decl.span.hi, stmt_decl(decl, self.get_id()));
        } else if is_ident(&*self.token)
            && !self.is_any_keyword(&copy *self.token)
            && self.look_ahead(1) == token::NOT {

            check_expected_item(self, first_item_attrs);

            // Potential trouble: if we allow macros with paths instead of
            // idents, we'd need to look ahead past the whole path here...
            let pth = self.parse_path_without_tps();
            self.bump();

            let id = if *self.token == token::LPAREN {
                token::special_idents::invalid // no special identifier
            } else {
                self.parse_ident()
            };

            let tts = self.parse_unspanned_seq(
                &token::LPAREN,
                &token::RPAREN,
                seq_sep_none(),
                |p| p.parse_token_tree()
            );
            let hi = self.span.hi;

            if id == token::special_idents::invalid {
                return @spanned(lo, hi, stmt_mac(
                    spanned(lo, hi, mac_invoc_tt(pth, tts)), false));
            } else {
                // if it has a special ident, it's definitely an item
                return @spanned(lo, hi, stmt_decl(
                    @spanned(lo, hi, decl_item(
                        self.mk_item(
                            lo, hi, id /*id is good here*/,
                            item_mac(spanned(lo, hi, mac_invoc_tt(pth, tts))),
                            inherited, ~[/*no attrs*/]))),
                    self.get_id()));
            }

        } else {
            let item_attrs = vec::append(first_item_attrs,
                                         self.parse_outer_attributes());

            match self.parse_item_or_view_item(/*bad*/ copy item_attrs,
                                               true, false, false) {
              iovi_item(i) => {
                let mut hi = i.span.hi;
                let decl = @spanned(lo, hi, decl_item(i));
                return @spanned(lo, hi, stmt_decl(decl, self.get_id()));
              }
              iovi_view_item(vi) => {
                self.span_fatal(vi.span, ~"view items must be declared at \
                                           the top of the block");
              }
              iovi_foreign_item(_) => {
                  self.fatal(~"foreign items are not allowed here");
              }
              iovi_none() => { /* fallthrough */ }
            }

            check_expected_item(self, item_attrs);

            // Remainder are line-expr stmts.
            let e = self.parse_expr_res(RESTRICT_STMT_EXPR);
            return @spanned(lo, e.span.hi, stmt_expr(e, self.get_id()));
        }
    }

    fn expr_is_complete(&self, e: @expr) -> bool {
        return *self.restriction == RESTRICT_STMT_EXPR &&
            !classify::expr_requires_semi_to_be_stmt(e);
    }

    fn parse_block(&self) -> blk {
        let (attrs, blk) = self.parse_inner_attrs_and_block(false);
        fail_unless!(vec::is_empty(attrs));
        return blk;
    }

    fn parse_inner_attrs_and_block(&self, parse_attrs: bool)
        -> (~[attribute], blk) {

        maybe_whole!(pair_empty self, nt_block);

        fn maybe_parse_inner_attrs_and_next(p: &Parser, parse_attrs: bool) ->
            (~[attribute], ~[attribute]) {
            if parse_attrs {
                p.parse_inner_attrs_and_next()
            } else {
                (~[], ~[])
            }
        }

        let lo = self.span.lo;
        if self.eat_keyword(&~"unsafe") {
            self.obsolete(copy *self.span, ObsoleteUnsafeBlock);
        }
        self.expect(&token::LBRACE);
        let (inner, next) =
            maybe_parse_inner_attrs_and_next(self, parse_attrs);

        (inner, self.parse_block_tail_(lo, default_blk, next))
    }

    fn parse_block_no_value(&self) -> blk {
        // We parse blocks that cannot have a value the same as any other
        // block; the type checker will make sure that the tail expression (if
        // any) has unit type.
        return self.parse_block();
    }

    // Precondition: already parsed the '{' or '#{'
    // I guess that also means "already parsed the 'impure'" if
    // necessary, and this should take a qualifier.
    // some blocks start with "#{"...
    fn parse_block_tail(&self, lo: BytePos, s: blk_check_mode) -> blk {
        self.parse_block_tail_(lo, s, ~[])
    }

    fn parse_block_tail_(&self, lo: BytePos, s: blk_check_mode,
                         +first_item_attrs: ~[attribute]) -> blk {
        let mut stmts = ~[];
        let mut expr = None;

        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: items,
            _
        } = self.parse_items_and_view_items(first_item_attrs,
                                            IMPORTS_AND_ITEMS_ALLOWED, false);

        for items.each |item| {
            let decl = @spanned(item.span.lo, item.span.hi, decl_item(*item));
            stmts.push(@spanned(item.span.lo, item.span.hi,
                                stmt_decl(decl, self.get_id())));
        }

        let mut initial_attrs = attrs_remaining;

        if *self.token == token::RBRACE && !vec::is_empty(initial_attrs) {
            self.fatal(~"expected item");
        }

        while *self.token != token::RBRACE {
            match *self.token {
                token::SEMI => {
                    self.bump(); // empty
                }
                _ => {
                    let stmt = self.parse_stmt(initial_attrs);
                    initial_attrs = ~[];
                    match stmt.node {
                        stmt_expr(e, stmt_id) => {
                            // Expression without semicolon
                            match *self.token {
                                token::SEMI => {
                                    self.bump();
                                    stmts.push(@codemap::spanned {
                                        node: stmt_semi(e, stmt_id),
                                        .. copy *stmt});
                                }
                                token::RBRACE => {
                                    expr = Some(e);
                                }
                                copy t => {
                                    if classify::stmt_ends_with_semi(*stmt) {
                                        self.fatal(
                                            fmt!(
                                                "expected `;` or `}` after \
                                                expression but found `%s`",
                                                token_to_str(self.reader, &t)
                                            )
                                        );
                                    }
                                    stmts.push(stmt);
                                }
                            }
                        }

                        stmt_mac(ref m, _) => {
                            // Statement macro; might be an expr
                            match *self.token {
                                token::SEMI => {
                                    self.bump();
                                    stmts.push(@codemap::spanned {
                                        node: stmt_mac(copy *m, true),
                                        .. copy *stmt});
                                }
                                token::RBRACE => {
                                    // if a block ends in `m!(arg)` without
                                    // a `;`, it must be an expr
                                    expr = Some(
                                        self.mk_mac_expr(stmt.span.lo,
                                                         stmt.span.hi,
                                                         copy m.node));
                                }
                                _ => { stmts.push(stmt); }
                            }
                        }

                        _ => { // All other kinds of statements:
                            stmts.push(stmt);

                            if classify::stmt_ends_with_semi(*stmt) {
                                self.expect(&token::SEMI);
                            }
                        }
                    }
                }
            }
        }
        let mut hi = self.span.hi;
        self.bump();
        let bloc = ast::blk_ {
            view_items: view_items,
            stmts: stmts,
            expr: expr,
            id: self.get_id(),
            rules: s,
        };
        spanned(lo, hi, bloc)
    }

    fn mk_ty_path(&self, i: ident) -> @Ty {
        @Ty {
            id: self.get_id(),
            node: ty_path(
                ident_to_path(*self.last_span, i),
                self.get_id()),
            span: *self.last_span,
        }
    }

    fn parse_optional_purity(&self) -> ast::purity {
        if self.eat_keyword(&~"pure") {
            self.obsolete(*self.last_span, ObsoletePurity);
            ast::impure_fn
        } else if self.eat_keyword(&~"unsafe") {
            ast::unsafe_fn
        } else {
            ast::impure_fn
        }
    }

    fn parse_optional_onceness(&self) -> ast::Onceness {
        if self.eat_keyword(&~"once") { ast::Once } else { ast::Many }
    }

    fn parse_optional_ty_param_bounds(&self) -> @OptVec<TyParamBound> {
        if !self.eat(&token::COLON) {
            return @opt_vec::Empty;
        }

        let mut result = opt_vec::Empty;
        loop {
            match *self.token {
                token::LIFETIME(lifetime) => {
                    if str::eq_slice(*self.id_to_str(lifetime), "static") {
                        result.push(RegionTyParamBound);
                    } else {
                        self.span_err(*self.span,
                                      ~"`'static` is the only permissible \
                                        region bound here");
                    }
                    self.bump();
                }
                token::MOD_SEP | token::IDENT(*) => {
                    let maybe_bound = match *self.token {
                        token::MOD_SEP => None,
                        token::IDENT(copy sid, _) => {
                            match *self.id_to_str(sid) {
                                ~"send" |
                                ~"copy" |
                                ~"const" |
                                ~"owned" => {
                                    self.obsolete(
                                        *self.span,
                                        ObsoleteLowerCaseKindBounds);

                                    // Bogus value, but doesn't matter, since
                                    // is an error
                                    Some(TraitTyParamBound(
                                        self.mk_ty_path(sid)))
                                }
                                _ => None
                            }
                        }
                        _ => fail!()
                    };

                    match maybe_bound {
                        Some(bound) => {
                            self.bump();
                            result.push(bound);
                        }
                        None => {
                            let ty = self.parse_ty(true);
                            result.push(TraitTyParamBound(ty));
                        }
                    }
                }
                _ => break,
            }

            if self.eat(&token::BINOP(token::PLUS)) {
                loop;
            }

            if is_ident_or_path(&*self.token) {
                self.obsolete(*self.span,
                              ObsoleteTraitBoundSeparator);
            }
        }

        return @result;
    }

    fn parse_ty_param(&self) -> TyParam {
        let ident = self.parse_ident();
        let bounds = self.parse_optional_ty_param_bounds();
        ast::TyParam { ident: ident, id: self.get_id(), bounds: bounds }
    }

    fn parse_generics(&self) -> ast::Generics {
        if self.eat(&token::LT) {
            let lifetimes = self.parse_lifetimes();
            let ty_params = self.parse_seq_to_gt(
                Some(token::COMMA),
                |p| p.parse_ty_param());
            ast::Generics { lifetimes: lifetimes, ty_params: ty_params }
        } else {
            ast_util::empty_generics()
        }
    }

    fn parse_generic_values(
        &self) -> (OptVec<ast::Lifetime>, ~[@Ty])
    {
        if !self.eat(&token::LT) {
            (opt_vec::Empty, ~[])
        } else {
            self.parse_generic_values_after_lt()
        }
    }

    fn parse_generic_values_after_lt(
        &self) -> (OptVec<ast::Lifetime>, ~[@Ty])
    {
        let lifetimes = self.parse_lifetimes();
        let result = self.parse_seq_to_gt(
            Some(token::COMMA),
            |p| p.parse_ty(false));
        (lifetimes, opt_vec::take_vec(result))
    }

    fn parse_fn_decl(&self, parse_arg_fn: &fn(&Parser) -> arg_or_capture_item)
        -> fn_decl
    {
        let args_or_capture_items: ~[arg_or_capture_item] =
            self.parse_unspanned_seq(
                &token::LPAREN,
                &token::RPAREN,
                seq_sep_trailing_disallowed(token::COMMA),
                parse_arg_fn
            );

        let inputs = either::lefts(args_or_capture_items);

        let (ret_style, ret_ty) = self.parse_ret_ty();
        ast::fn_decl {
            inputs: inputs,
            output: ret_ty,
            cf: ret_style,
        }
    }

    fn is_self_ident(&self) -> bool {
        match *self.token {
          token::IDENT(id, false) if id == special_idents::self_
            => true,
          _ => false
        }
    }

    fn expect_self_ident(&self) {
        if !self.is_self_ident() {
            self.fatal(
                fmt!(
                    "expected `self` but found `%s`",
                    token_to_str(self.reader, &copy *self.token)
                )
            );
        }
        self.bump();
    }

    fn parse_fn_decl_with_self(
        &self,
        parse_arg_fn:
        &fn(&Parser) -> arg_or_capture_item
    ) -> (self_ty, fn_decl) {
        fn maybe_parse_self_ty(
            cnstr: &fn(+v: mutability) -> ast::self_ty_,
            p: &Parser
        ) -> ast::self_ty_ {
            // We need to make sure it isn't a mode or a type
            if p.token_is_keyword(&~"self", &p.look_ahead(1)) ||
                ((p.token_is_keyword(&~"const", &p.look_ahead(1)) ||
                  p.token_is_keyword(&~"mut", &p.look_ahead(1))) &&
                 p.token_is_keyword(&~"self", &p.look_ahead(2))) {

                p.bump();
                let mutability = p.parse_mutability();
                p.expect_self_ident();
                cnstr(mutability)
            } else {
                sty_static
            }
        }

        fn maybe_parse_borrowed_self_ty(
            self: &Parser
        ) -> ast::self_ty_ {
            // The following things are possible to see here:
            //
            //     fn(&self)
            //     fn(&mut self)
            //     fn(&'lt self)
            //     fn(&'lt mut self)
            //
            // We already know that the current token is `&`.

            if (
                self.token_is_keyword(&~"self", &self.look_ahead(1)))
            {
                self.bump();
                self.expect_self_ident();
                sty_region(None, m_imm)
            } else if (
                self.token_is_mutability(&self.look_ahead(1)) &&
                self.token_is_keyword(&~"self", &self.look_ahead(2)))
            {
                self.bump();
                let mutability = self.parse_mutability();
                self.expect_self_ident();
                sty_region(None, mutability)
            } else if (
                self.token_is_lifetime(&self.look_ahead(1)) &&
                self.token_is_keyword(&~"self", &self.look_ahead(2)))
            {
                self.bump();
                let lifetime = @self.parse_lifetime();
                self.expect_self_ident();
                sty_region(Some(lifetime), m_imm)
            } else if (
                self.token_is_lifetime(&self.look_ahead(1)) &&
                self.token_is_mutability(&self.look_ahead(2)) &&
                self.token_is_keyword(&~"self", &self.look_ahead(3)))
            {
                self.bump();
                let lifetime = @self.parse_lifetime();
                let mutability = self.parse_mutability();
                self.expect_self_ident();
                sty_region(Some(lifetime), mutability)
            } else {
                sty_static
            }
        }

        self.expect(&token::LPAREN);

        // A bit of complexity and lookahead is needed here in order to to be
        // backwards compatible.
        let lo = self.span.lo;
        let self_ty = match *self.token {
          token::BINOP(token::AND) => {
            maybe_parse_borrowed_self_ty(self)
          }
          token::AT => {
            maybe_parse_self_ty(sty_box, self)
          }
          token::TILDE => {
            maybe_parse_self_ty(sty_uniq, self)
          }
          token::IDENT(*) if self.is_self_ident() => {
            self.bump();
            sty_value
          }
          _ => {
            sty_static
          }
        };

        // If we parsed a self type, expect a comma before the argument list.
        let args_or_capture_items;
        if self_ty != sty_static {
            match *self.token {
                token::COMMA => {
                    self.bump();
                    let sep = seq_sep_trailing_disallowed(token::COMMA);
                    args_or_capture_items = self.parse_seq_to_before_end(
                        &token::RPAREN,
                        sep,
                        parse_arg_fn
                    );
                }
                token::RPAREN => {
                    args_or_capture_items = ~[];
                }
                _ => {
                    self.fatal(
                        fmt!(
                            "expected `,` or `)`, found `%s`",
                            token_to_str(self.reader, &copy *self.token)
                        )
                    );
                }
            }
        } else {
            let sep = seq_sep_trailing_disallowed(token::COMMA);
            args_or_capture_items = self.parse_seq_to_before_end(
                &token::RPAREN,
                sep,
                parse_arg_fn
            );
        }

        self.expect(&token::RPAREN);

        let hi = self.span.hi;

        let inputs = either::lefts(args_or_capture_items);
        let (ret_style, ret_ty) = self.parse_ret_ty();

        let fn_decl = ast::fn_decl {
            inputs: inputs,
            output: ret_ty,
            cf: ret_style
        };

        (spanned(lo, hi, self_ty), fn_decl)
    }

    fn parse_fn_block_decl(&self) -> fn_decl {
        let inputs_captures = {
            if self.eat(&token::OROR) {
                ~[]
            } else {
                self.parse_unspanned_seq(
                    &token::BINOP(token::OR),
                    &token::BINOP(token::OR),
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_fn_block_arg()
                )
            }
        };
        let output = if self.eat(&token::RARROW) {
            self.parse_ty(false)
        } else {
            @Ty { id: self.get_id(), node: ty_infer, span: *self.span }
        };

        ast::fn_decl {
            inputs: either::lefts(inputs_captures),
            output: output,
            cf: return_val,
        }
    }

    fn parse_fn_header(&self) -> (ident, ast::Generics) {
        let id = self.parse_ident();
        let generics = self.parse_generics();
        (id, generics)
    }

    fn mk_item(&self, +lo: BytePos, +hi: BytePos, +ident: ident,
               +node: item_, vis: visibility,
               +attrs: ~[attribute]) -> @item {
        @ast::item { ident: ident,
                     attrs: attrs,
                     id: self.get_id(),
                     node: node,
                     vis: vis,
                     span: mk_sp(lo, hi) }
    }

    fn parse_item_fn(&self, purity: purity) -> item_info {
        let (ident, generics) = self.parse_fn_header();
        let decl = self.parse_fn_decl(|p| p.parse_arg());
        let (inner_attrs, body) = self.parse_inner_attrs_and_block(true);
        (ident, item_fn(decl, purity, generics, body), Some(inner_attrs))
    }

    fn parse_method(&self) -> @method {
        let attrs = self.parse_outer_attributes();
        let lo = self.span.lo;

        let visa = self.parse_visibility();
        let pur = self.parse_fn_purity();
        let ident = self.parse_ident();
        let generics = self.parse_generics();
        let (self_ty, decl) = do self.parse_fn_decl_with_self() |p| {
            p.parse_arg()
        };

        let (inner_attrs, body) = self.parse_inner_attrs_and_block(true);
        let hi = body.span.hi;
        let attrs = vec::append(attrs, inner_attrs);
        @ast::method {
            ident: ident,
            attrs: attrs,
            generics: generics,
            self_ty: self_ty,
            purity: pur,
            decl: decl,
            body: body,
            id: self.get_id(),
            span: mk_sp(lo, hi),
            self_id: self.get_id(),
            vis: visa,
        }
    }

    fn parse_item_trait(&self) -> item_info {
        let ident = self.parse_ident();
        self.parse_region_param();
        let tps = self.parse_generics();

        // Parse traits, if necessary.
        let traits;
        if *self.token == token::COLON {
            self.bump();
            traits = self.parse_trait_ref_list(&token::LBRACE);
        } else {
            traits = ~[];
        }

        let meths = self.parse_trait_methods();
        (ident, item_trait(tps, traits, meths), None)
    }

    // Parses two variants (with the region/type params always optional):
    //    impl<T> Foo { ... }
    //    impl<T> ToStr for ~[T] { ... }
    fn parse_item_impl(&self, visibility: ast::visibility) -> item_info {
        // First, parse type parameters if necessary.
        let generics = self.parse_generics();

        // This is a new-style impl declaration.
        // XXX: clownshoes
        let ident = special_idents::clownshoes_extensions;

        // Special case: if the next identifier that follows is '(', don't
        // allow this to be parsed as a trait.
        let could_be_trait = *self.token != token::LPAREN;

        // Parse the trait.
        let mut ty = self.parse_ty(false);

        // Parse traits, if necessary.
        let opt_trait = if could_be_trait && self.eat_keyword(&~"for") {
            // New-style trait. Reinterpret the type as a trait.
            let opt_trait_ref = match ty.node {
                ty_path(path, node_id) => {
                    Some(@trait_ref {
                        path: path,
                        ref_id: node_id
                    })
                }
                _ => {
                    self.span_err(*self.span, ~"not a trait");
                    None
                }
            };

            ty = self.parse_ty(false);
            opt_trait_ref
        } else if self.eat(&token::COLON) {
            self.obsolete(copy *self.span, ObsoleteImplSyntax);
            Some(self.parse_trait_ref())
        } else {
            None
        };

        // Do not allow visibility to be specified in `impl...for...`. It is
        // meaningless.
        if opt_trait.is_some() && visibility != ast::inherited {
            self.obsolete(*self.span, ObsoleteTraitImplVisibility);
        }

        let mut meths = ~[];
        if !self.eat(&token::SEMI) {
            self.expect(&token::LBRACE);
            while !self.eat(&token::RBRACE) {
                meths.push(self.parse_method());
            }
        }

        (ident, item_impl(generics, opt_trait, ty, meths), None)
    }

    fn parse_trait_ref(&self) -> @trait_ref {
        @ast::trait_ref {
            path: self.parse_path_with_tps(false),
            ref_id: self.get_id(),
        }
    }

    fn parse_trait_ref_list(&self, ket: &token::Token) -> ~[@trait_ref] {
        self.parse_seq_to_before_end(
            ket,
            seq_sep_trailing_disallowed(token::BINOP(token::PLUS)),
            |p| p.parse_trait_ref()
        )
    }

    fn parse_item_struct(&self) -> item_info {
        let class_name = self.parse_ident();
        self.parse_region_param();
        let generics = self.parse_generics();
        if self.eat(&token::COLON) {
            self.obsolete(copy *self.span, ObsoleteClassTraits);
            let _ = self.parse_trait_ref_list(&token::LBRACE);
        }

        let mut fields: ~[@struct_field];
        let mut the_dtor: Option<(blk, ~[attribute], codemap::span)> = None;
        let is_tuple_like;

        if self.eat(&token::LBRACE) {
            // It's a record-like struct.
            is_tuple_like = false;
            fields = ~[];
            while *self.token != token::RBRACE {
                match self.parse_class_item() {
                  dtor_decl(ref blk, ref attrs, s) => {
                      match the_dtor {
                        Some((_, _, s_first)) => {
                          self.span_note(s, fmt!("Duplicate destructor \
                                     declaration for class %s",
                                     *self.interner.get(class_name)));
                          self.span_fatal(copy s_first, ~"First destructor \
                                                          declared here");
                        }
                        None => {
                          the_dtor = Some((copy *blk, copy *attrs, s));
                        }
                      }
                  }
                  members(mms) => {
                    for mms.each |struct_field| {
                        fields.push(*struct_field)
                    }
                  }
                }
            }
            if fields.len() == 0 {
                self.fatal(fmt!("Unit-like struct should be written as: struct %s;",
                                *self.interner.get(class_name)));
            }
            self.bump();
        } else if *self.token == token::LPAREN {
            // It's a tuple-like struct.
            is_tuple_like = true;
            fields = do self.parse_unspanned_seq(
                &token::LPAREN,
                &token::RPAREN,
                seq_sep_trailing_allowed(token::COMMA)
            ) |p| {
                let lo = p.span.lo;
                let struct_field_ = ast::struct_field_ {
                    kind: unnamed_field,
                    id: self.get_id(),
                    ty: p.parse_ty(false)
                };
                @spanned(lo, p.span.hi, struct_field_)
            };
            self.expect(&token::SEMI);
        } else if self.eat(&token::SEMI) {
            // It's a unit-like struct.
            is_tuple_like = true;
            fields = ~[];
        } else {
            self.fatal(
                fmt!(
                    "expected `{`, `(`, or `;` after struct name \
                    but found `%s`",
                    token_to_str(self.reader, &copy *self.token)
                )
            );
        }

        let actual_dtor = do the_dtor.map |dtor| {
            let (d_body, d_attrs, d_s) = copy *dtor;
            codemap::spanned { node: ast::struct_dtor_ { id: self.get_id(),
                                                     attrs: d_attrs,
                                                     self_id: self.get_id(),
                                                     body: d_body},
                       span: d_s}};
        let _ = self.get_id();  // XXX: Workaround for crazy bug.
        let new_id = self.get_id();
        (class_name,
         item_struct(@ast::struct_def {
             fields: fields,
             dtor: actual_dtor,
             ctor_id: if is_tuple_like { Some(new_id) } else { None }
         }, generics),
         None)
    }

    fn token_is_pound_or_doc_comment(&self, ++tok: token::Token) -> bool {
        match tok {
            token::POUND | token::DOC_COMMENT(_) => true,
            _ => false
        }
    }

    fn parse_single_class_item(&self, vis: visibility) -> @struct_field {
        if self.eat_obsolete_ident("let") {
            self.obsolete(*self.last_span, ObsoleteLet);
        }

        let a_var = self.parse_instance_var(vis);
        match *self.token {
            token::SEMI => {
                self.obsolete(copy *self.span, ObsoleteFieldTerminator);
                self.bump();
            }
            token::COMMA => {
                self.bump();
            }
            token::RBRACE => {}
            _ => {
                self.span_fatal(
                    copy *self.span,
                    fmt!(
                        "expected `;`, `,`, or '}' but found `%s`",
                        token_to_str(self.reader, &copy *self.token)
                    )
                );
            }
        }
        a_var
    }

    fn parse_dtor(&self, +attrs: ~[attribute]) -> class_contents {
        let lo = self.last_span.lo;
        let body = self.parse_block();
        dtor_decl(body, attrs, mk_sp(lo, self.last_span.hi))
    }

    fn parse_class_item(&self) -> class_contents {

        if self.try_parse_obsolete_priv_section() {
            return members(~[]);
        }

        let attrs = self.parse_outer_attributes();

        if self.eat_keyword(&~"priv") {
            return members(~[self.parse_single_class_item(private)])
        }

        if self.eat_keyword(&~"pub") {
           return members(~[self.parse_single_class_item(public)]);
        }

        if self.try_parse_obsolete_struct_ctor() {
            return members(~[]);
        }

        if self.eat_keyword(&~"drop") {
           return self.parse_dtor(attrs);
        }
        else {
           return members(~[self.parse_single_class_item(inherited)]);
        }
    }

    fn parse_visibility(&self) -> visibility {
        if self.eat_keyword(&~"pub") { public }
        else if self.eat_keyword(&~"priv") { private }
        else { inherited }
    }

    fn parse_staticness(&self) -> bool {
        if self.eat_keyword(&~"static") {
            self.obsolete(*self.last_span, ObsoleteStaticMethod);
            true
        } else {
            false
        }
    }

    // given a termination token and a vector of already-parsed
    // attributes (of length 0 or 1), parse all of the items in a module
    fn parse_mod_items(&self, term: token::Token,
                       +first_item_attrs: ~[attribute]) -> _mod {
        // parse all of the items up to closing or an attribute.
        // view items are legal here.
        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: starting_items,
            _
        } = self.parse_items_and_view_items(first_item_attrs,
                                            VIEW_ITEMS_AND_ITEMS_ALLOWED,
                                            true);
        let mut items: ~[@item] = starting_items;
        let attrs_remaining_len = attrs_remaining.len();

        // looks like this code depends on the invariant that
        // outer attributes can't occur on view items (or macros
        // invocations?)
        let mut first = true;
        while *self.token != term {
            let mut attrs = self.parse_outer_attributes();
            if first {
                attrs = attrs_remaining + attrs;
                first = false;
            }
            debug!("parse_mod_items: parse_item_or_view_item(attrs=%?)",
                   attrs);
            match self.parse_item_or_view_item(
                /*bad*/ copy attrs,
                true,
                false,
                true
            ) {
              iovi_item(item) => items.push(item),
              iovi_view_item(view_item) => {
                self.span_fatal(view_item.span, ~"view items must be \
                                                  declared at the top of the \
                                                  module");
              }
              _ => {
                self.fatal(
                    fmt!(
                        "expected item but found `%s`",
                        token_to_str(self.reader, &copy *self.token)
                    )
                );
              }
            }
            debug!("parse_mod_items: attrs=%?", attrs);
        }

        if first && attrs_remaining_len > 0u {
            // We parsed attributes for the first item but didn't find it
            self.fatal(~"expected item");
        }

        ast::_mod { view_items: view_items, items: items }
    }

    fn parse_item_const(&self) -> item_info {
        let id = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        self.expect(&token::EQ);
        let e = self.parse_expr();
        self.expect(&token::SEMI);
        (id, item_const(ty, e), None)
    }

    fn parse_item_mod(&self, outer_attrs: ~[ast::attribute]) -> item_info {
        let id_span = *self.span;
        let id = self.parse_ident();
        let info_ = if *self.token == token::SEMI {
            self.bump();
            // This mod is in an external file. Let's go get it!
            let (m, attrs) = self.eval_src_mod(id, outer_attrs, id_span);
            (id, m, Some(attrs))
        } else {
            self.push_mod_path(id, outer_attrs);
            self.expect(&token::LBRACE);
            let (inner, next) = self.parse_inner_attrs_and_next();
            let m = self.parse_mod_items(token::RBRACE, next);
            self.expect(&token::RBRACE);
            self.pop_mod_path();
            (id, item_mod(m), Some(inner))
        };

        // XXX: Transitionary hack to do the template work inside core
        // (int-template, iter-trait). If there's a 'merge' attribute
        // on the mod, then we'll go and suck in another file and merge
        // its contents
        match ::attr::first_attr_value_str_by_name(outer_attrs, ~"merge") {
            Some(path) => {
                let prefix = Path(
                    self.sess.cm.span_to_filename(*self.span));
                let prefix = prefix.dir_path();
                let path = Path(copy *path);
                let (new_mod_item, new_attrs) = self.eval_src_mod_from_path(
                    prefix, path, ~[], id_span);

                let (main_id, main_mod_item, main_attrs) = info_;
                let main_attrs = main_attrs.get();

                let (main_mod, new_mod) =
                    match (main_mod_item, new_mod_item) {
                    (item_mod(m), item_mod(n)) => (m, n),
                    _ => self.bug(~"parsed mod item should be mod")
                };
                let merged_mod = ast::_mod {
                    view_items: main_mod.view_items + new_mod.view_items,
                    items: main_mod.items + new_mod.items
                };

                let merged_attrs = main_attrs + new_attrs;
                (main_id, item_mod(merged_mod), Some(merged_attrs))
            }
            None => info_
        }
    }

    fn push_mod_path(&self, id: ident, attrs: ~[ast::attribute]) {
        let default_path = self.sess.interner.get(id);
        let file_path = match ::attr::first_attr_value_str_by_name(
            attrs, ~"path") {

            Some(d) => copy *d,
            None => copy *default_path
        };
        self.mod_path_stack.push(file_path)
    }

    fn pop_mod_path(&self) {
        self.mod_path_stack.pop();
    }

    fn eval_src_mod(&self, id: ast::ident,
                    outer_attrs: ~[ast::attribute],
                    id_sp: span) -> (ast::item_, ~[ast::attribute]) {

        let prefix = Path(self.sess.cm.span_to_filename(*self.span));
        let prefix = prefix.dir_path();
        let mod_path_stack = &*self.mod_path_stack;
        let mod_path = Path(".").push_many(*mod_path_stack);
        let default_path = *self.sess.interner.get(id) + ~".rs";
        let file_path = match ::attr::first_attr_value_str_by_name(
            outer_attrs, ~"path") {
            Some(d) => {
                let path = Path(copy *d);
                if !path.is_absolute {
                    mod_path.push(copy *d)
                } else {
                    path
                }
            }
            None => mod_path.push(default_path)
        };

        self.eval_src_mod_from_path(prefix, file_path,
                                    outer_attrs, id_sp)
    }

    fn eval_src_mod_from_path(&self, prefix: Path, +path: Path,
                              outer_attrs: ~[ast::attribute],
                              id_sp: span
                             ) -> (ast::item_, ~[ast::attribute]) {

        let full_path = if path.is_absolute {
            path
        } else {
            prefix.push_many(path.components)
        };
        let full_path = full_path.normalize();
        let p0 =
            new_sub_parser_from_file(self.sess, copy self.cfg,
                                     &full_path, id_sp);
        let (inner, next) = p0.parse_inner_attrs_and_next();
        let mod_attrs = vec::append(
            /*bad*/ copy outer_attrs,
            inner
        );
        let first_item_outer_attrs = next;
        let m0 = p0.parse_mod_items(token::EOF, first_item_outer_attrs);
        return (ast::item_mod(m0), mod_attrs);

        fn cdir_path_opt(+default: ~str, attrs: ~[ast::attribute]) -> ~str {
            match ::attr::first_attr_value_str_by_name(attrs, ~"path") {
                Some(d) => copy *d,
                None => default
            }
        }
    }

    fn parse_item_foreign_fn(&self,  +attrs: ~[attribute]) -> @foreign_item {
        let lo = self.span.lo;
        let vis = self.parse_visibility();
        let purity = self.parse_fn_purity();
        let (ident, generics) = self.parse_fn_header();
        let decl = self.parse_fn_decl(|p| p.parse_arg());
        let mut hi = self.span.hi;
        self.expect(&token::SEMI);
        @ast::foreign_item { ident: ident,
                             attrs: attrs,
                             node: foreign_item_fn(decl, purity, generics),
                             id: self.get_id(),
                             span: mk_sp(lo, hi),
                             vis: vis }
    }

    fn parse_item_foreign_const(&self, vis: ast::visibility,
                                +attrs: ~[attribute]) -> @foreign_item {
        let lo = self.span.lo;

        // XXX: Obsolete; remove after snap.
        if self.eat_keyword(&~"const") {
            self.obsolete(*self.last_span, ObsoleteConstItem);
        } else {
            self.expect_keyword(&~"static");
        }

        let ident = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        let hi = self.span.hi;
        self.expect(&token::SEMI);
        @ast::foreign_item { ident: ident,
                             attrs: attrs,
                             node: foreign_item_const(ty),
                             id: self.get_id(),
                             span: mk_sp(lo, hi),
                             vis: vis }
    }

    fn parse_fn_purity(&self) -> purity {
        if self.eat_keyword(&~"fn") { impure_fn }
        else if self.eat_keyword(&~"pure") {
            self.obsolete(*self.last_span, ObsoletePurity);
            self.expect_keyword(&~"fn");
            // NB: We parse this as impure for bootstrapping purposes.
            impure_fn
        } else if self.eat_keyword(&~"unsafe") {
            self.expect_keyword(&~"fn");
            unsafe_fn
        }
        else { self.unexpected(); }
    }

    fn parse_foreign_item(&self, +attrs: ~[attribute]) -> @foreign_item {
        let vis = self.parse_visibility();
        if self.is_keyword(&~"const") || self.is_keyword(&~"static") {
            self.parse_item_foreign_const(vis, attrs)
        } else {
            self.parse_item_foreign_fn(attrs)
        }
    }

    fn parse_foreign_mod_items(&self, sort: ast::foreign_mod_sort,
                               +abi: ast::ident,
                               +first_item_attrs: ~[attribute])
                            -> foreign_mod {
        // Shouldn't be any view items since we've already parsed an item attr
        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: _,
            foreign_items: foreign_items
        } = self.parse_items_and_view_items(first_item_attrs,
                                            FOREIGN_ITEMS_ALLOWED,
                                            true);

        let mut items: ~[@foreign_item] = foreign_items;
        let mut initial_attrs = attrs_remaining;
        while *self.token != token::RBRACE {
            let attrs = vec::append(initial_attrs,
                                    self.parse_outer_attributes());
            initial_attrs = ~[];
            items.push(self.parse_foreign_item(attrs));
        }
        ast::foreign_mod {
            sort: sort,
            abi: abi,
            view_items: view_items,
            items: items
        }
    }

    fn parse_item_foreign_mod(&self, lo: BytePos,
                              visibility: visibility,
                              attrs: ~[attribute],
                              items_allowed: bool)
                           -> item_or_view_item {

        // Parse the ABI.
        let abi_opt;
        match *self.token {
            token::LIT_STR(copy found_abi) => {
                self.bump();
                abi_opt = Some(found_abi);
            }
            _ => {
                abi_opt = None;
            }
        }

        let mut must_be_named_mod = false;
        if self.is_keyword(&~"mod") {
            must_be_named_mod = true;
            self.expect_keyword(&~"mod");
        } else if *self.token != token::LBRACE {
            self.span_fatal(
                copy *self.span,
                fmt!(
                    "expected `{` or `mod` but found `%s`",
                    token_to_str(self.reader, &copy *self.token)
                )
            );
        }

        let (sort, ident) = match *self.token {
            token::IDENT(*) => (ast::named, self.parse_ident()),
            _ => {
                if must_be_named_mod {
                    self.span_fatal(
                        copy *self.span,
                        fmt!(
                            "expected foreign module name but found `%s`",
                            token_to_str(self.reader, &copy *self.token)
                        )
                    );
                }

                (ast::anonymous,
                 special_idents::clownshoes_foreign_mod)
            }
        };

        // extern mod { ... }
        if items_allowed && self.eat(&token::LBRACE) {
            let abi;
            match abi_opt {
                Some(found_abi) => abi = found_abi,
                None => abi = special_idents::c_abi,
            }

            let (inner, next) = self.parse_inner_attrs_and_next();
            let m = self.parse_foreign_mod_items(sort, abi, next);
            self.expect(&token::RBRACE);

            return iovi_item(self.mk_item(lo, self.last_span.hi, ident,
                                          item_foreign_mod(m), visibility,
                                          maybe_append(/*bad*/ copy attrs,
                                                       Some(inner))));
        }

        match abi_opt {
            None => {}  // OK.
            Some(_) => {
                self.span_err(*self.span, ~"an ABI may not be specified \
                                                here");
            }
        }

        // extern mod foo;
        let metadata = self.parse_optional_meta();
        self.expect(&token::SEMI);
        iovi_view_item(@ast::view_item {
            node: view_item_extern_mod(ident, metadata, self.get_id()),
            attrs: copy attrs,
            vis: visibility,
            span: mk_sp(lo, self.last_span.hi)
        })
    }

    fn parse_type_decl(&self) -> (BytePos, ident) {
        let lo = self.last_span.lo;
        let id = self.parse_ident();
        (lo, id)
    }

    fn parse_item_type(&self) -> item_info {
        let (_, ident) = self.parse_type_decl();
        self.parse_region_param();
        let tps = self.parse_generics();
        self.expect(&token::EQ);
        let ty = self.parse_ty(false);
        self.expect(&token::SEMI);
        (ident, item_ty(ty, tps), None)
    }

    fn parse_region_param(&self) {
        if self.eat(&token::BINOP(token::SLASH)) {
            self.obsolete(*self.last_span, ObsoleteLifetimeNotation);
            self.expect(&token::BINOP(token::AND));
        }
    }

    fn parse_struct_def(&self) -> @struct_def {
        let mut the_dtor: Option<(blk, ~[attribute], codemap::span)> = None;
        let mut fields: ~[@struct_field] = ~[];
        while *self.token != token::RBRACE {
            match self.parse_class_item() {
                dtor_decl(ref blk, ref attrs, s) => {
                    match the_dtor {
                        Some((_, _, s_first)) => {
                            self.span_note(s, ~"duplicate destructor \
                                                declaration");
                            self.span_fatal(copy s_first,
                                            ~"first destructor \
                                              declared here");
                        }
                        None => {
                            the_dtor = Some((copy *blk, copy *attrs, s));
                        }
                    }
                }
                members(mms) => {
                    for mms.each |struct_field| {
                        fields.push(*struct_field);
                    }
                }
            }
        }
        self.bump();
        let mut actual_dtor = do the_dtor.map |dtor| {
            let (d_body, d_attrs, d_s) = copy *dtor;
            codemap::spanned { node: ast::struct_dtor_ { id: self.get_id(),
                                                     attrs: d_attrs,
                                                     self_id: self.get_id(),
                                                     body: d_body },
                      span: d_s }
        };

        return @ast::struct_def {
            fields: fields,
            dtor: actual_dtor,
            ctor_id: None
        };
    }

    fn parse_enum_def(&self, generics: &ast::Generics) -> enum_def {
        let mut variants = ~[];
        let mut all_nullary = true, have_disr = false;
        while *self.token != token::RBRACE {
            let variant_attrs = self.parse_outer_attributes();
            let vlo = self.span.lo;

            let vis = self.parse_visibility();

            let ident, needs_comma, kind;
            let mut args = ~[], disr_expr = None;
            ident = self.parse_ident();
            if self.eat(&token::LBRACE) {
                // Parse a struct variant.
                all_nullary = false;
                kind = struct_variant_kind(self.parse_struct_def());
            } else if *self.token == token::LPAREN {
                all_nullary = false;
                let arg_tys = self.parse_unspanned_seq(
                    &token::LPAREN,
                    &token::RPAREN,
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_ty(false)
                );
                for arg_tys.each |ty| {
                    args.push(ast::variant_arg {
                        ty: *ty,
                        id: self.get_id(),
                    });
                }
                kind = tuple_variant_kind(args);
            } else if self.eat(&token::EQ) {
                have_disr = true;
                disr_expr = Some(self.parse_expr());
                kind = tuple_variant_kind(args);
            } else {
                kind = tuple_variant_kind(~[]);
            }
            needs_comma = true;

            let vr = ast::variant_ {
                name: ident,
                attrs: variant_attrs,
                kind: kind,
                id: self.get_id(),
                disr_expr: disr_expr,
                vis: vis,
            };
            variants.push(spanned(vlo, self.last_span.hi, vr));

            if needs_comma && !self.eat(&token::COMMA) { break; }
        }
        self.expect(&token::RBRACE);
        if (have_disr && !all_nullary) {
            self.fatal(~"discriminator values can only be used with a c-like \
                        enum");
        }

        ast::enum_def { variants: variants }
    }

    fn parse_item_enum(&self) -> item_info {
        let id = self.parse_ident();
        self.parse_region_param();
        let generics = self.parse_generics();
        // Newtype syntax
        if *self.token == token::EQ {
            self.bump();
            let ty = self.parse_ty(false);
            self.expect(&token::SEMI);
            let variant = spanned(ty.span.lo, ty.span.hi, ast::variant_ {
                name: id,
                attrs: ~[],
                kind: tuple_variant_kind(
                    ~[ast::variant_arg {ty: ty, id: self.get_id()}]
                ),
                id: self.get_id(),
                disr_expr: None,
                vis: public,
            });

            self.obsolete(*self.last_span, ObsoleteNewtypeEnum);

            return (
                id,
                item_enum(
                    ast::enum_def { variants: ~[variant] },
                    generics),
                None
            );
        }
        self.expect(&token::LBRACE);

        let enum_definition = self.parse_enum_def(&generics);
        (id, item_enum(enum_definition, generics), None)
    }

    fn parse_fn_ty_sigil(&self) -> Option<Sigil> {
        match *self.token {
            token::AT => {
                self.bump();
                Some(ManagedSigil)
            }
            token::TILDE => {
                self.bump();
                Some(OwnedSigil)
            }
            token::BINOP(token::AND) => {
                self.bump();
                Some(BorrowedSigil)
            }
            _ => {
                None
            }
        }
    }

    fn fn_expr_lookahead(&self, tok: token::Token) -> bool {
        match tok {
          token::LPAREN | token::AT | token::TILDE | token::BINOP(_) => true,
          _ => false
        }
    }

    // parse one of the items or view items allowed by the
    // flags; on failure, return iovi_none.
    fn parse_item_or_view_item(
        &self,
        +attrs: ~[attribute],
        items_allowed: bool,
        foreign_items_allowed: bool,
        macros_allowed: bool
    ) -> item_or_view_item {
        fail_unless!(items_allowed != foreign_items_allowed);

        maybe_whole!(iovi self, nt_item);
        let lo = self.span.lo;

        let visibility;
        if self.eat_keyword(&~"pub") {
            visibility = public;
        } else if self.eat_keyword(&~"priv") {
            visibility = private;
        } else {
            visibility = inherited;
        }

        if items_allowed &&
                (self.is_keyword(&~"const") ||
                (self.is_keyword(&~"static") &&
                    !self.token_is_keyword(&~"fn", &self.look_ahead(1)))) {
            // CONST ITEM
            if self.is_keyword(&~"const") {
                self.obsolete(*self.span, ObsoleteConstItem);
            }
            self.bump();
            let (ident, item_, extra_attrs) = self.parse_item_const();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if foreign_items_allowed &&
                (self.is_keyword(&~"const") || self.is_keyword(&~"static")) {
            // FOREIGN CONST ITEM
            let item = self.parse_item_foreign_const(visibility, attrs);
            return iovi_foreign_item(item);
        }
        if items_allowed &&
            // FUNCTION ITEM (not sure about lookahead condition...)
            self.is_keyword(&~"fn") &&
            !self.fn_expr_lookahead(self.look_ahead(1u)) {
            self.bump();
            let (ident, item_, extra_attrs) = self.parse_item_fn(impure_fn);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if items_allowed && self.eat_keyword(&~"pure") {
            // PURE FUNCTION ITEM
            self.obsolete(*self.last_span, ObsoletePurity);
            self.expect_keyword(&~"fn");
            let (ident, item_, extra_attrs) = self.parse_item_fn(impure_fn);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if foreign_items_allowed &&
            (self.is_keyword(&~"fn") || self.is_keyword(&~"pure") ||
             self.is_keyword(&~"unsafe")) {
            // FOREIGN FUNCTION ITEM (no items allowed)
                let item = self.parse_item_foreign_fn(attrs);
                return iovi_foreign_item(item);
        }
        if items_allowed && self.is_keyword(&~"unsafe")
            && self.look_ahead(1u) != token::LBRACE {
            // UNSAFE FUNCTION ITEM (where items are allowed)
            self.bump();
            self.expect_keyword(&~"fn");
            let (ident, item_, extra_attrs) = self.parse_item_fn(unsafe_fn);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if self.eat_keyword(&~"extern") {
            if items_allowed && self.eat_keyword(&~"fn") {
                // EXTERN FUNCTION ITEM
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(extern_fn);
                return iovi_item(self.mk_item(lo, self.last_span.hi, ident,
                                              item_, visibility,
                                              maybe_append(attrs,
                                                           extra_attrs)));
            }
            if !foreign_items_allowed {
                // EXTERN MODULE ITEM
                return self.parse_item_foreign_mod(lo, visibility, attrs,
                                                   items_allowed);
            }
        }
        if items_allowed && !foreign_items_allowed &&
                self.eat_keyword(&~"mod") {
            // MODULE ITEM
            let (ident, item_, extra_attrs) = self.parse_item_mod(attrs);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if items_allowed && !foreign_items_allowed &&
                self.eat_keyword(&~"type") {
            // TYPE ITEM
            let (ident, item_, extra_attrs) = self.parse_item_type();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if items_allowed && !foreign_items_allowed &&
                self.eat_keyword(&~"enum") {
            // ENUM ITEM
            let (ident, item_, extra_attrs) = self.parse_item_enum();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if items_allowed && !foreign_items_allowed &&
                self.eat_keyword(&~"trait") {
            // TRAIT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_trait();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if items_allowed && !foreign_items_allowed &&
                self.eat_keyword(&~"impl") {
            // IMPL ITEM
            let (ident, item_, extra_attrs) =
                self.parse_item_impl(visibility);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if items_allowed && !foreign_items_allowed &&
                self.eat_keyword(&~"struct") {
            // STRUCT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_struct();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        }
        if !foreign_items_allowed && self.eat_keyword(&~"use") {
            // USE ITEM
            let view_item = self.parse_use();
            self.expect(&token::SEMI);
            return iovi_view_item(@ast::view_item {
                node: view_item,
                attrs: attrs,
                vis: visibility,
                span: mk_sp(lo, self.last_span.hi)
            });
        }
        if macros_allowed && !self.is_any_keyword(&copy *self.token)
                && self.look_ahead(1) == token::NOT
                && (is_plain_ident(&self.look_ahead(2))
                    || self.look_ahead(2) == token::LPAREN
                    || self.look_ahead(2) == token::LBRACE) {
            // MACRO INVOCATION ITEM
            if attrs.len() > 0 {
                self.fatal(~"attrs on macros are not yet supported");
            }

            // item macro.
            let pth = self.parse_path_without_tps();
            self.expect(&token::NOT);

            // a 'special' identifier (like what `macro_rules!` uses)
            // is optional. We should eventually unify invoc syntax
            // and remove this.
            let id = if is_plain_ident(&*self.token) {
                self.parse_ident()
            } else {
                token::special_idents::invalid // no special identifier
            };
            // eat a matched-delimiter token tree:
            let tts = match *self.token {
                token::LPAREN | token::LBRACE => {
                    let ket = token::flip_delimiter(&*self.token);
                    self.parse_unspanned_seq(
                        &copy *self.token,
                        &ket,
                        seq_sep_none(),
                        |p| p.parse_token_tree()
                    )
                }
                _ => self.fatal(~"expected open delimiter")
            };
            // single-variant-enum... :
            let m = ast::mac_invoc_tt(pth, tts);
            let m: ast::mac = codemap::spanned { node: m,
                                             span: mk_sp(self.span.lo,
                                                         self.span.hi) };
            let item_ = item_mac(m);
            return iovi_item(self.mk_item(lo, self.last_span.hi, id, item_,
                                          visibility, attrs));
        }

        // FAILURE TO PARSE ITEM
        if visibility != inherited {
            let mut s = ~"unmatched visibility `";
            s += if visibility == public { ~"pub" } else { ~"priv" };
            s += ~"`";
            self.span_fatal(*self.last_span, s);
        }
        return iovi_none;
    }

    fn parse_item(&self, +attrs: ~[attribute]) -> Option<@ast::item> {
        match self.parse_item_or_view_item(attrs, true, false, true) {
            iovi_none =>
                None,
            iovi_view_item(_) =>
                self.fatal(~"view items are not allowed here"),
            iovi_foreign_item(_) =>
                self.fatal(~"foreign items are not allowed here"),
            iovi_item(item) =>
                Some(item)
        }
    }

    fn parse_use(&self) -> view_item_ {
        return view_item_use(self.parse_view_paths());
    }

    fn parse_view_path(&self) -> @view_path {
        let lo = self.span.lo;

        let namespace;
        if self.eat_keyword(&~"mod") {
            namespace = module_ns;
        } else {
            namespace = type_value_ns;
        }

        let first_ident = self.parse_ident();
        let mut path = ~[first_ident];
        debug!("parsed view_path: %s", *self.id_to_str(first_ident));
        match *self.token {
          token::EQ => {
            // x = foo::bar
            self.bump();
            path = ~[self.parse_ident()];
            while *self.token == token::MOD_SEP {
                self.bump();
                let id = self.parse_ident();
                path.push(id);
            }
            let path = @ast::path { span: mk_sp(lo, self.span.hi),
                                    global: false,
                                    idents: path,
                                    rp: None,
                                    types: ~[] };
            return @spanned(lo, self.span.hi,
                         view_path_simple(first_ident, path, namespace,
                                          self.get_id()));
          }

          token::MOD_SEP => {
            // foo::bar or foo::{a,b,c} or foo::*
            while *self.token == token::MOD_SEP {
                self.bump();

                match *self.token {
                  token::IDENT(i, _) => {
                    self.bump();
                    path.push(i);
                  }

                  // foo::bar::{a,b,c}
                  token::LBRACE => {
                    let idents = self.parse_unspanned_seq(
                        &token::LBRACE,
                        &token::RBRACE,
                        seq_sep_trailing_allowed(token::COMMA),
                        |p| p.parse_path_list_ident()
                    );
                    let path = @ast::path { span: mk_sp(lo, self.span.hi),
                                            global: false,
                                            idents: path,
                                            rp: None,
                                            types: ~[] };
                    return @spanned(lo, self.span.hi,
                                 view_path_list(path, idents, self.get_id()));
                  }

                  // foo::bar::*
                  token::BINOP(token::STAR) => {
                    self.bump();
                    let path = @ast::path { span: mk_sp(lo, self.span.hi),
                                            global: false,
                                            idents: path,
                                            rp: None,
                                            types: ~[] };
                    return @spanned(lo, self.span.hi,
                                 view_path_glob(path, self.get_id()));
                  }

                  _ => break
                }
            }
          }
          _ => ()
        }
        let last = path[vec::len(path) - 1u];
        let path = @ast::path { span: mk_sp(lo, self.span.hi),
                                global: false,
                                idents: path,
                                rp: None,
                                types: ~[] };
        return @spanned(lo, self.span.hi,
                     view_path_simple(last, path, namespace, self.get_id()));
    }

    fn parse_view_paths(&self) -> ~[@view_path] {
        let mut vp = ~[self.parse_view_path()];
        while *self.token == token::COMMA {
            self.bump();
            vp.push(self.parse_view_path());
        }
        return vp;
    }

    fn is_view_item(&self) -> bool {
        let tok, next_tok;
        if !self.is_keyword(&~"pub") && !self.is_keyword(&~"priv") {
            tok = copy *self.token;
            next_tok = self.look_ahead(1);
        } else {
            tok = self.look_ahead(1);
            next_tok = self.look_ahead(2);
        };
        self.token_is_keyword(&~"use", &tok)
            || (self.token_is_keyword(&~"extern", &tok) &&
                self.token_is_keyword(&~"mod", &next_tok))
    }

    // parse a view item.
    fn parse_view_item(
        &self,
        +attrs: ~[attribute],
        vis: visibility
    ) -> @view_item {
        let lo = self.span.lo;
        let node = if self.eat_keyword(&~"use") {
            self.parse_use()
        } else if self.eat_keyword(&~"extern") {
            self.expect_keyword(&~"mod");
            let ident = self.parse_ident();
            let metadata = self.parse_optional_meta();
            view_item_extern_mod(ident, metadata, self.get_id())
        } else {
            self.bug(~"expected view item");
        };
        self.expect(&token::SEMI);
        @ast::view_item { node: node,
                          attrs: attrs,
                          vis: vis,
                          span: mk_sp(lo, self.last_span.hi) }
    }

    // Parses a sequence of items. Stops when it finds program
    // text that can't be parsed as an item
    fn parse_items_and_view_items(&self, +first_item_attrs: ~[attribute],
                                  mode: view_item_parse_mode,
                                  macros_allowed: bool)
                                -> ParsedItemsAndViewItems {
        let mut attrs = vec::append(first_item_attrs,
                                    self.parse_outer_attributes());

        let items_allowed = match mode {
            VIEW_ITEMS_AND_ITEMS_ALLOWED | IMPORTS_AND_ITEMS_ALLOWED => true,
            FOREIGN_ITEMS_ALLOWED => false
        };
        let foreign_items_allowed = match mode {
            FOREIGN_ITEMS_ALLOWED => true,
            VIEW_ITEMS_AND_ITEMS_ALLOWED | IMPORTS_AND_ITEMS_ALLOWED => false
        };

        // First, parse view items.
        let mut (view_items, items, foreign_items) = (~[], ~[], ~[]);
        let mut done = false;
        if mode != FOREIGN_ITEMS_ALLOWED {
            let mut extern_mod_allowed = match mode {
                VIEW_ITEMS_AND_ITEMS_ALLOWED => true,
                IMPORTS_AND_ITEMS_ALLOWED => false,
                FOREIGN_ITEMS_ALLOWED => {
                    self.bug(~"couldn't get here with FOREIGN_ITEMS_ALLOWED")
                }
            };

            loop {
                match self.parse_item_or_view_item(/*bad*/ copy attrs,
                                                   items_allowed,
                                                   foreign_items_allowed,
                                                   macros_allowed) {
                    iovi_none => {
                        done = true;
                        break;
                    }
                    iovi_view_item(view_item) => {
                        match view_item.node {
                            view_item_use(*) => {
                                // `extern mod` must precede `use`.
                                extern_mod_allowed = false;
                            }
                            view_item_extern_mod(*)
                                    if !extern_mod_allowed => {
                                self.span_err(view_item.span,
                                              ~"\"extern mod\" \
                                                declarations are not \
                                                allowed here");
                            }
                            view_item_extern_mod(*) => {}
                        }
                        view_items.push(view_item);
                    }
                    iovi_item(item) => {
                        fail_unless!(items_allowed);
                        items.push(item);
                        attrs = self.parse_outer_attributes();
                        break;
                    }
                    iovi_foreign_item(foreign_item) => {
                        fail_unless!(foreign_items_allowed);
                        foreign_items.push(foreign_item);
                        attrs = self.parse_outer_attributes();
                        break;
                    }
                }
                attrs = self.parse_outer_attributes();
            }
        }

        // Next, parse items.
        if !done {
            loop {
                match self.parse_item_or_view_item(/*bad*/ copy attrs,
                                                   items_allowed,
                                                   foreign_items_allowed,
                                                   macros_allowed) {
                    iovi_none => break,
                    iovi_view_item(view_item) => {
                        self.span_err(view_item.span,
                                      ~"`use` and `extern mod` declarations \
                                        must precede items");
                        view_items.push(view_item);
                    }
                    iovi_item(item) => {
                        fail_unless!(items_allowed);
                        items.push(item)
                    }
                    iovi_foreign_item(foreign_item) => {
                        fail_unless!(foreign_items_allowed);
                        foreign_items.push(foreign_item);
                    }
                }
                attrs = self.parse_outer_attributes();
            }
        }

        ParsedItemsAndViewItems {
            attrs_remaining: attrs,
            view_items: view_items,
            items: items,
            foreign_items: foreign_items
        }
    }

    // Parses a source module as a crate
    fn parse_crate_mod(&self, _cfg: crate_cfg) -> @crate {
        let lo = self.span.lo;
        // parse the crate's inner attrs, maybe (oops) one
        // of the attrs of an item:
        let (inner, next) = self.parse_inner_attrs_and_next();
        let first_item_outer_attrs = next;
        // parse the items inside the crate:
        let m = self.parse_mod_items(token::EOF, first_item_outer_attrs);
        @spanned(lo, self.span.lo,
                 ast::crate_ { module: m,
                               attrs: inner,
                               config: copy self.cfg })
    }

    fn parse_str(&self) -> @~str {
        match *self.token {
          token::LIT_STR(s) => { self.bump(); self.id_to_str(s) }
          _ =>  self.fatal(~"expected string literal")
        }
    }
}


//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

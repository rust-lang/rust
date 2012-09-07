use print::pprust::expr_to_str;

use result::Result;
use either::{Either, Left, Right};
use std::map::{hashmap, str_hash};
use token::{can_begin_expr, is_ident, is_ident_or_path, is_plain_ident,
               INTERPOLATED};
use codemap::{span,fss_none};
use util::interner::interner;
use ast_util::{spanned, respan, mk_sp, ident_to_path, operator_prec};
use lexer::reader;
use prec::{as_prec, token_to_binop};
use attr::parser_attr;
use common::{seq_sep_trailing_disallowed, seq_sep_trailing_allowed,
                seq_sep_none, token_to_str};
use dvec::DVec;
use vec::{push};
use ast::{_mod, add, alt_check, alt_exhaustive, arg, arm, attribute,
             bind_by_ref, bind_by_implicit_ref, bind_by_value, bind_by_move,
             bitand, bitor, bitxor, blk, blk_check_mode, bound_const,
             bound_copy, bound_send, bound_trait, bound_owned, box, by_copy,
             by_move, by_mutbl_ref, by_ref, by_val, capture_clause,
             capture_item, cdir_dir_mod, cdir_src_mod, cdir_view_item,
             class_immutable, class_mutable,
             crate, crate_cfg, crate_directive, decl, decl_item, decl_local,
             default_blk, deref, div, enum_def, enum_variant_kind, expl, expr,
             expr_, expr_addr_of, expr_match, expr_again, expr_assert,
             expr_assign, expr_assign_op, expr_binary, expr_block, expr_break,
             expr_call, expr_cast, expr_copy, expr_do_body, expr_fail,
             expr_field, expr_fn, expr_fn_block, expr_if, expr_index,
             expr_lit, expr_log, expr_loop, expr_loop_body, expr_mac,
             expr_move, expr_path, expr_rec, expr_repeat, expr_ret, expr_swap,
             expr_struct, expr_tup, expr_unary, expr_unary_move, expr_vec,
             expr_vstore, expr_while, extern_fn, field, fn_decl, foreign_item,
             foreign_item_const, foreign_item_fn, foreign_mod, ident,
             impure_fn, infer, inherited, init_assign, init_move, initializer,
             item, item_, item_class, item_const, item_enum, item_fn,
             item_foreign_mod, item_impl, item_mac, item_mod, item_trait,
             item_ty, lit, lit_, lit_bool, lit_float, lit_int,
             lit_int_unsuffixed, lit_nil, lit_str, lit_uint, local, m_const,
             m_imm, m_mutbl, mac_, mac_aq, mac_ellipsis, mac_invoc,
             mac_invoc_tt, mac_var, matcher, match_nonterminal, match_seq,
             match_tok, method, mode, module_ns, mt, mul, mutability,
             named_field, neg, noreturn, not, pat, pat_box, pat_enum,
             pat_ident, pat_lit, pat_range, pat_rec, pat_struct, pat_tup,
             pat_uniq, pat_wild, path, private, proto, proto_bare,
             proto_block, proto_box, proto_uniq, provided, public, pure_fn,
             purity, re_anon, re_named, region, rem, required, ret_style,
             return_val, self_ty, shl, shr, stmt, stmt_decl, stmt_expr,
             stmt_semi, struct_def, struct_field, struct_variant_kind,
             subtract, sty_box, sty_by_ref, sty_region, sty_static, sty_uniq,
             sty_value, token_tree, trait_method, trait_ref, tt_delim, tt_seq,
             tt_tok, tt_nonterminal, tuple_variant_kind, ty, ty_, ty_bot,
             ty_box, ty_field, ty_fn, ty_infer, ty_mac, ty_method, ty_nil,
             ty_param, ty_param_bound, ty_path, ty_ptr, ty_rec, ty_rptr,
             ty_tup, ty_u32, ty_uniq, ty_vec, ty_fixed_length, type_value_ns,
             unchecked_blk, uniq, unnamed_field, unsafe_blk, unsafe_fn,
             variant, view_item, view_item_, view_item_export,
             view_item_import, view_item_use, view_path, view_path_glob,
             view_path_list, view_path_simple, visibility, vstore, vstore_box,
             vstore_fixed, vstore_slice, vstore_uniq};

export file_type;
export parser;
export CRATE_FILE;
export SOURCE_FILE;

// FIXME (#1893): #ast expects to find this here but it's actually
// defined in `parse` Fixing this will be easier when we have export
// decls on individual items -- then parse can export this publicly, and
// everything else crate-visibly.
use parse::parse_from_source_str;
export parse_from_source_str;

export item_or_view_item, iovi_none, iovi_view_item, iovi_item;

enum restriction {
    UNRESTRICTED,
    RESTRICT_STMT_EXPR,
    RESTRICT_NO_CALL_EXPRS,
    RESTRICT_NO_BAR_OP,
    RESTRICT_NO_BAR_OR_DOUBLEBAR_OP,
}

enum file_type { CRATE_FILE, SOURCE_FILE, }


// We don't allow single-entry tuples in the true AST; that indicates a
// parenthesized expression.  However, we preserve them temporarily while
// parsing because `(while{...})+3` parses differently from `while{...}+3`.
//
// To reflect the fact that the @expr is not a true expr that should be
// part of the AST, we wrap such expressions in the pexpr enum.  They
// can then be converted to true expressions by a call to `to_expr()`.
enum pexpr {
    pexpr(@expr),
}

enum class_member {
    field_member(@struct_field),
    method_member(@method)
}

/*
  So that we can distinguish a class ctor or dtor
  from other class members
 */
enum class_contents { ctor_decl(fn_decl, ~[attribute], blk, codemap::span),
                      dtor_decl(blk, ~[attribute], codemap::span),
                      members(~[@class_member]) }

type arg_or_capture_item = Either<arg, capture_item>;
type item_info = (ident, item_, Option<~[attribute]>);

enum item_or_view_item {
    iovi_none,
    iovi_item(@item),
    iovi_view_item(@view_item)
}

enum view_item_parse_mode {
    VIEW_ITEMS_AND_ITEMS_ALLOWED,
    VIEW_ITEMS_ALLOWED,
    IMPORTS_AND_ITEMS_ALLOWED
}

/* The expr situation is not as complex as I thought it would be.
The important thing is to make sure that lookahead doesn't balk
at INTERPOLATED tokens */
macro_rules! maybe_whole_expr (
    ($p:expr) => { match copy $p.token {
      INTERPOLATED(token::nt_expr(e)) => {
        $p.bump();
        return pexpr(e);
      }
      INTERPOLATED(token::nt_path(pt)) => {
        $p.bump();
        return $p.mk_pexpr($p.span.lo, $p.span.lo,
                       expr_path(pt));
      }
      _ => ()
    }}
)

macro_rules! maybe_whole (
    ($p:expr, $constructor:ident) => { match copy $p.token {
      INTERPOLATED(token::$constructor(x)) => { $p.bump(); return x; }
      _ => ()
    }} ;
    (deref $p:expr, $constructor:ident) => { match copy $p.token {
      INTERPOLATED(token::$constructor(x)) => { $p.bump(); return *x; }
      _ => ()
    }} ;
    (Some $p:expr, $constructor:ident) => { match copy $p.token {
      INTERPOLATED(token::$constructor(x)) => { $p.bump(); return Some(x); }
      _ => ()
    }} ;
    (iovi $p:expr, $constructor:ident) => { match copy $p.token {
      INTERPOLATED(token::$constructor(x)) => {
        $p.bump();
        return iovi_item(x);
      }
      _ => ()
    }} ;
    (pair_empty $p:expr, $constructor:ident) => { match copy $p.token {
      INTERPOLATED(token::$constructor(x)) => { $p.bump(); return (~[], x); }
      _ => ()
    }}

)


pure fn maybe_append(+lhs: ~[attribute], rhs: Option<~[attribute]>)
                  -> ~[attribute] {
    match rhs {
        None => lhs,
        Some(attrs) => vec::append(lhs, attrs)
    }
}


/* ident is handled by common.rs */

fn parser(sess: parse_sess, cfg: ast::crate_cfg,
          +rdr: reader, ftype: file_type) -> parser {

    let tok0 = rdr.next_token();
    let span0 = tok0.sp;
    let interner = rdr.interner();

    parser {
        reader: move rdr,
        interner: move interner,
        sess: sess,
        cfg: cfg,
        file_type: ftype,
        token: tok0.tok,
        span: span0,
        last_span: span0,
        buffer: [mut
            {tok: tok0.tok, sp: span0},
            {tok: tok0.tok, sp: span0},
            {tok: tok0.tok, sp: span0},
            {tok: tok0.tok, sp: span0}
        ]/4,
        buffer_start: 0,
        buffer_end: 0,
        restriction: UNRESTRICTED,
        quote_depth: 0u,
        keywords: token::keyword_table(),
        restricted_keywords: token::restricted_keyword_table()
    }
}

struct parser {
    sess: parse_sess,
    cfg: crate_cfg,
    file_type: file_type,
    mut token: token::token,
    mut span: span,
    mut last_span: span,
    mut buffer: [mut {tok: token::token, sp: span}]/4,
    mut buffer_start: int,
    mut buffer_end: int,
    mut restriction: restriction,
    mut quote_depth: uint, // not (yet) related to the quasiquoter
    reader: reader,
    interner: interner<@~str>,
    keywords: hashmap<~str, ()>,
    restricted_keywords: hashmap<~str, ()>,

    drop {} /* do not copy the parser; its state is tied to outside state */

    fn bump() {
        self.last_span = self.span;
        let next = if self.buffer_start == self.buffer_end {
            self.reader.next_token()
        } else {
            let next = self.buffer[self.buffer_start];
            self.buffer_start = (self.buffer_start + 1) & 3;
            next
        };
        self.token = next.tok;
        self.span = next.sp;
    }
    fn swap(next: token::token, lo: uint, hi: uint) {
        self.token = next;
        self.span = mk_sp(lo, hi);
    }
    fn buffer_length() -> int {
        if self.buffer_start <= self.buffer_end {
            return self.buffer_end - self.buffer_start;
        }
        return (4 - self.buffer_start) + self.buffer_end;
    }
    fn look_ahead(distance: uint) -> token::token {
        let dist = distance as int;
        while self.buffer_length() < dist {
            self.buffer[self.buffer_end] = self.reader.next_token();
            self.buffer_end = (self.buffer_end + 1) & 3;
        }
        return copy self.buffer[(self.buffer_start + dist - 1) & 3].tok;
    }
    fn fatal(m: ~str) -> ! {
        self.sess.span_diagnostic.span_fatal(copy self.span, m)
    }
    fn span_fatal(sp: span, m: ~str) -> ! {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }
    fn span_note(sp: span, m: ~str) {
        self.sess.span_diagnostic.span_note(sp, m)
    }
    fn bug(m: ~str) -> ! {
        self.sess.span_diagnostic.span_bug(copy self.span, m)
    }
    fn warn(m: ~str) {
        self.sess.span_diagnostic.span_warn(copy self.span, m)
    }
    fn get_id() -> node_id { next_node_id(self.sess) }

    pure fn id_to_str(id: ident) -> @~str { self.sess.interner.get(id) }

    fn parse_ty_fn(purity: ast::purity) -> ty_ {
        let proto, bounds;
        if self.eat_keyword(~"extern") {
            self.expect_keyword(~"fn");
            proto = ast::proto_bare;
            bounds = @~[];
        } else {
            self.expect_keyword(~"fn");
            proto = self.parse_fn_ty_proto();
            bounds = self.parse_optional_ty_param_bounds();
        };
        ty_fn(proto, purity, bounds, self.parse_ty_fn_decl())
    }

    fn parse_ty_fn_decl() -> fn_decl {
        let inputs = do self.parse_unspanned_seq(
            token::LPAREN, token::RPAREN,
            seq_sep_trailing_disallowed(token::COMMA)) |p| {

            p.parse_arg_general(false)
        };
        let (ret_style, ret_ty) = self.parse_ret_ty();
        return {inputs: inputs, output: ret_ty,
                cf: ret_style};
    }

    fn parse_trait_methods() -> ~[trait_method] {
        do self.parse_unspanned_seq(token::LBRACE, token::RBRACE,
                                    seq_sep_none()) |p| {
            let attrs = p.parse_outer_attributes();
            let lo = p.span.lo;
            let is_static = p.parse_staticness();
            let static_sty = spanned(lo, p.span.hi, sty_static);

            let pur = p.parse_fn_purity();
            // NB: at the moment, trait methods are public by default; this
            // could change.
            let vis = p.parse_visibility();
            let ident = p.parse_method_name();

            let tps = p.parse_ty_params();

            let (self_ty, d, _) = do self.parse_fn_decl_with_self() |p| {
                // This is somewhat dubious; We don't want to allow argument
                // names to be left off if there is a definition...
                either::Left(p.parse_arg_general(false))
            };
            // XXX: Wrong. Shouldn't allow both static and self_ty
            let self_ty = if is_static { static_sty } else { self_ty };

            let hi = p.last_span.hi;
            debug!("parse_trait_methods(): trait method signature ends in \
                    `%s`",
                   token_to_str(p.reader, p.token));
            match p.token {
              token::SEMI => {
                p.bump();
                debug!("parse_trait_methods(): parsing required method");
                // NB: at the moment, visibility annotations on required
                // methods are ignored; this could change.
                required({ident: ident, attrs: attrs,
                          purity: pur, decl: d, tps: tps,
                          self_ty: self_ty,
                          id: p.get_id(), span: mk_sp(lo, hi)})
              }
              token::LBRACE => {
                debug!("parse_trait_methods(): parsing provided method");
                let (inner_attrs, body) =
                    p.parse_inner_attrs_and_block(true);
                let attrs = vec::append(attrs, inner_attrs);
                provided(@{ident: ident,
                           attrs: attrs,
                           tps: tps,
                           self_ty: self_ty,
                           purity: pur,
                           decl: d,
                           body: body,
                           id: p.get_id(),
                           span: mk_sp(lo, hi),
                           self_id: p.get_id(),
                           vis: vis})
              }

              _ => { p.fatal(~"expected `;` or `}` but found `" +
                          token_to_str(p.reader, p.token) + ~"`");
                }
            }
        }
    }


    fn parse_mt() -> mt {
        let mutbl = self.parse_mutability();
        let t = self.parse_ty(false);
        return {ty: t, mutbl: mutbl};
    }

    fn parse_ty_field() -> ty_field {
        let lo = self.span.lo;
        let mutbl = self.parse_mutability();
        let id = self.parse_ident();
        self.expect(token::COLON);
        let ty = self.parse_ty(false);
        return spanned(lo, ty.span.hi, {
            ident: id, mt: {ty: ty, mutbl: mutbl}
        });
    }

    fn parse_ret_ty() -> (ret_style, @ty) {
        return if self.eat(token::RARROW) {
            let lo = self.span.lo;
            if self.eat(token::NOT) {
                (noreturn, @{id: self.get_id(),
                             node: ty_bot,
                             span: mk_sp(lo, self.last_span.hi)})
            } else {
                (return_val, self.parse_ty(false))
            }
        } else {
            let pos = self.span.lo;
            (return_val, @{id: self.get_id(),
                           node: ty_nil,
                           span: mk_sp(pos, pos)})
        }
    }

    fn region_from_name(s: Option<ident>) -> @region {
        let r = match s {
          Some (id) => re_named(id),
          None => re_anon
        };

        @{id: self.get_id(), node: r}
    }

    // Parses something like "&x"
    fn parse_region() -> @region {
        self.expect(token::BINOP(token::AND));

        match copy self.token {
          token::IDENT(sid, _) => {
            self.bump();
            self.region_from_name(Some(sid))
          }
          _ => {
            self.region_from_name(None)
          }
        }
    }

    // Parses something like "&x/" (note the trailing slash)
    fn parse_region_with_sep() -> @region {
        let name =
            match copy self.token {
              token::IDENT(sid, _) => {
                if self.look_ahead(1u) == token::BINOP(token::SLASH) {
                    self.bump(); self.bump();
                    Some(sid)
                } else {
                    None
                }
              }
              _ => { None }
            };
        self.region_from_name(name)
    }

    fn parse_ty(colons_before_params: bool) -> @ty {
        maybe_whole!(self, nt_ty);

        let lo = self.span.lo;

        match self.maybe_parse_dollar_mac() {
          Some(e) => {
            return @{id: self.get_id(),
                  node: ty_mac(spanned(lo, self.span.hi, e)),
                  span: mk_sp(lo, self.span.hi)};
          }
          None => ()
        }

        let t = if self.token == token::LPAREN {
            self.bump();
            if self.token == token::RPAREN {
                self.bump();
                ty_nil
            } else {
                let mut ts = ~[self.parse_ty(false)];
                while self.token == token::COMMA {
                    self.bump();
                    vec::push(ts, self.parse_ty(false));
                }
                let t = if vec::len(ts) == 1u { ts[0].node }
                else { ty_tup(ts) };
                self.expect(token::RPAREN);
                t
            }
        } else if self.token == token::AT {
            self.bump();
            ty_box(self.parse_mt())
        } else if self.token == token::TILDE {
            self.bump();
            ty_uniq(self.parse_mt())
        } else if self.token == token::BINOP(token::STAR) {
            self.bump();
            ty_ptr(self.parse_mt())
        } else if self.token == token::LBRACE {
            let elems = self.parse_unspanned_seq(
                token::LBRACE, token::RBRACE,
                seq_sep_trailing_allowed(token::COMMA),
                |p| p.parse_ty_field());
            if vec::len(elems) == 0u {
                self.unexpected_last(token::RBRACE);
            }
            ty_rec(elems)
        } else if self.token == token::LBRACKET {
            self.expect(token::LBRACKET);
            let mut t = ty_vec(self.parse_mt());

            // Parse the `* 3` in `[ int * 3 ]`
            match self.maybe_parse_fixed_vstore_with_star() {
                None => {}
                Some(suffix) => {
                    t = ty_fixed_length(@{
                        id: self.get_id(),
                        node: t,
                        span: mk_sp(lo, self.last_span.hi)
                    }, suffix)
                }
            }
            self.expect(token::RBRACKET);
            t
        } else if self.token == token::BINOP(token::AND) {
            self.bump();
            let region = self.parse_region_with_sep();
            let mt = self.parse_mt();
            ty_rptr(region, mt)
        } else if self.eat_keyword(~"pure") {
            self.parse_ty_fn(ast::pure_fn)
        } else if self.eat_keyword(~"unsafe") {
            self.parse_ty_fn(ast::unsafe_fn)
        } else if self.is_keyword(~"fn") {
            self.parse_ty_fn(ast::impure_fn)
        } else if self.eat_keyword(~"extern") {
            self.expect_keyword(~"fn");
            ty_fn(proto_bare, ast::impure_fn, @~[], self.parse_ty_fn_decl())
        } else if self.token == token::MOD_SEP || is_ident(self.token) {
            let path = self.parse_path_with_tps(colons_before_params);
            ty_path(path, self.get_id())
        } else { self.fatal(~"expected type"); };

        let sp = mk_sp(lo, self.last_span.hi);
        return @{id: self.get_id(),
              node: match self.maybe_parse_fixed_vstore() {
                // Consider a fixed vstore suffix (/N or /_)
                None => t,
                Some(v) => {
                  ty_fixed_length(@{id: self.get_id(), node:t, span: sp}, v)
                } },
              span: sp}
    }

    fn parse_arg_mode() -> mode {
        if self.eat(token::BINOP(token::AND)) {
            expl(by_mutbl_ref)
        } else if self.eat(token::BINOP(token::MINUS)) {
            expl(by_move)
        } else if self.eat(token::ANDAND) {
            expl(by_ref)
        } else if self.eat(token::BINOP(token::PLUS)) {
            if self.eat(token::BINOP(token::PLUS)) {
                expl(by_val)
            } else {
                expl(by_copy)
            }
        } else { infer(self.get_id()) }
    }

    fn parse_capture_item_or(parse_arg_fn: fn(parser) -> arg_or_capture_item)
        -> arg_or_capture_item {

        fn parse_capture_item(p:parser, is_move: bool) -> capture_item {
            let sp = mk_sp(p.span.lo, p.span.hi);
            let ident = p.parse_ident();
            @{id: p.get_id(), is_move: is_move, name: ident, span: sp}
        }

        if self.eat_keyword(~"move") {
            either::Right(parse_capture_item(self, true))
        } else if self.eat_keyword(~"copy") {
            either::Right(parse_capture_item(self, false))
        } else {
            parse_arg_fn(self)
        }
    }

    // This version of parse arg doesn't necessarily require
    // identifier names.
    fn parse_arg_general(require_name: bool) -> arg {
        let m = self.parse_arg_mode();
        let i = if require_name {
            let name = self.parse_value_ident();
            self.expect(token::COLON);
            name
        } else {
            if is_plain_ident(self.token)
                && self.look_ahead(1u) == token::COLON {
                let name = self.parse_value_ident();
                self.bump();
                name
            } else { token::special_idents::invalid }
        };

        let t = self.parse_ty(false);

        {mode: m, ty: t, ident: i, id: self.get_id()}
    }

    fn parse_arg() -> arg_or_capture_item {
        either::Left(self.parse_arg_general(true))
    }

    fn parse_arg_or_capture_item() -> arg_or_capture_item {
        self.parse_capture_item_or(|p| p.parse_arg())
    }

    fn parse_fn_block_arg() -> arg_or_capture_item {
        do self.parse_capture_item_or |p| {
            let m = p.parse_arg_mode();
            let i = p.parse_value_ident();
            let t = if p.eat(token::COLON) {
                p.parse_ty(false)
            } else {
                @{id: p.get_id(),
                  node: ty_infer,
                  span: mk_sp(p.span.lo, p.span.hi)}
            };
            either::Left({mode: m, ty: t, ident: i, id: p.get_id()})
        }
    }

    fn maybe_parse_dollar_mac() -> Option<mac_> {
        match copy self.token {
          token::DOLLAR => {
            let lo = self.span.lo;
            self.bump();
            match copy self.token {
              token::LIT_INT_UNSUFFIXED(num) => {
                self.bump();
                Some(mac_var(num as uint))
              }
              token::LPAREN => {
                self.bump();
                let e = self.parse_expr();
                self.expect(token::RPAREN);
                let hi = self.last_span.hi;
                Some(mac_aq(mk_sp(lo,hi), e))
              }
              _ => {
                self.fatal(~"expected `(` or unsuffixed integer literal");
              }
            }
          }
          _ => None
        }
    }

    fn maybe_parse_fixed_vstore() -> Option<Option<uint>> {
        if self.token == token::BINOP(token::SLASH) {
            self.bump();
            match copy self.token {
              token::UNDERSCORE => {
                self.bump(); Some(None)
              }
              token::LIT_INT_UNSUFFIXED(i) if i >= 0i64 => {
                self.bump(); Some(Some(i as uint))
              }
              _ => None
            }
        } else {
            None
        }
    }

    fn maybe_parse_fixed_vstore_with_star() -> Option<Option<uint>> {
        if self.eat(token::BINOP(token::STAR)) {
            match copy self.token {
              token::UNDERSCORE => {
                self.bump(); Some(None)
              }
              token::LIT_INT_UNSUFFIXED(i) if i >= 0i64 => {
                self.bump(); Some(Some(i as uint))
              }
              _ => None
            }
        } else {
            None
        }
    }

    fn lit_from_token(tok: token::token) -> lit_ {
        match tok {
          token::LIT_INT(i, it) => lit_int(i, it),
          token::LIT_UINT(u, ut) => lit_uint(u, ut),
          token::LIT_INT_UNSUFFIXED(i) => lit_int_unsuffixed(i),
          token::LIT_FLOAT(s, ft) => lit_float(self.id_to_str(s), ft),
          token::LIT_STR(s) => lit_str(self.id_to_str(s)),
          token::LPAREN => { self.expect(token::RPAREN); lit_nil },
          _ => { self.unexpected_last(tok); }
        }
    }

    fn parse_lit() -> lit {
        let lo = self.span.lo;
        let lit = if self.eat_keyword(~"true") {
            lit_bool(true)
        } else if self.eat_keyword(~"false") {
            lit_bool(false)
        } else {
            let tok = self.token;
            self.bump();
            self.lit_from_token(tok)
        };
        return {node: lit, span: mk_sp(lo, self.last_span.hi)};
    }

    fn parse_path_without_tps() -> @path {
        self.parse_path_without_tps_(|p| p.parse_ident(),
                                     |p| p.parse_ident())
    }

    fn parse_path_without_tps_(
        parse_ident: fn(parser) -> ident,
        parse_last_ident: fn(parser) -> ident) -> @path {

        maybe_whole!(self, nt_path);
        let lo = self.span.lo;
        let global = self.eat(token::MOD_SEP);
        let mut ids = ~[];
        loop {
            let is_not_last =
                self.look_ahead(2u) != token::LT
                && self.look_ahead(1u) == token::MOD_SEP;

            if is_not_last {
                vec::push(ids, parse_ident(self));
                self.expect(token::MOD_SEP);
            } else {
                vec::push(ids, parse_last_ident(self));
                break;
            }
        }
        @{span: mk_sp(lo, self.last_span.hi), global: global,
          idents: ids, rp: None, types: ~[]}
    }

    fn parse_value_path() -> @path {
        self.parse_path_without_tps_(|p| p.parse_ident(),
                                     |p| p.parse_value_ident())
    }

    fn parse_path_with_tps(colons: bool) -> @path {
        debug!("parse_path_with_tps(colons=%b)", colons);

        maybe_whole!(self, nt_path);
        let lo = self.span.lo;
        let path = self.parse_path_without_tps();
        if colons && !self.eat(token::MOD_SEP) {
            return path;
        }

        // Parse the region parameter, if any, which will
        // be written "foo/&x"
        let rp = {
            // Hack: avoid parsing vstores like /@ and /~.  This is painful
            // because the notation for region bounds and the notation for
            // vstores is... um... the same.  I guess that's my fault.  This
            // is still not ideal as for &str we end up parsing more than we
            // ought to and have to sort it out later.
            if self.token == token::BINOP(token::SLASH)
                && self.look_ahead(1u) == token::BINOP(token::AND) {

                self.expect(token::BINOP(token::SLASH));
                Some(self.parse_region())
            } else {
                None
            }
        };

        // Parse any type parameters which may appear:
        let tps = {
            if self.token == token::LT {
                self.parse_seq_lt_gt(Some(token::COMMA),
                                     |p| p.parse_ty(false))
            } else {
                {node: ~[], span: path.span}
            }
        };

        return @{span: mk_sp(lo, tps.span.hi),
              rp: rp,
              types: tps.node,.. *path};
    }

    fn parse_mutability() -> mutability {
        if self.eat_keyword(~"mut") {
            m_mutbl
        } else if self.eat_keyword(~"const") {
            m_const
        } else {
            m_imm
        }
    }

    fn parse_field(sep: token::token) -> field {
        let lo = self.span.lo;
        let m = self.parse_mutability();
        let i = self.parse_ident();
        self.expect(sep);
        let e = self.parse_expr();
        return spanned(lo, e.span.hi, {mutbl: m, ident: i, expr: e});
    }

    fn mk_expr(lo: uint, hi: uint, +node: expr_) -> @expr {
        return @{id: self.get_id(), callee_id: self.get_id(),
              node: node, span: mk_sp(lo, hi)};
    }

    fn mk_mac_expr(lo: uint, hi: uint, m: mac_) -> @expr {
        return @{id: self.get_id(),
              callee_id: self.get_id(),
              node: expr_mac({node: m, span: mk_sp(lo, hi)}),
              span: mk_sp(lo, hi)};
    }

    fn mk_lit_u32(i: u32) -> @expr {
        let span = self.span;
        let lv_lit = @{node: lit_uint(i as u64, ty_u32),
                       span: span};

        return @{id: self.get_id(), callee_id: self.get_id(),
              node: expr_lit(lv_lit), span: span};
    }

    fn mk_pexpr(lo: uint, hi: uint, node: expr_) -> pexpr {
        return pexpr(self.mk_expr(lo, hi, node));
    }

    fn to_expr(e: pexpr) -> @expr {
        match e.node {
          expr_tup(es) if vec::len(es) == 1u => es[0u],
          _ => *e
        }
    }

    fn parse_bottom_expr() -> pexpr {
        maybe_whole_expr!(self);
        let lo = self.span.lo;
        let mut hi = self.span.hi;

        let mut ex: expr_;

        match self.maybe_parse_dollar_mac() {
          Some(x) => return pexpr(self.mk_mac_expr(lo, self.span.hi, x)),
          _ => ()
        }

        if self.token == token::LPAREN {
            self.bump();
            if self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = @spanned(lo, hi, lit_nil);
                return self.mk_pexpr(lo, hi, expr_lit(lit));
            }
            let mut es = ~[self.parse_expr()];
            while self.token == token::COMMA {
                self.bump(); vec::push(es, self.parse_expr());
            }
            hi = self.span.hi;
            self.expect(token::RPAREN);

            // Note: we retain the expr_tup() even for simple
            // parenthesized expressions, but only for a "little while".
            // This is so that wrappers around parse_bottom_expr()
            // can tell whether the expression was parenthesized or not,
            // which affects expr_is_complete().
            return self.mk_pexpr(lo, hi, expr_tup(es));
        } else if self.token == token::LBRACE {
            if self.looking_at_record_literal() {
                ex = self.parse_record_literal();
                hi = self.span.hi;
            } else {
                self.bump();
                let blk = self.parse_block_tail(lo, default_blk);
                return self.mk_pexpr(blk.span.lo, blk.span.hi,
                                     expr_block(blk));
            }
        } else if token::is_bar(self.token) {
            return pexpr(self.parse_lambda_expr());
        } else if self.eat_keyword(~"if") {
            return pexpr(self.parse_if_expr());
        } else if self.eat_keyword(~"for") {
            return pexpr(self.parse_sugary_call_expr(~"for", expr_loop_body));
        } else if self.eat_keyword(~"do") {
            return pexpr(self.parse_sugary_call_expr(~"do", expr_do_body));
        } else if self.eat_keyword(~"while") {
            return pexpr(self.parse_while_expr());
        } else if self.eat_keyword(~"again") || self.eat_keyword(~"loop") {
            return pexpr(self.parse_loop_expr());
        } else if self.eat_keyword(~"match") {
            return pexpr(self.parse_alt_expr());
        } else if self.eat_keyword(~"fn") {
            let proto = self.parse_fn_ty_proto();
            match proto {
              proto_bare => self.fatal(~"fn expr are deprecated, use fn@"),
              _ => { /* fallthrough */ }
            }
            return pexpr(self.parse_fn_expr(proto));
        } else if self.eat_keyword(~"unchecked") {
            return pexpr(self.parse_block_expr(lo, unchecked_blk));
        } else if self.eat_keyword(~"unsafe") {
            return pexpr(self.parse_block_expr(lo, unsafe_blk));
        } else if self.token == token::LBRACKET {
            self.bump();
            let mutbl = self.parse_mutability();
            if self.token == token::RBRACKET {
                // Empty vector.
                self.bump();
                ex = expr_vec(~[], mutbl);
            } else {
                // Nonempty vector.
                let first_expr = self.parse_expr();
                if self.token == token::COMMA &&
                        self.look_ahead(1) == token::DOTDOT {
                    // Repeating vector syntax: [ 0, ..512 ]
                    self.bump();
                    self.bump();
                    let count = self.parse_expr();
                    self.expect(token::RBRACKET);
                    ex = expr_repeat(first_expr, count, mutbl);
                } else if self.token == token::COMMA {
                    // Vector with two or more elements.
                    self.bump();
                    let remaining_exprs =
                        self.parse_seq_to_end(token::RBRACKET,
                            seq_sep_trailing_allowed(token::COMMA),
                            |p| p.parse_expr());
                    ex = expr_vec(~[first_expr] + remaining_exprs, mutbl);
                } else {
                    // Vector with one element.
                    self.expect(token::RBRACKET);
                    ex = expr_vec(~[first_expr], mutbl);
                }
            }
            hi = self.span.hi;
        } else if self.token == token::ELLIPSIS {
            self.bump();
            return pexpr(self.mk_mac_expr(lo, self.span.hi, mac_ellipsis));
        } else if self.token == token::POUND {
            let ex_ext = self.parse_syntax_ext();
            hi = ex_ext.span.hi;
            ex = ex_ext.node;
        } else if self.eat_keyword(~"fail") {
            if can_begin_expr(self.token) {
                let e = self.parse_expr();
                hi = e.span.hi;
                ex = expr_fail(Some(e));
            } else { ex = expr_fail(None); }
        } else if self.eat_keyword(~"log") {
            self.expect(token::LPAREN);
            let lvl = self.parse_expr();
            self.expect(token::COMMA);
            let e = self.parse_expr();
            ex = expr_log(ast::other, lvl, e);
            hi = self.span.hi;
            self.expect(token::RPAREN);
        } else if self.eat_keyword(~"assert") {
            let e = self.parse_expr();
            ex = expr_assert(e);
            hi = e.span.hi;
        } else if self.eat_keyword(~"return") {
            if can_begin_expr(self.token) {
                let e = self.parse_expr();
                hi = e.span.hi;
                ex = expr_ret(Some(e));
            } else { ex = expr_ret(None); }
        } else if self.eat_keyword(~"break") {
            if is_ident(self.token) {
                ex = expr_break(Some(self.parse_ident()));
            } else {
                ex = expr_break(None);
            }
            hi = self.span.hi;
        } else if self.eat_keyword(~"copy") {
            let e = self.parse_expr();
            ex = expr_copy(e);
            hi = e.span.hi;
        } else if self.eat_keyword(~"move") {
            let e = self.parse_expr();
            ex = expr_unary_move(e);
            hi = e.span.hi;
        } else if self.token == token::MOD_SEP ||
            is_ident(self.token) && !self.is_keyword(~"true") &&
            !self.is_keyword(~"false") {
            let pth = self.parse_path_with_tps(true);

            /* `!`, as an operator, is prefix, so we know this isn't that */
            if self.token == token::NOT {
                self.bump();
                let tts = self.parse_unspanned_seq(
                    token::LPAREN, token::RPAREN, seq_sep_none(),
                    |p| p.parse_token_tree());

                let hi = self.span.hi;

                return pexpr(self.mk_mac_expr(
                    lo, hi, mac_invoc_tt(pth, tts)));
            } else if self.token == token::LBRACE {
                // This might be a struct literal.
                if self.looking_at_record_literal() {
                    // It's a struct literal.
                    self.bump();
                    let mut fields = ~[];
                    vec::push(fields, self.parse_field(token::COLON));
                    while self.token != token::RBRACE {
                        self.expect(token::COMMA);
                        if self.token == token::RBRACE ||
                                self.token == token::DOTDOT {
                            // Accept an optional trailing comma.
                            break;
                        }
                        vec::push(fields, self.parse_field(token::COLON));
                    }

                    let base;
                    if self.eat(token::DOTDOT) {
                        base = Some(self.parse_expr());
                    } else {
                        base = None;
                    }

                    hi = pth.span.hi;
                    self.expect(token::RBRACE);
                    ex = expr_struct(pth, fields, base);
                    return self.mk_pexpr(lo, hi, ex);
                }
            }

            hi = pth.span.hi;
            ex = expr_path(pth);
        } else {
            let lit = self.parse_lit();
            hi = lit.span.hi;
            ex = expr_lit(@lit);
        }

        // Vstore is legal following expr_lit(lit_str(...)) and expr_vec(...)
        // only.
        match ex {
          expr_lit(@{node: lit_str(_), span: _}) |
          expr_vec(_, _)  => match self.maybe_parse_fixed_vstore() {
            None => (),
            Some(v) => {
                hi = self.span.hi;
                ex = expr_vstore(self.mk_expr(lo, hi, ex), vstore_fixed(v));
            }
          },
          _ => ()
        }

        return self.mk_pexpr(lo, hi, ex);
    }

    fn parse_block_expr(lo: uint, blk_mode: blk_check_mode) -> @expr {
        self.expect(token::LBRACE);
        let blk = self.parse_block_tail(lo, blk_mode);
        return self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk));
    }

    fn parse_syntax_ext() -> @expr {
        let lo = self.span.lo;
        self.expect(token::POUND);
        return self.parse_syntax_ext_naked(lo);
    }

    fn parse_syntax_ext_naked(lo: uint) -> @expr {
        match self.token {
          token::IDENT(_, _) => (),
          _ => self.fatal(~"expected a syntax expander name")
        }
        let pth = self.parse_path_without_tps();
        //temporary for a backwards-compatible cycle:
        let sep = seq_sep_trailing_disallowed(token::COMMA);
        let mut e = None;
        if (self.token == token::LPAREN || self.token == token::LBRACKET) {
            let lo = self.span.lo;
            let es =
                if self.token == token::LPAREN {
                    self.parse_unspanned_seq(token::LPAREN, token::RPAREN,
                                             sep, |p| p.parse_expr())
                } else {
                    self.parse_unspanned_seq(token::LBRACKET, token::RBRACKET,
                                             sep, |p| p.parse_expr())
                };
            let hi = self.span.hi;
            e = Some(self.mk_expr(lo, hi, expr_vec(es, m_imm)));
        }
        let mut b = None;
        if self.token == token::LBRACE {
            self.bump();
            let lo = self.span.lo;
            let mut depth = 1u;
            while (depth > 0u) {
                match (self.token) {
                  token::LBRACE => depth += 1u,
                  token::RBRACE => depth -= 1u,
                  token::EOF => self.fatal(~"unexpected EOF in macro body"),
                  _ => ()
                }
                self.bump();
            }
            let hi = self.last_span.lo;
            b = Some({span: mk_sp(lo,hi)});
        }
        return self.mk_mac_expr(lo, self.span.hi, mac_invoc(pth, e, b));
    }

    fn parse_dot_or_call_expr() -> pexpr {
        let b = self.parse_bottom_expr();
        self.parse_dot_or_call_expr_with(b)
    }

    fn permits_call() -> bool {
        return self.restriction != RESTRICT_NO_CALL_EXPRS;
    }

    fn parse_dot_or_call_expr_with(e0: pexpr) -> pexpr {
        let mut e = e0;
        let lo = e.span.lo;
        let mut hi;
        loop {
            // expr.f
            if self.eat(token::DOT) {
                match copy self.token {
                  token::IDENT(i, _) => {
                    hi = self.span.hi;
                    self.bump();
                    let tys = if self.eat(token::MOD_SEP) {
                        self.expect(token::LT);
                        self.parse_seq_to_gt(Some(token::COMMA),
                                             |p| p.parse_ty(false))
                    } else { ~[] };
                    e = self.mk_pexpr(lo, hi, expr_field(self.to_expr(e), i,
                                                         tys));
                  }
                  _ => self.unexpected()
                }
                again;
            }
            if self.expr_is_complete(e) { break; }
            match copy self.token {
              // expr(...)
              token::LPAREN if self.permits_call() => {
                let es = self.parse_unspanned_seq(
                    token::LPAREN, token::RPAREN,
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_expr());
                hi = self.span.hi;

                let nd = expr_call(self.to_expr(e), es, false);
                e = self.mk_pexpr(lo, hi, nd);
              }

              // expr[...]
              token::LBRACKET => {
                self.bump();
                let ix = self.parse_expr();
                hi = ix.span.hi;
                self.expect(token::RBRACKET);
                e = self.mk_pexpr(lo, hi, expr_index(self.to_expr(e), ix));
              }

              _ => return e
            }
        }
        return e;
    }

    fn parse_sep_and_zerok() -> (Option<token::token>, bool) {
        if self.token == token::BINOP(token::STAR)
            || self.token == token::BINOP(token::PLUS) {
            let zerok = self.token == token::BINOP(token::STAR);
            self.bump();
            return (None, zerok);
        } else {
            let sep = self.token;
            self.bump();
            if self.token == token::BINOP(token::STAR)
                || self.token == token::BINOP(token::PLUS) {
                let zerok = self.token == token::BINOP(token::STAR);
                self.bump();
                return (Some(sep), zerok);
            } else {
                self.fatal(~"expected `*` or `+`");
            }
        }
    }

    fn parse_token_tree() -> token_tree {
        maybe_whole!(deref self, nt_tt);

        fn parse_tt_tok(p: parser, delim_ok: bool) -> token_tree {
            match p.token {
              token::RPAREN | token::RBRACE | token::RBRACKET
              if !delim_ok => {
                p.fatal(~"incorrect close delimiter: `"
                           + token_to_str(p.reader, p.token) + ~"`");
              }
              token::EOF => {
                p.fatal(~"file ended in the middle of a macro invocation");
              }
              /* we ought to allow different depths of unquotation */
              token::DOLLAR if p.quote_depth > 0u => {
                p.bump();
                let sp = p.span;

                if p.token == token::LPAREN {
                    let seq = p.parse_seq(token::LPAREN, token::RPAREN,
                                          seq_sep_none(),
                                          |p| p.parse_token_tree());
                    let (s, z) = p.parse_sep_and_zerok();
                    return tt_seq(mk_sp(sp.lo ,p.span.hi), seq.node, s, z);
                } else {
                    return tt_nonterminal(sp, p.parse_ident());
                }
              }
              _ => { /* ok */ }
            }
            let res = tt_tok(p.span, p.token);
            p.bump();
            return res;
        }

        return match self.token {
          token::LPAREN | token::LBRACE | token::LBRACKET => {
            let ket = token::flip_delimiter(self.token);
            tt_delim(vec::append(
                ~[parse_tt_tok(self, true)],
                vec::append(
                    self.parse_seq_to_before_end(
                        ket, seq_sep_none(),
                        |p| p.parse_token_tree()),
                    ~[parse_tt_tok(self, true)])))
          }
          _ => parse_tt_tok(self, false)
        };
    }

    fn parse_matchers() -> ~[matcher] {
        // unification of matchers and token_trees would vastly improve
        // the interpolation of matchers
        maybe_whole!(self, nt_matchers);
        let name_idx = @mut 0u;
        return match self.token {
          token::LBRACE | token::LPAREN | token::LBRACKET => {
            self.parse_matcher_subseq(name_idx, copy self.token,
                                      token::flip_delimiter(self.token))
          }
          _ => self.fatal(~"expected open delimiter")
        }
    }


    // This goofy function is necessary to correctly match parens in matchers.
    // Otherwise, `$( ( )` would be a valid matcher, and `$( () )` would be
    // invalid. It's similar to common::parse_seq.
    fn parse_matcher_subseq(name_idx: @mut uint, bra: token::token,
                            ket: token::token) -> ~[matcher] {
        let mut ret_val = ~[];
        let mut lparens = 0u;

        self.expect(bra);

        while self.token != ket || lparens > 0u {
            if self.token == token::LPAREN { lparens += 1u; }
            if self.token == token::RPAREN { lparens -= 1u; }
            vec::push(ret_val, self.parse_matcher(name_idx));
        }

        self.bump();

        return ret_val;
    }

    fn parse_matcher(name_idx: @mut uint) -> matcher {
        let lo = self.span.lo;

        let m = if self.token == token::DOLLAR {
            self.bump();
            if self.token == token::LPAREN {
                let name_idx_lo = *name_idx;
                let ms = self.parse_matcher_subseq(name_idx, token::LPAREN,
                                                   token::RPAREN);
                if ms.len() == 0u {
                    self.fatal(~"repetition body must be nonempty");
                }
                let (sep, zerok) = self.parse_sep_and_zerok();
                match_seq(ms, sep, zerok, name_idx_lo, *name_idx)
            } else {
                let bound_to = self.parse_ident();
                self.expect(token::COLON);
                let nt_name = self.parse_ident();
                let m = match_nonterminal(bound_to, nt_name, *name_idx);
                *name_idx += 1u;
                m
            }
        } else {
            let m = match_tok(self.token);
            self.bump();
            m
        };

        return spanned(lo, self.span.hi, m);
    }


    fn parse_prefix_expr() -> pexpr {
        let lo = self.span.lo;
        let mut hi;

        let mut ex;
        match copy self.token {
          token::NOT => {
            self.bump();
            let e = self.to_expr(self.parse_prefix_expr());
            hi = e.span.hi;
            self.get_id(); // see ast_util::op_expr_callee_id
            ex = expr_unary(not, e);
          }
          token::BINOP(b) => {
            match b {
              token::MINUS => {
                self.bump();
                let e = self.to_expr(self.parse_prefix_expr());
                hi = e.span.hi;
                self.get_id(); // see ast_util::op_expr_callee_id
                ex = expr_unary(neg, e);
              }
              token::STAR => {
                self.bump();
                let e = self.to_expr(self.parse_prefix_expr());
                hi = e.span.hi;
                ex = expr_unary(deref, e);
              }
              token::AND => {
                self.bump();
                let m = self.parse_mutability();
                let e = self.to_expr(self.parse_prefix_expr());
                hi = e.span.hi;
                // HACK: turn &[...] into a &-evec
                ex = match e.node {
                  expr_vec(*) | expr_lit(@{node: lit_str(_), span: _})
                  if m == m_imm => {
                    expr_vstore(e, vstore_slice(self.region_from_name(None)))
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
            let e = self.to_expr(self.parse_prefix_expr());
            hi = e.span.hi;
            // HACK: turn @[...] into a @-evec
            ex = match e.node {
              expr_vec(*) | expr_lit(@{node: lit_str(_), span: _})
              if m == m_imm => expr_vstore(e, vstore_box),
              _ => expr_unary(box(m), e)
            };
          }
          token::TILDE => {
            self.bump();
            let m = self.parse_mutability();
            let e = self.to_expr(self.parse_prefix_expr());
            hi = e.span.hi;
            // HACK: turn ~[...] into a ~-evec
            ex = match e.node {
              expr_vec(*) | expr_lit(@{node: lit_str(_), span: _})
              if m == m_imm => expr_vstore(e, vstore_uniq),
              _ => expr_unary(uniq(m), e)
            };
          }
          _ => return self.parse_dot_or_call_expr()
        }
        return self.mk_pexpr(lo, hi, ex);
    }


    fn parse_binops() -> @expr {
        return self.parse_more_binops(self.parse_prefix_expr(), 0u);
    }

    fn parse_more_binops(plhs: pexpr, min_prec: uint) ->
        @expr {
        let lhs = self.to_expr(plhs);
        if self.expr_is_complete(plhs) { return lhs; }
        let peeked = self.token;
        if peeked == token::BINOP(token::OR) &&
            (self.restriction == RESTRICT_NO_BAR_OP ||
             self.restriction == RESTRICT_NO_BAR_OR_DOUBLEBAR_OP) {
            return lhs;
        }
        if peeked == token::OROR &&
            self.restriction == RESTRICT_NO_BAR_OR_DOUBLEBAR_OP {
            return lhs;
        }
        let cur_opt   = token_to_binop(peeked);
        match cur_opt {
          Some(cur_op) => {
            let cur_prec = operator_prec(cur_op);
            if cur_prec > min_prec {
                self.bump();
                let expr = self.parse_prefix_expr();
                let rhs = self.parse_more_binops(expr, cur_prec);
                self.get_id(); // see ast_util::op_expr_callee_id
                let bin = self.mk_pexpr(lhs.span.lo, rhs.span.hi,
                                        expr_binary(cur_op, lhs, rhs));
                return self.parse_more_binops(bin, min_prec);
            }
          }
          _ => ()
        }
        if as_prec > min_prec && self.eat_keyword(~"as") {
            let rhs = self.parse_ty(true);
            let _as =
                self.mk_pexpr(lhs.span.lo, rhs.span.hi, expr_cast(lhs, rhs));
            return self.parse_more_binops(_as, min_prec);
        }
        return lhs;
    }

    fn parse_assign_expr() -> @expr {
        let lo = self.span.lo;
        let lhs = self.parse_binops();
        match copy self.token {
          token::EQ => {
            self.bump();
            let rhs = self.parse_expr();
            return self.mk_expr(lo, rhs.span.hi, expr_assign(lhs, rhs));
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
            return self.mk_expr(lo, rhs.span.hi,
                                expr_assign_op(aop, lhs, rhs));
          }
          token::LARROW => {
            self.bump();
            let rhs = self.parse_expr();
            return self.mk_expr(lo, rhs.span.hi, expr_move(lhs, rhs));
          }
          token::DARROW => {
            self.bump();
            let rhs = self.parse_expr();
            return self.mk_expr(lo, rhs.span.hi, expr_swap(lhs, rhs));
          }
          _ => {/* fall through */ }
        }
        return lhs;
    }

    fn parse_if_expr() -> @expr {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let thn = self.parse_block();
        let mut els: Option<@expr> = None;
        let mut hi = thn.span.hi;
        if self.eat_keyword(~"else") {
            let elexpr = self.parse_else_expr();
            els = Some(elexpr);
            hi = elexpr.span.hi;
        }
        let q = {cond: cond, then: thn, els: els, lo: lo, hi: hi};
        return self.mk_expr(q.lo, q.hi, expr_if(q.cond, q.then, q.els));
    }

    fn parse_fn_expr(proto: proto) -> @expr {
        let lo = self.last_span.lo;

        // if we want to allow fn expression argument types to be inferred in
        // the future, just have to change parse_arg to parse_fn_block_arg.
        let (decl, capture_clause) =
            self.parse_fn_decl(|p| p.parse_arg_or_capture_item());

        let body = self.parse_block();
        return self.mk_expr(lo, body.span.hi,
                         expr_fn(proto, decl, body, capture_clause));
    }

    // `|args| { ... }` like in `do` expressions
    fn parse_lambda_block_expr() -> @expr {
        self.parse_lambda_expr_(
            || {
                match self.token {
                  token::BINOP(token::OR) | token::OROR => {
                    self.parse_fn_block_decl()
                  }
                  _ => {
                    // No argument list - `do foo {`
                    ({
                        {
                            inputs: ~[],
                            output: @{
                                id: self.get_id(),
                                node: ty_infer,
                                span: self.span
                            },
                            cf: return_val
                        }
                    },
                    @~[])
                  }
                }
            },
            || {
                let blk = self.parse_block();
                self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk))
            })
    }

    // `|args| expr`
    fn parse_lambda_expr() -> @expr {
        self.parse_lambda_expr_(|| self.parse_fn_block_decl(),
                                || self.parse_expr())
    }

    fn parse_lambda_expr_(parse_decl: fn&() -> (fn_decl, capture_clause),
                          parse_body: fn&() -> @expr) -> @expr {
        let lo = self.last_span.lo;
        let (decl, captures) = parse_decl();
        let body = parse_body();
        let fakeblock = {view_items: ~[], stmts: ~[], expr: Some(body),
                         id: self.get_id(), rules: default_blk};
        let fakeblock = spanned(body.span.lo, body.span.hi,
                                fakeblock);
        return self.mk_expr(lo, body.span.hi,
                         expr_fn_block(decl, fakeblock, captures));
    }

    fn parse_else_expr() -> @expr {
        if self.eat_keyword(~"if") {
            return self.parse_if_expr();
        } else {
            let blk = self.parse_block();
            return self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk));
        }
    }

    fn parse_sugary_call_expr(keyword: ~str,
                              ctor: fn(+@expr) -> expr_) -> @expr {
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
          expr_call(f, args, false) => {
            let block = self.parse_lambda_block_expr();
            let last_arg = self.mk_expr(block.span.lo, block.span.hi,
                                    ctor(block));
            let args = vec::append(args, ~[last_arg]);
            @{node: expr_call(f, args, true),
              .. *e}
          }
          expr_path(*) | expr_field(*) | expr_call(*) => {
            let block = self.parse_lambda_block_expr();
            let last_arg = self.mk_expr(block.span.lo, block.span.hi,
                                    ctor(block));
            self.mk_expr(lo.lo, last_arg.span.hi,
                         expr_call(e, ~[last_arg], true))
          }
          _ => {
            // There may be other types of expressions that can
            // represent the callee in `for` and `do` expressions
            // but they aren't represented by tests
            debug!("sugary call on %?", e.node);
            self.span_fatal(
                lo, fmt!("`%s` must be followed by a block call", keyword));
          }
        }
    }

    fn parse_while_expr() -> @expr {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let body = self.parse_block_no_value();
        let mut hi = body.span.hi;
        return self.mk_expr(lo, hi, expr_while(cond, body));
    }

    fn parse_loop_expr() -> @expr {
        // loop headers look like 'loop {' or 'loop unsafe {'
        let is_loop_header =
            self.token == token::LBRACE
            || (is_ident(copy self.token)
                && self.look_ahead(1) == token::LBRACE);
        // labeled loop headers look like 'loop foo: {'
        let is_labeled_loop_header =
            is_ident(self.token)
            && !self.is_any_keyword(copy self.token)
            && self.look_ahead(1) == token::COLON;

        if is_loop_header || is_labeled_loop_header {
            // This is a loop body
            let opt_ident;
            if is_labeled_loop_header {
                opt_ident = Some(self.parse_ident());
                self.expect(token::COLON);
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
            let ex = if is_ident(self.token) {
                expr_again(Some(self.parse_ident()))
            } else {
                expr_again(None)
            };
            let hi = self.span.hi;
            return self.mk_expr(lo, hi, ex);
        }
    }

    // For distingishing between record literals and blocks
    fn looking_at_record_literal() -> bool {
        let lookahead = self.look_ahead(1);
        self.token == token::LBRACE &&
            (self.token_is_keyword(~"mut", lookahead) ||
             (is_plain_ident(lookahead) &&
              self.look_ahead(2) == token::COLON))
    }

    fn parse_record_literal() -> expr_ {
        self.expect(token::LBRACE);
        let mut fields = ~[self.parse_field(token::COLON)];
        let mut base = None;
        while self.token != token::RBRACE {
            if self.token == token::COMMA
                && self.look_ahead(1) == token::DOTDOT {
                self.bump();
                self.bump();
                base = Some(self.parse_expr()); break;
            }

            self.expect(token::COMMA);
            if self.token == token::RBRACE {
                // record ends by an optional trailing comma
                break;
            }
            vec::push(fields, self.parse_field(token::COLON));
        }
        self.expect(token::RBRACE);
        return expr_rec(fields, base);
    }

    fn parse_alt_expr() -> @expr {
        let lo = self.last_span.lo;
        let discriminant = self.parse_expr();
        self.expect(token::LBRACE);
        let mut arms: ~[arm] = ~[];
        while self.token != token::RBRACE {
            let pats = self.parse_pats();
            let mut guard = None;
            if self.eat_keyword(~"if") { guard = Some(self.parse_expr()); }
            self.expect(token::FAT_ARROW);
            let expr = self.parse_expr_res(RESTRICT_STMT_EXPR);

            let require_comma =
                !classify::expr_is_simple_block(expr)
                && self.token != token::RBRACE;

            if require_comma {
                self.expect(token::COMMA);
            } else {
                self.eat(token::COMMA);
            }

            let blk = {node: {view_items: ~[],
                              stmts: ~[],
                              expr: Some(expr),
                              id: self.get_id(),
                              rules: default_blk},
                       span: expr.span};

            vec::push(arms, {pats: pats, guard: guard, body: blk});
        }
        let mut hi = self.span.hi;
        self.bump();
        return self.mk_expr(lo, hi, expr_match(discriminant, arms));
    }

    fn parse_expr() -> @expr {
        return self.parse_expr_res(UNRESTRICTED);
    }

    fn parse_expr_res(r: restriction) -> @expr {
        let old = self.restriction;
        self.restriction = r;
        let e = self.parse_assign_expr();
        self.restriction = old;
        return e;
    }

    fn parse_initializer() -> Option<initializer> {
        match self.token {
          token::EQ => {
            self.bump();
            return Some({op: init_assign, expr: self.parse_expr()});
          }
          token::LARROW => {
            self.bump();
            return Some({op: init_move, expr: self.parse_expr()});
          }
          // Now that the the channel is the first argument to receive,
          // combining it with an initializer doesn't really make sense.
          // case (token::RECV) {
          //     self.bump();
          //     return Some(rec(op = init_recv,
          //                  expr = self.parse_expr()));
          // }
          _ => {
            return None;
          }
        }
    }

    fn parse_pats() -> ~[@pat] {
        let mut pats = ~[];
        loop {
            vec::push(pats, self.parse_pat(true));
            if self.token == token::BINOP(token::OR) { self.bump(); }
            else { return pats; }
        };
    }

    fn parse_pat_fields(refutable: bool) -> (~[ast::field_pat], bool) {
        let mut fields = ~[];
        let mut etc = false;
        let mut first = true;
        while self.token != token::RBRACE {
            if first { first = false; }
            else { self.expect(token::COMMA); }

            if self.token == token::UNDERSCORE {
                self.bump();
                if self.token != token::RBRACE {
                    self.fatal(~"expected `}`, found `" +
                               token_to_str(self.reader, self.token) +
                               ~"`");
                }
                etc = true;
                break;
            }

            let lo1 = self.last_span.lo;
            let fieldname = if self.look_ahead(1u) == token::COLON {
                self.parse_ident()
            } else {
                self.parse_value_ident()
            };
            let hi1 = self.last_span.lo;
            let fieldpath = ast_util::ident_to_path(mk_sp(lo1, hi1),
                                                    fieldname);
            let mut subpat;
            if self.token == token::COLON {
                self.bump();
                subpat = self.parse_pat(refutable);
            } else {
                subpat = @{
                    id: self.get_id(),
                    node: pat_ident(bind_by_implicit_ref,
                                    fieldpath,
                                    None),
                    span: self.last_span
                };
            }
            vec::push(fields, {ident: fieldname, pat: subpat});
        }
        return (fields, etc);
    }

    fn parse_pat(refutable: bool) -> @pat {
        maybe_whole!(self, nt_pat);

        let lo = self.span.lo;
        let mut hi = self.span.hi;
        let mut pat;
        match self.token {
          token::UNDERSCORE => { self.bump(); pat = pat_wild; }
          token::AT => {
            self.bump();
            let sub = self.parse_pat(refutable);
            hi = sub.span.hi;
            // HACK: parse @"..." as a literal of a vstore @str
            pat = match sub.node {
              pat_lit(e@@{
                node: expr_lit(@{node: lit_str(_), span: _}), _
              }) => {
                let vst = @{id: self.get_id(), callee_id: self.get_id(),
                            node: expr_vstore(e, vstore_box),
                            span: mk_sp(lo, hi)};
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
              pat_lit(e@@{
                node: expr_lit(@{node: lit_str(_), span: _}), _
              }) => {
                let vst = @{id: self.get_id(), callee_id: self.get_id(),
                            node: expr_vstore(e, vstore_uniq),
                            span: mk_sp(lo, hi)};
                pat_lit(vst)
              }
              _ => pat_uniq(sub)
            };

          }
          token::LBRACE => {
            self.bump();
            let (fields, etc) = self.parse_pat_fields(refutable);
            hi = self.span.hi;
            self.bump();
            pat = pat_rec(fields, etc);
          }
          token::LPAREN => {
            self.bump();
            if self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = @{node: lit_nil, span: mk_sp(lo, hi)};
                let expr = self.mk_expr(lo, hi, expr_lit(lit));
                pat = pat_lit(expr);
            } else {
                let mut fields = ~[self.parse_pat(refutable)];
                while self.token == token::COMMA {
                    self.bump();
                    vec::push(fields, self.parse_pat(refutable));
                }
                if vec::len(fields) == 1u { self.expect(token::COMMA); }
                hi = self.span.hi;
                self.expect(token::RPAREN);
                pat = pat_tup(fields);
            }
          }
          tok => {
            if !is_ident_or_path(tok)
                || self.is_keyword(~"true")
                || self.is_keyword(~"false")
            {
                let val = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                if self.eat(token::DOTDOT) {
                    let end = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                    pat = pat_range(val, end);
                } else {
                    pat = pat_lit(val);
                }
            } else if self.eat_keyword(~"ref") {
                let mutbl = self.parse_mutability();
                pat = self.parse_pat_ident(refutable, bind_by_ref(mutbl));
            } else if self.eat_keyword(~"copy") {
                pat = self.parse_pat_ident(refutable, bind_by_value);
            } else if self.eat_keyword(~"move") {
                pat = self.parse_pat_ident(refutable, bind_by_move);
            } else if !is_plain_ident(self.token) {
                pat = self.parse_enum_variant(refutable);
            } else {
                let binding_mode;
                // XXX: Aren't these two cases deadcode? -- bblum
                if self.eat_keyword(~"copy") {
                    binding_mode = bind_by_value;
                } else if self.eat_keyword(~"move") {
                    binding_mode = bind_by_move;
                } else if refutable {
                    // XXX: Should be bind_by_value, but that's not
                    // backward compatible.
                    binding_mode = bind_by_implicit_ref;
                } else {
                    binding_mode = bind_by_value;
                }

                let cannot_be_enum_or_struct;
                match self.look_ahead(1) {
                    token::LPAREN | token::LBRACKET | token::LT |
                    token::LBRACE =>
                        cannot_be_enum_or_struct = false,
                    _ =>
                        cannot_be_enum_or_struct = true
                }

                if is_plain_ident(self.token) && cannot_be_enum_or_struct {
                    let name = self.parse_value_path();
                    let sub;
                    if self.eat(token::AT) {
                        sub = Some(self.parse_pat(refutable));
                    } else {
                        sub = None;
                    };
                    pat = pat_ident(binding_mode, name, sub);
                } else {
                    let enum_path = self.parse_path_with_tps(true);
                    match self.token {
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
                            match self.token {
                              token::LPAREN => match self.look_ahead(1u) {
                                token::BINOP(token::STAR) => {
                                    // This is a "top constructor only" pat
                                      self.bump(); self.bump();
                                      star_pat = true;
                                      self.expect(token::RPAREN);
                                  }
                                _ => {
                                    args = self.parse_unspanned_seq(
                                        token::LPAREN, token::RPAREN,
                                        seq_sep_trailing_disallowed
                                            (token::COMMA),
                                        |p| p.parse_pat(refutable));
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
        return @{id: self.get_id(), node: pat, span: mk_sp(lo, hi)};
    }

    fn parse_pat_ident(refutable: bool,
                       binding_mode: ast::binding_mode) -> ast::pat_ {
        if !is_plain_ident(self.token) {
            self.span_fatal(
                copy self.last_span,
                ~"expected identifier, found path");
        }
        let name = self.parse_value_path();
        let sub = if self.eat(token::AT) {
            Some(self.parse_pat(refutable))
        } else { None };

        // just to be friendly, if they write something like
        //   ref Some(i)
        // we end up here with ( as the current token.  This shortly
        // leads to a parse error.  Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to parse_enum_variant()
        if self.token == token::LPAREN {
            self.span_fatal(
                copy self.last_span,
                ~"expected identifier, found enum pattern");
        }

        pat_ident(binding_mode, name, sub)
    }

    fn parse_enum_variant(refutable: bool) -> ast::pat_ {
        let enum_path = self.parse_path_with_tps(true);
        match self.token {
          token::LPAREN => {
            match self.look_ahead(1u) {
              token::BINOP(token::STAR) => { // foo(*)
                self.expect(token::LPAREN);
                self.expect(token::BINOP(token::STAR));
                self.expect(token::RPAREN);
                pat_enum(enum_path, None)
              }
              _ => { // foo(a, ..., z)
                let args = self.parse_unspanned_seq(
                    token::LPAREN, token::RPAREN,
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_pat(refutable));
                pat_enum(enum_path, Some(args))
              }
            }
          }
          _ => { // option::None
            pat_enum(enum_path, Some(~[]))
          }
        }
    }

    fn parse_local(is_mutbl: bool,
                   allow_init: bool) -> @local {
        let lo = self.span.lo;
        let pat = self.parse_pat(false);
        let mut ty = @{id: self.get_id(),
                       node: ty_infer,
                       span: mk_sp(lo, lo)};
        if self.eat(token::COLON) { ty = self.parse_ty(false); }
        let init = if allow_init { self.parse_initializer() } else { None };
        return @spanned(lo, self.last_span.hi,
                     {is_mutbl: is_mutbl, ty: ty, pat: pat,
                      init: init, id: self.get_id()});
    }

    fn parse_let() -> @decl {
        let is_mutbl = self.eat_keyword(~"mut");
        let lo = self.span.lo;
        let mut locals = ~[self.parse_local(is_mutbl, true)];
        while self.eat(token::COMMA) {
            vec::push(locals, self.parse_local(is_mutbl, true));
        }
        return @spanned(lo, self.last_span.hi, decl_local(locals));
    }

    /* assumes "let" token has already been consumed */
    fn parse_instance_var(pr: visibility) -> @class_member {
        let mut is_mutbl = class_immutable;
        let lo = self.span.lo;
        if self.eat_keyword(~"mut") {
            is_mutbl = class_mutable;
        }
        if !is_plain_ident(self.token) {
            self.fatal(~"expected ident");
        }
        let name = self.parse_ident();
        self.expect(token::COLON);
        let ty = self.parse_ty(false);
        return @field_member(@spanned(lo, self.last_span.hi, {
            kind: named_field(name, is_mutbl, pr),
            id: self.get_id(),
            ty: ty
        }));
    }

    fn parse_stmt(+first_item_attrs: ~[attribute]) -> @stmt {
        maybe_whole!(self, nt_stmt);

        fn check_expected_item(p: parser, current_attrs: ~[attribute]) {
            // If we have attributes then we should have an item
            if vec::is_not_empty(current_attrs) {
                p.fatal(~"expected item");
            }
        }

        let lo = self.span.lo;
        if self.is_keyword(~"let") {
            check_expected_item(self, first_item_attrs);
            self.expect_keyword(~"let");
            let decl = self.parse_let();
            return @spanned(lo, decl.span.hi, stmt_decl(decl, self.get_id()));
        } else {
            let mut item_attrs;
            match self.parse_outer_attrs_or_ext(first_item_attrs) {
              None => item_attrs = ~[],
              Some(Left(attrs)) => item_attrs = attrs,
              Some(Right(ext)) => {
                return @spanned(lo, ext.span.hi,
                                stmt_expr(ext, self.get_id()));
              }
            }

            let item_attrs = vec::append(first_item_attrs, item_attrs);

            match self.parse_item_or_view_item(item_attrs, true) {
              iovi_item(i) => {
                let mut hi = i.span.hi;
                let decl = @spanned(lo, hi, decl_item(i));
                return @spanned(lo, hi, stmt_decl(decl, self.get_id()));
              }
              iovi_view_item(vi) => {
                self.span_fatal(vi.span, ~"view items must be declared at \
                                           the top of the block");
              }
              iovi_none() => { /* fallthrough */ }
            }

            check_expected_item(self, item_attrs);

            // Remainder are line-expr stmts.
            let e = self.parse_expr_res(RESTRICT_STMT_EXPR);
            return @spanned(lo, e.span.hi, stmt_expr(e, self.get_id()));
        }
    }

    fn expr_is_complete(e: pexpr) -> bool {
        return self.restriction == RESTRICT_STMT_EXPR &&
            !classify::expr_requires_semi_to_be_stmt(*e);
    }

    fn parse_block() -> blk {
        let (attrs, blk) = self.parse_inner_attrs_and_block(false);
        assert vec::is_empty(attrs);
        return blk;
    }

    fn parse_inner_attrs_and_block(parse_attrs: bool)
        -> (~[attribute], blk) {

        maybe_whole!(pair_empty self, nt_block);

        fn maybe_parse_inner_attrs_and_next(p: parser, parse_attrs: bool) ->
            {inner: ~[attribute], next: ~[attribute]} {
            if parse_attrs {
                p.parse_inner_attrs_and_next()
            } else {
                {inner: ~[], next: ~[]}
            }
        }

        let lo = self.span.lo;
        if self.eat_keyword(~"unchecked") {
            self.expect(token::LBRACE);
            let {inner, next} = maybe_parse_inner_attrs_and_next(self,
                                                                 parse_attrs);
            return (inner, self.parse_block_tail_(lo, unchecked_blk, next));
        } else if self.eat_keyword(~"unsafe") {
            self.expect(token::LBRACE);
            let {inner, next} = maybe_parse_inner_attrs_and_next(self,
                                                                 parse_attrs);
            return (inner, self.parse_block_tail_(lo, unsafe_blk, next));
        } else {
            self.expect(token::LBRACE);
            let {inner, next} = maybe_parse_inner_attrs_and_next(self,
                                                                 parse_attrs);
            return (inner, self.parse_block_tail_(lo, default_blk, next));
        }
    }

    fn parse_block_no_value() -> blk {
        // We parse blocks that cannot have a value the same as any other
        // block; the type checker will make sure that the tail expression (if
        // any) has unit type.
        return self.parse_block();
    }

    // Precondition: already parsed the '{' or '#{'
    // I guess that also means "already parsed the 'impure'" if
    // necessary, and this should take a qualifier.
    // some blocks start with "#{"...
    fn parse_block_tail(lo: uint, s: blk_check_mode) -> blk {
        self.parse_block_tail_(lo, s, ~[])
    }

    fn parse_block_tail_(lo: uint, s: blk_check_mode,
                         +first_item_attrs: ~[attribute]) -> blk {
        let mut stmts = ~[];
        let mut expr = None;

        let {attrs_remaining, view_items, items: items} =
            self.parse_items_and_view_items(first_item_attrs,
                                            IMPORTS_AND_ITEMS_ALLOWED);

        for items.each |item| {
            let decl = @spanned(item.span.lo, item.span.hi, decl_item(item));
            push(stmts, @spanned(item.span.lo, item.span.hi,
                                 stmt_decl(decl, self.get_id())));
        }

        let mut initial_attrs = attrs_remaining;

        if self.token == token::RBRACE && !vec::is_empty(initial_attrs) {
            self.fatal(~"expected item");
        }

        while self.token != token::RBRACE {
            match self.token {
              token::SEMI => {
                self.bump(); // empty
              }
              _ => {
                let stmt = self.parse_stmt(initial_attrs);
                initial_attrs = ~[];
                match stmt.node {
                  stmt_expr(e, stmt_id) => { // Expression without semicolon:
                    match self.token {
                      token::SEMI => {
                        self.bump();
                        push(stmts,
                             @{node: stmt_semi(e, stmt_id),.. *stmt});
                      }
                      token::RBRACE => {
                        expr = Some(e);
                      }
                      t => {
                        if classify::stmt_ends_with_semi(*stmt) {
                            self.fatal(~"expected `;` or `}` after \
                                         expression but found `"
                                       + token_to_str(self.reader, t) + ~"`");
                        }
                        vec::push(stmts, stmt);
                      }
                    }
                  }

                  _ => { // All other kinds of statements:
                    vec::push(stmts, stmt);

                    if classify::stmt_ends_with_semi(*stmt) {
                        self.expect(token::SEMI);
                    }
                  }
                }
              }
            }
        }
        let mut hi = self.span.hi;
        self.bump();
        let bloc = {view_items: view_items, stmts: stmts, expr: expr,
                    id: self.get_id(), rules: s};
        return spanned(lo, hi, bloc);
    }

    fn parse_optional_ty_param_bounds() -> @~[ty_param_bound] {
        let mut bounds = ~[];
        if self.eat(token::COLON) {
            while is_ident(self.token) {
                if self.eat_keyword(~"send") {
                    push(bounds, bound_send); }
                else if self.eat_keyword(~"copy") {
                    push(bounds, bound_copy) }
                else if self.eat_keyword(~"const") {
                    push(bounds, bound_const);
                } else if self.eat_keyword(~"owned") {
                    push(bounds, bound_owned);
                } else if is_ident(self.token) {
                    // XXX: temporary until kinds become traits
                    let maybe_bound = match self.token {
                      token::IDENT(sid, _) => {
                        match *self.id_to_str(sid) {
                          ~"Send" => Some(bound_send),
                          ~"Copy" => Some(bound_copy),
                          ~"Const" => Some(bound_const),
                          ~"Owned" => Some(bound_owned),
                          _ => None
                        }
                      }
                      _ => fail
                    };

                    match maybe_bound {
                      Some(bound) => {
                        self.bump();
                        push(bounds, bound);
                      }
                      None => {
                        push(bounds, bound_trait(self.parse_ty(false)));
                      }
                    }
                } else {
                    push(bounds, bound_trait(self.parse_ty(false)));
                }
            }
        }
        return @move bounds;
    }

    fn parse_ty_param() -> ty_param {
        let ident = self.parse_ident();
        let bounds = self.parse_optional_ty_param_bounds();
        return {ident: ident, id: self.get_id(), bounds: bounds};
    }

    fn parse_ty_params() -> ~[ty_param] {
        if self.eat(token::LT) {
            self.parse_seq_to_gt(Some(token::COMMA), |p| p.parse_ty_param())
        } else { ~[] }
    }

    fn parse_fn_decl(parse_arg_fn: fn(parser) -> arg_or_capture_item)
        -> (fn_decl, capture_clause) {

        let args_or_capture_items: ~[arg_or_capture_item] =
            self.parse_unspanned_seq(
                token::LPAREN, token::RPAREN,
                seq_sep_trailing_disallowed(token::COMMA), parse_arg_fn);

        let inputs = either::lefts(args_or_capture_items);
        let capture_clause = @either::rights(args_or_capture_items);

        let (ret_style, ret_ty) = self.parse_ret_ty();
        return ({inputs: inputs,
                 output: ret_ty,
                 cf: ret_style}, capture_clause);
    }

    fn is_self_ident() -> bool {
        match self.token {
          token::IDENT(id, false) if id == token::special_idents::self_
            => true,
          _ => false
        }
    }

    fn expect_self_ident() {
        if !self.is_self_ident() {
            self.fatal(#fmt("expected `self` but found `%s`",
                            token_to_str(self.reader, self.token)));
        }
        self.bump();
    }

    fn parse_fn_decl_with_self(parse_arg_fn:
                                    fn(parser) -> arg_or_capture_item)
                            -> (self_ty, fn_decl, capture_clause) {

        fn maybe_parse_self_ty(cnstr: fn(+mutability) -> ast::self_ty_,
                               p: parser) -> ast::self_ty_ {
            // We need to make sure it isn't a mode or a type
            if p.token_is_keyword(~"self", p.look_ahead(1)) ||
                ((p.token_is_keyword(~"const", p.look_ahead(1)) ||
                  p.token_is_keyword(~"mut", p.look_ahead(1))) &&
                 p.token_is_keyword(~"self", p.look_ahead(2))) {

                p.bump();
                let mutability = p.parse_mutability();
                p.expect_self_ident();
                cnstr(mutability)
            } else {
                sty_by_ref
            }
        }

        self.expect(token::LPAREN);

        // A bit of complexity and lookahead is needed here in order to to be
        // backwards compatible.
        let lo = self.span.lo;
        let self_ty = match copy self.token {
          token::BINOP(token::AND) => {
            maybe_parse_self_ty(sty_region, self)
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
            sty_by_ref
          }
        };

        // If we parsed a self type, expect a comma before the argument list.
        let args_or_capture_items;
        if self_ty != sty_by_ref {
            match copy self.token {
                token::COMMA => {
                    self.bump();
                    let sep = seq_sep_trailing_disallowed(token::COMMA);
                    args_or_capture_items =
                        self.parse_seq_to_before_end(token::RPAREN,
                                                     sep,
                                                     parse_arg_fn);
                }
                token::RPAREN => {
                    args_or_capture_items = ~[];
                }
                _ => {
                    self.fatal(~"expected `,` or `)`, found `" +
                               token_to_str(self.reader, self.token) + ~"`");
                }
            }
        } else {
            let sep = seq_sep_trailing_disallowed(token::COMMA);
            args_or_capture_items =
                self.parse_seq_to_before_end(token::RPAREN,
                                             sep,
                                             parse_arg_fn);
        }

        self.expect(token::RPAREN);

        let hi = self.span.hi;

        let inputs = either::lefts(args_or_capture_items);
        let capture_clause = @either::rights(args_or_capture_items);
        let (ret_style, ret_ty) = self.parse_ret_ty();

        let fn_decl = {
            inputs: inputs,
            output: ret_ty,
            cf: ret_style
        };

        (spanned(lo, hi, self_ty), fn_decl, capture_clause)
    }

    fn parse_fn_block_decl() -> (fn_decl, capture_clause) {
        let inputs_captures = {
            if self.eat(token::OROR) {
                ~[]
            } else {
                self.parse_unspanned_seq(
                    token::BINOP(token::OR), token::BINOP(token::OR),
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_fn_block_arg())
            }
        };
        let output = if self.eat(token::RARROW) {
            self.parse_ty(false)
        } else {
            @{id: self.get_id(), node: ty_infer, span: self.span}
        };
        return ({inputs: either::lefts(inputs_captures),
                 output: output,
                 cf: return_val},
                @either::rights(inputs_captures));
    }

    fn parse_fn_header() -> {ident: ident, tps: ~[ty_param]} {
        let id = self.parse_value_ident();
        let ty_params = self.parse_ty_params();
        return {ident: id, tps: ty_params};
    }

    fn mk_item(lo: uint, hi: uint, +ident: ident,
               +node: item_, vis: visibility,
               +attrs: ~[attribute]) -> @item {
        return @{ident: ident,
              attrs: attrs,
              id: self.get_id(),
              node: node,
              vis: vis,
              span: mk_sp(lo, hi)};
    }

    fn parse_item_fn(purity: purity) -> item_info {
        let t = self.parse_fn_header();
        let (decl, _) = self.parse_fn_decl(|p| p.parse_arg());
        let (inner_attrs, body) = self.parse_inner_attrs_and_block(true);
        (t.ident, item_fn(decl, purity, t.tps, body), Some(inner_attrs))
    }

    fn parse_method_name() -> ident {
        self.parse_value_ident()
    }

    fn parse_method(pr: visibility) -> @method {
        let attrs = self.parse_outer_attributes();
        let lo = self.span.lo;

        let is_static = self.parse_staticness();
        let static_sty = spanned(lo, self.span.hi, sty_static);

        let pur = self.parse_fn_purity();
        let ident = self.parse_method_name();
        let tps = self.parse_ty_params();
        let (self_ty, decl, _) = do self.parse_fn_decl_with_self() |p| {
            p.parse_arg()
        };
        // XXX: interaction between staticness, self_ty is broken now
        let self_ty = if is_static { static_sty} else { self_ty };

        let (inner_attrs, body) = self.parse_inner_attrs_and_block(true);
        let attrs = vec::append(attrs, inner_attrs);
        @{ident: ident, attrs: attrs,
          tps: tps, self_ty: self_ty, purity: pur, decl: decl,
          body: body, id: self.get_id(), span: mk_sp(lo, body.span.hi),
          self_id: self.get_id(), vis: pr}
    }

    fn parse_item_trait() -> item_info {
        let ident = self.parse_ident();
        self.parse_region_param();
        let tps = self.parse_ty_params();

        // Parse traits, if necessary.
        let traits;
        if self.token == token::COLON {
            self.bump();
            traits = self.parse_trait_ref_list(token::LBRACE);
        } else {
            traits = ~[];
        }

        let meths = self.parse_trait_methods();
        (ident, item_trait(tps, traits, meths), None)
    }

    // Parses four variants (with the region/type params always optional):
    //    impl<T> ~[T] : to_str { ... }
    fn parse_item_impl() -> item_info {
        fn wrap_path(p: parser, pt: @path) -> @ty {
            @{id: p.get_id(), node: ty_path(pt, p.get_id()), span: pt.span}
        }

        // We do two separate paths here: old-style impls and new-style impls.

        // First, parse type parameters if necessary.
        let mut tps;
        if self.token == token::LT {
            tps = self.parse_ty_params();
        } else {
            tps = ~[];
        }

        // This is a new-style impl declaration.
        // XXX: clownshoes
        let ident = token::special_idents::clownshoes_extensions;

        // Parse the type.
        let ty = self.parse_ty(false);


        // Parse traits, if necessary.
        let traits = if self.token == token::COLON {
            self.bump();
            self.parse_trait_ref_list(token::LBRACE)
        } else {
            ~[]
        };

        let mut meths = ~[];
        self.expect(token::LBRACE);
        while !self.eat(token::RBRACE) {
            let vis = self.parse_visibility();
            vec::push(meths, self.parse_method(vis));
        }
        (ident, item_impl(tps, traits, ty, meths), None)
    }

    // Instantiates ident <i> with references to <typarams> as arguments.
    // Used to create a path that refers to a class which will be defined as
    // the return type of the ctor function.
    fn ident_to_path_tys(i: ident,
                         typarams: ~[ty_param]) -> @path {
        let s = self.last_span;

        @{span: s, global: false, idents: ~[i],
          rp: None,
          types: vec::map(typarams, |tp| {
              @{id: self.get_id(),
                node: ty_path(ident_to_path(s, tp.ident), self.get_id()),
                span: s}})
         }
    }

    fn parse_trait_ref() -> @trait_ref {
        @{path: self.parse_path_with_tps(false),
          ref_id: self.get_id(), impl_id: self.get_id()}
    }

    fn parse_trait_ref_list(ket: token::token) -> ~[@trait_ref] {
        self.parse_seq_to_before_end(
            ket, seq_sep_trailing_disallowed(token::COMMA),
            |p| p.parse_trait_ref())
    }

    fn parse_item_class() -> item_info {
        let class_name = self.parse_value_ident();
        self.parse_region_param();
        let ty_params = self.parse_ty_params();
        let traits : ~[@trait_ref] = if self.eat(token::COLON)
            { self.parse_trait_ref_list(token::LBRACE) }
        else { ~[] };

        let mut fields: ~[@struct_field];
        let mut methods: ~[@method] = ~[];
        let mut the_ctor: Option<(fn_decl, ~[attribute], blk, codemap::span)>
            = None;
        let mut the_dtor: Option<(blk, ~[attribute], codemap::span)> = None;
        let ctor_id = self.get_id();

        if self.eat(token::LBRACE) {
            // It's a record-like struct.
            fields = ~[];
            while self.token != token::RBRACE {
                match self.parse_class_item() {
                  ctor_decl(a_fn_decl, attrs, blk, s) => {
                      match the_ctor {
                        Some((_, _, _, s_first)) => {
                          self.span_note(s, #fmt("Duplicate constructor \
                                     declaration for class %s",
                                     *self.interner.get(class_name)));
                           self.span_fatal(copy s_first, ~"First constructor \
                                                          declared here");
                        }
                        None    => {
                          the_ctor = Some((a_fn_decl, attrs, blk, s));
                        }
                      }
                  }
                  dtor_decl(blk, attrs, s) => {
                      match the_dtor {
                        Some((_, _, s_first)) => {
                          self.span_note(s, #fmt("Duplicate destructor \
                                     declaration for class %s",
                                     *self.interner.get(class_name)));
                          self.span_fatal(copy s_first, ~"First destructor \
                                                          declared here");
                        }
                        None => {
                          the_dtor = Some((blk, attrs, s));
                        }
                      }
                  }
                  members(mms) => {
                    for mms.each |mm| {
                        match mm {
                            @field_member(struct_field) =>
                                vec::push(fields, struct_field),
                            @method_member(the_method_member) =>
                                vec::push(methods, the_method_member)
                        }
                    }
                  }
                }
            }
            self.bump();
        } else if self.token == token::LPAREN {
            // It's a tuple-like struct.
            fields = do self.parse_unspanned_seq(token::LPAREN, token::RPAREN,
                                                 seq_sep_trailing_allowed
                                                    (token::COMMA)) |p| {
                let lo = p.span.lo;
                let struct_field_ = {
                    kind: unnamed_field,
                    id: self.get_id(),
                    ty: p.parse_ty(false)
                };
                @spanned(lo, p.span.hi, struct_field_)
            };
            self.expect(token::SEMI);
        } else if self.eat(token::SEMI) {
            // It's a unit-like struct.
            fields = ~[];
        } else {
            self.fatal(fmt!("expected `{`, `(`, or `;` after struct name \
                             but found `%s`",
                            token_to_str(self.reader, self.token)));
        }

        let actual_dtor = do option::map(the_dtor) |dtor| {
            let (d_body, d_attrs, d_s) = dtor;
            {node: {id: self.get_id(),
                    attrs: d_attrs,
                    self_id: self.get_id(),
                    body: d_body},
             span: d_s}};
        match the_ctor {
          Some((ct_d, ct_attrs, ct_b, ct_s)) => {
            (class_name,
             item_class(@{
                traits: traits,
                fields: move fields,
                methods: move methods,
                ctor: Some({
                 node: {id: ctor_id,
                        attrs: ct_attrs,
                        self_id: self.get_id(),
                        dec: ct_d,
                        body: ct_b},
                 span: ct_s}),
                dtor: actual_dtor
             }, ty_params),
             None)
          }
          None => {
            (class_name,
             item_class(@{
                    traits: traits,
                    fields: move fields,
                    methods: move methods,
                    ctor: None,
                    dtor: actual_dtor
             }, ty_params),
             None)
          }
        }
    }

    fn token_is_pound_or_doc_comment(++tok: token::token) -> bool {
        match tok {
            token::POUND | token::DOC_COMMENT(_) => true,
            _ => false
        }
    }

    fn parse_single_class_item(vis: visibility) -> @class_member {
        if (self.token_is_keyword(~"mut", copy self.token) ||
                !self.is_any_keyword(copy self.token)) &&
                !self.token_is_pound_or_doc_comment(self.token) {
            let a_var = self.parse_instance_var(vis);
            match self.token {
                token::SEMI | token::COMMA => {
                    self.bump();
                }
                token::RBRACE => {}
                _ => {
                    self.span_fatal(copy self.span,
                                    fmt!("expected `;`, `,`, or '}' but \
                                          found `%s`",
                                         token_to_str(self.reader,
                                                      self.token)));
                }
            }
            return a_var;
        } else {
            let m = self.parse_method(vis);
            return @method_member(m);
        }
    }

    fn parse_dtor(attrs: ~[attribute]) -> class_contents {
        let lo = self.last_span.lo;
        let body = self.parse_block();
        dtor_decl(body, attrs, mk_sp(lo, self.last_span.hi))
    }

    fn parse_class_item() -> class_contents {
        if self.eat_keyword(~"priv") {
            // XXX: Remove after snapshot.
            match self.token {
                token::LBRACE => {
                    self.bump();
                    let mut results = ~[];
                    while self.token != token::RBRACE {
                        vec::push(results,
                                  self.parse_single_class_item(private));
                    }
                    self.bump();
                    return members(results);
                }
                _ =>
                   return members(~[self.parse_single_class_item(private)])
            }
        }

        if self.eat_keyword(~"pub") {
           return members(~[self.parse_single_class_item(public)]);
        }

        let attrs = self.parse_outer_attributes();

        if self.eat_keyword(~"drop") {
           return self.parse_dtor(attrs);
        }
        else {
           return members(~[self.parse_single_class_item(inherited)]);
        }
    }

    fn parse_visibility() -> visibility {
        if self.eat_keyword(~"pub") { public }
        else if self.eat_keyword(~"priv") { private }
        else { inherited }
    }
    fn parse_staticness() -> bool {
        self.eat_keyword(~"static")
    }

    fn parse_mod_items(term: token::token,
                       +first_item_attrs: ~[attribute]) -> _mod {
        // Shouldn't be any view items since we've already parsed an item attr
        let {attrs_remaining, view_items, items: starting_items} =
            self.parse_items_and_view_items(first_item_attrs,
                                            VIEW_ITEMS_AND_ITEMS_ALLOWED);
        let mut items: ~[@item] = move starting_items;

        let mut first = true;
        while self.token != term {
            let mut attrs = self.parse_outer_attributes();
            if first {
                attrs = vec::append(attrs_remaining, attrs);
                first = false;
            }
            debug!("parse_mod_items: parse_item_or_view_item(attrs=%?)",
                   attrs);
            match self.parse_item_or_view_item(attrs, true) {
              iovi_item(item) => vec::push(items, item),
              iovi_view_item(view_item) => {
                self.span_fatal(view_item.span, ~"view items must be \
                                                  declared at the top of the \
                                                  module");
              }
              _ => {
                self.fatal(~"expected item but found `" +
                           token_to_str(self.reader, self.token) + ~"`");
              }
            }
            debug!("parse_mod_items: attrs=%?", attrs);
        }

        if first && attrs_remaining.len() > 0u {
            // We parsed attributes for the first item but didn't find it
            self.fatal(~"expected item");
        }

        return {view_items: view_items, items: items};
    }

    fn parse_item_const() -> item_info {
        let id = self.parse_value_ident();
        self.expect(token::COLON);
        let ty = self.parse_ty(false);
        self.expect(token::EQ);
        let e = self.parse_expr();
        self.expect(token::SEMI);
        (id, item_const(ty, e), None)
    }

    fn parse_item_mod() -> item_info {
        let id = self.parse_ident();
        self.expect(token::LBRACE);
        let inner_attrs = self.parse_inner_attrs_and_next();
        let m = self.parse_mod_items(token::RBRACE, inner_attrs.next);
        self.expect(token::RBRACE);
        (id, item_mod(m), Some(inner_attrs.inner))
    }

    fn parse_item_foreign_fn(+attrs: ~[attribute]) -> @foreign_item {
        let lo = self.span.lo;
        let purity = self.parse_fn_purity();
        let t = self.parse_fn_header();
        let (decl, _) = self.parse_fn_decl(|p| p.parse_arg());
        let mut hi = self.span.hi;
        self.expect(token::SEMI);
        return @{ident: t.ident,
                 attrs: attrs,
                 node: foreign_item_fn(decl, purity, t.tps),
                 id: self.get_id(),
                 span: mk_sp(lo, hi)};
    }

    fn parse_item_foreign_const(+attrs: ~[attribute]) -> @foreign_item {
        let lo = self.span.lo;
        self.expect_keyword(~"const");
        let ident = self.parse_ident();
        self.expect(token::COLON);
        let ty = self.parse_ty(false);
        let hi = self.span.hi;
        self.expect(token::SEMI);
        return @{ident: ident,
                 attrs: attrs,
                 node: foreign_item_const(move ty),
                 id: self.get_id(),
                 span: mk_sp(lo, hi)};
    }

    fn parse_fn_purity() -> purity {
        if self.eat_keyword(~"fn") { impure_fn }
        else if self.eat_keyword(~"pure") {
            self.expect_keyword(~"fn");
            pure_fn
        } else if self.eat_keyword(~"unsafe") {
            self.expect_keyword(~"fn");
            unsafe_fn
        }
        else { self.unexpected(); }
    }

    fn parse_foreign_item(+attrs: ~[attribute]) -> @foreign_item {
        if self.is_keyword(~"const") {
            self.parse_item_foreign_const(move attrs)
        } else {
            self.parse_item_foreign_fn(move attrs)
        }
    }

    fn parse_foreign_mod_items(sort: ast::foreign_mod_sort,
                               +first_item_attrs: ~[attribute]) ->
        foreign_mod {
        // Shouldn't be any view items since we've already parsed an item attr
        let {attrs_remaining, view_items, items: _} =
            self.parse_items_and_view_items(first_item_attrs,
                                            VIEW_ITEMS_ALLOWED);

        let mut items: ~[@foreign_item] = ~[];
        let mut initial_attrs = attrs_remaining;
        while self.token != token::RBRACE {
            let attrs = vec::append(initial_attrs,
                                    self.parse_outer_attributes());
            initial_attrs = ~[];
            vec::push(items, self.parse_foreign_item(attrs));
        }
        return {sort: sort, view_items: view_items,
             items: items};
    }

    fn parse_item_foreign_mod(lo: uint,
                              visibility: visibility,
                              attrs: ~[attribute],
                              items_allowed: bool)
                           -> item_or_view_item {

        let mut must_be_named_mod = false;
        if self.is_keyword(~"mod") {
            must_be_named_mod = true;
            self.expect_keyword(~"mod");
        } else if self.is_keyword(~"module") {
            must_be_named_mod = true;
            self.expect_keyword(~"module");
        } else if self.token != token::LBRACE {
            self.span_fatal(copy self.span,
                            fmt!("expected `{` or `mod` but found %s",
                                 token_to_str(self.reader, self.token)));
        }

        let (sort, ident) = match self.token {
            token::IDENT(*) => (ast::named, self.parse_ident()),
            _ => {
                if must_be_named_mod {
                    self.span_fatal(copy self.span,
                                    fmt!("expected foreign module name but \
                                          found %s",
                                         token_to_str(self.reader,
                                                      self.token)));
                }

                (ast::anonymous,
                 token::special_idents::clownshoes_foreign_mod)
            }
        };

        // extern mod { ... }
        if items_allowed && self.eat(token::LBRACE) {
            let extra_attrs = self.parse_inner_attrs_and_next();
            let m = self.parse_foreign_mod_items(sort, extra_attrs.next);
            self.expect(token::RBRACE);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident,
                                          item_foreign_mod(m), visibility,
                                          maybe_append(attrs,
                                                       Some(extra_attrs.
                                                            inner))));
        }

        // extern mod foo;
        let metadata = self.parse_optional_meta();
        self.expect(token::SEMI);
        return iovi_view_item(@{
            node: view_item_use(ident, metadata, self.get_id()),
            attrs: attrs,
            vis: visibility,
            span: mk_sp(lo, self.last_span.hi)
        });
    }

    fn parse_type_decl() -> {lo: uint, ident: ident} {
        let lo = self.last_span.lo;
        let id = self.parse_ident();
        return {lo: lo, ident: id};
    }

    fn parse_item_type() -> item_info {
        let t = self.parse_type_decl();
        self.parse_region_param();
        let tps = self.parse_ty_params();
        self.expect(token::EQ);
        let ty = self.parse_ty(false);
        self.expect(token::SEMI);
        (t.ident, item_ty(ty, tps), None)
    }

    fn parse_region_param() {
        if self.eat(token::BINOP(token::SLASH)) {
            self.expect(token::BINOP(token::AND));
        }
    }

    fn parse_struct_def() -> @struct_def {
        let mut the_dtor: Option<(blk, ~[attribute], codemap::span)> = None;
        let mut fields: ~[@struct_field] = ~[];
        let mut methods: ~[@method] = ~[];
        while self.token != token::RBRACE {
            match self.parse_class_item() {
                ctor_decl(*) => {
                    self.span_fatal(copy self.span,
                                    ~"deprecated explicit \
                                      constructors are not allowed \
                                      here");
                }
                dtor_decl(blk, attrs, s) => {
                    match the_dtor {
                        Some((_, _, s_first)) => {
                            self.span_note(s, ~"duplicate destructor \
                                                declaration");
                            self.span_fatal(copy s_first,
                                            ~"first destructor \
                                              declared here");
                        }
                        None => {
                            the_dtor = Some((blk, attrs, s));
                        }
                    }
                }
                members(mms) => {
                    for mms.each |mm| {
                        match mm {
                            @field_member(struct_field) =>
                                vec::push(fields, struct_field),
                            @method_member(the_method_member) =>
                                vec::push(methods, the_method_member)
                        }
                    }
                }
            }
        }
        self.bump();
        let mut actual_dtor = do option::map(the_dtor) |dtor| {
            let (d_body, d_attrs, d_s) = dtor;
            {node: {id: self.get_id(),
                    attrs: d_attrs,
                    self_id: self.get_id(),
                    body: d_body},
             span: d_s}
        };

        return @{
            traits: ~[],
            fields: move fields,
            methods: move methods,
            ctor: None,
            dtor: actual_dtor
        };
    }

    fn parse_enum_def(ty_params: ~[ast::ty_param])
                   -> enum_def {
        let mut variants: ~[variant] = ~[];
        let mut all_nullary = true, have_disr = false;
        let mut common_fields = None;

        while self.token != token::RBRACE {
            let variant_attrs = self.parse_outer_attributes();
            let vlo = self.span.lo;

            // Is this a common field declaration?
            if self.eat_keyword(~"struct") {
                if common_fields.is_some() {
                    self.fatal(~"duplicate declaration of shared fields");
                }
                self.expect(token::LBRACE);
                common_fields = Some(self.parse_struct_def());
                again;
            }

            let vis = self.parse_visibility();

            // Is this a nested enum declaration?
            let ident, needs_comma, kind;
            let mut args = ~[], disr_expr = None;
            if self.eat_keyword(~"enum") {
                ident = self.parse_ident();
                self.expect(token::LBRACE);
                let nested_enum_def = self.parse_enum_def(ty_params);
                kind = enum_variant_kind(move nested_enum_def);
                needs_comma = false;
            } else {
                ident = self.parse_value_ident();
                if self.eat(token::LBRACE) {
                    // Parse a struct variant.
                    all_nullary = false;
                    kind = struct_variant_kind(self.parse_struct_def());
                } else if self.token == token::LPAREN {
                    all_nullary = false;
                    let arg_tys = self.parse_unspanned_seq(
                        token::LPAREN, token::RPAREN,
                        seq_sep_trailing_disallowed(token::COMMA),
                        |p| p.parse_ty(false));
                    for arg_tys.each |ty| {
                        vec::push(args, {ty: ty, id: self.get_id()});
                    }
                    kind = tuple_variant_kind(args);
                } else if self.eat(token::EQ) {
                    have_disr = true;
                    disr_expr = Some(self.parse_expr());
                    kind = tuple_variant_kind(args);
                } else {
                    kind = tuple_variant_kind(~[]);
                }
                needs_comma = true;
            }

            let vr = {name: ident, attrs: variant_attrs,
                      kind: kind, id: self.get_id(),
                      disr_expr: disr_expr, vis: vis};
            vec::push(variants, spanned(vlo, self.last_span.hi, vr));

            if needs_comma && !self.eat(token::COMMA) { break; }
        }
        self.expect(token::RBRACE);
        if (have_disr && !all_nullary) {
            self.fatal(~"discriminator values can only be used with a c-like \
                        enum");
        }

        return enum_def({ variants: variants, common: common_fields });
    }

    fn parse_item_enum() -> item_info {
        let id = self.parse_ident();
        self.parse_region_param();
        let ty_params = self.parse_ty_params();
        // Newtype syntax
        if self.token == token::EQ {
            self.check_restricted_keywords_(*self.id_to_str(id));
            self.bump();
            let ty = self.parse_ty(false);
            self.expect(token::SEMI);
            let variant =
                spanned(ty.span.lo, ty.span.hi,
                        {name: id,
                         attrs: ~[],
                         kind: tuple_variant_kind
                            (~[{ty: ty, id: self.get_id()}]),
                         id: self.get_id(),
                         disr_expr: None,
                         vis: public});
            return (id, item_enum(enum_def({ variants: ~[variant],
                                             common: None }),
                                  ty_params), None);
        }
        self.expect(token::LBRACE);

        let enum_definition = self.parse_enum_def(ty_params);
        (id, item_enum(enum_definition, ty_params), None)
    }

    fn parse_fn_ty_proto() -> proto {
        match self.token {
          token::AT => {
            self.bump();
            proto_box
          }
          token::TILDE => {
            self.bump();
            proto_uniq
          }
          token::BINOP(token::AND) => {
            self.bump();
            proto_block
          }
          _ => {
            proto_block
          }
        }
    }

    fn fn_expr_lookahead(tok: token::token) -> bool {
        match tok {
          token::LPAREN | token::AT | token::TILDE | token::BINOP(_) => true,
          _ => false
        }
    }

    fn parse_item_or_view_item(+attrs: ~[attribute], items_allowed: bool)
                            -> item_or_view_item {
        maybe_whole!(iovi self,nt_item);
        let lo = self.span.lo;

        let visibility;
        if self.eat_keyword(~"pub") {
            visibility = public;
        } else if self.eat_keyword(~"priv") {
            visibility = private;
        } else {
            visibility = inherited;
        }

        if items_allowed && self.eat_keyword(~"const") {
            let (ident, item_, extra_attrs) = self.parse_item_const();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed &&
            self.is_keyword(~"fn") &&
            !self.fn_expr_lookahead(self.look_ahead(1u)) {
            self.bump();
            let (ident, item_, extra_attrs) = self.parse_item_fn(impure_fn);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed && self.eat_keyword(~"pure") {
            self.expect_keyword(~"fn");
            let (ident, item_, extra_attrs) = self.parse_item_fn(pure_fn);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed && self.is_keyword(~"unsafe")
            && self.look_ahead(1u) != token::LBRACE {
            self.bump();
            self.expect_keyword(~"fn");
            let (ident, item_, extra_attrs) = self.parse_item_fn(unsafe_fn);
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if self.eat_keyword(~"extern") {
            if items_allowed && self.eat_keyword(~"fn") {
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(extern_fn);
                return iovi_item(self.mk_item(lo, self.last_span.hi, ident,
                                              item_, visibility,
                                              maybe_append(attrs,
                                                           extra_attrs)));
            }
            return self.parse_item_foreign_mod(lo, visibility, attrs,
                                               items_allowed);
        } else if items_allowed && (self.eat_keyword(~"mod") ||
                                    self.eat_keyword(~"module")) {
            let (ident, item_, extra_attrs) = self.parse_item_mod();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed && self.eat_keyword(~"type") {
            let (ident, item_, extra_attrs) = self.parse_item_type();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed && self.eat_keyword(~"enum") {
            let (ident, item_, extra_attrs) = self.parse_item_enum();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed && self.eat_keyword(~"trait") {
            let (ident, item_, extra_attrs) = self.parse_item_trait();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed && self.eat_keyword(~"impl") {
            let (ident, item_, extra_attrs) = self.parse_item_impl();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if items_allowed && self.eat_keyword(~"struct") {
            let (ident, item_, extra_attrs) = self.parse_item_class();
            return iovi_item(self.mk_item(lo, self.last_span.hi, ident, item_,
                                          visibility,
                                          maybe_append(attrs, extra_attrs)));
        } else if self.eat_keyword(~"use") {
            let view_item = self.parse_use(visibility);
            self.expect(token::SEMI);
            return iovi_view_item(@{
                node: view_item,
                attrs: attrs,
                vis: visibility,
                span: mk_sp(lo, self.last_span.hi)
            });
        } else if self.eat_keyword(~"import") {
            let view_paths = self.parse_view_paths();
            self.expect(token::SEMI);
            return iovi_view_item(@{
                node: view_item_import(view_paths),
                attrs: attrs,
                vis: visibility,
                span: mk_sp(lo, self.last_span.hi)
            });
        } else if self.eat_keyword(~"export") {
            let view_paths = self.parse_view_paths();
            self.expect(token::SEMI);
            return iovi_view_item(@{
                node: view_item_export(view_paths),
                attrs: attrs,
                vis: visibility,
                span: mk_sp(lo, self.last_span.hi)
            });
        } else if items_allowed && (!self.is_any_keyword(copy self.token)
                && self.look_ahead(1) == token::NOT
                && is_plain_ident(self.look_ahead(2))) {
            // item macro.
            let pth = self.parse_path_without_tps();
            self.expect(token::NOT);
            let id = self.parse_ident();
            let tts = self.parse_unspanned_seq(
                token::LPAREN, token::RPAREN, seq_sep_none(),
                |p| p.parse_token_tree());
            let m = ast::mac_invoc_tt(pth, tts);
            let m: ast::mac = {node: m,
                               span: {lo: self.span.lo,
                                      hi: self.span.hi,
                                      expn_info: None}};
            let item_ = item_mac(m);
            return iovi_item(self.mk_item(lo, self.last_span.hi, id, item_,
                                          visibility, attrs));
        } else {
            return iovi_none;
        };
    }

    fn parse_item(+attrs: ~[attribute]) -> Option<@ast::item> {
        match self.parse_item_or_view_item(attrs, true) {
            iovi_none =>
                None,
            iovi_view_item(_) =>
                self.fatal(~"view items are not allowed here"),
            iovi_item(item) =>
                Some(item)
        }
    }

    fn parse_use(vis: visibility) -> view_item_ {
        if vis != public && (self.look_ahead(1) == token::SEMI ||
                             self.look_ahead(1) == token::LPAREN) {
            // Old-style "use"; i.e. what we now call "extern mod".
            let ident = self.parse_ident();
            let metadata = self.parse_optional_meta();
            return view_item_use(ident, metadata, self.get_id());
        }

        return view_item_import(self.parse_view_paths());
    }

    fn parse_view_path() -> @view_path {
        let lo = self.span.lo;

        let namespace;
        if self.eat_keyword(~"mod") {
            namespace = module_ns;
        } else {
            namespace = type_value_ns;
        }

        let first_ident = self.parse_ident();
        let mut path = ~[first_ident];
        debug!("parsed view_path: %s", *self.id_to_str(first_ident));
        match self.token {
          token::EQ => {
            // x = foo::bar
            self.bump();
            path = ~[self.parse_ident()];
            while self.token == token::MOD_SEP {
                self.bump();
                let id = self.parse_ident();
                vec::push(path, id);
            }
            let path = @{span: mk_sp(lo, self.span.hi), global: false,
                         idents: path, rp: None, types: ~[]};
            return @spanned(lo, self.span.hi,
                         view_path_simple(first_ident, path, namespace,
                                          self.get_id()));
          }

          token::MOD_SEP => {
            // foo::bar or foo::{a,b,c} or foo::*
            while self.token == token::MOD_SEP {
                self.bump();

                match copy self.token {

                  token::IDENT(i, _) => {
                    self.bump();
                    vec::push(path, i);
                  }

                  // foo::bar::{a,b,c}
                  token::LBRACE => {
                    let idents = self.parse_unspanned_seq(
                        token::LBRACE, token::RBRACE,
                        seq_sep_trailing_allowed(token::COMMA),
                        |p| p.parse_path_list_ident());
                    let path = @{span: mk_sp(lo, self.span.hi),
                                 global: false, idents: path,
                                 rp: None, types: ~[]};
                    return @spanned(lo, self.span.hi,
                                 view_path_list(path, idents, self.get_id()));
                  }

                  // foo::bar::*
                  token::BINOP(token::STAR) => {
                    self.bump();
                    let path = @{span: mk_sp(lo, self.span.hi),
                                 global: false, idents: path,
                                 rp: None, types: ~[]};
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
        let path = @{span: mk_sp(lo, self.span.hi), global: false,
                     idents: path, rp: None, types: ~[]};
        return @spanned(lo, self.span.hi,
                     view_path_simple(last, path, namespace, self.get_id()));
    }

    fn parse_view_paths() -> ~[@view_path] {
        let mut vp = ~[self.parse_view_path()];
        while self.token == token::COMMA {
            self.bump();
            vec::push(vp, self.parse_view_path());
        }
        return vp;
    }

    fn is_view_item() -> bool {
        let tok, next_tok;
        if !self.is_keyword(~"pub") && !self.is_keyword(~"priv") {
            tok = self.token;
            next_tok = self.look_ahead(1);
        } else {
            tok = self.look_ahead(1);
            next_tok = self.look_ahead(2);
        };
        self.token_is_keyword(~"use", tok)
            || self.token_is_keyword(~"import", tok)
            || self.token_is_keyword(~"export", tok)
            || (self.token_is_keyword(~"extern", tok) &&
                self.token_is_keyword(~"mod", next_tok))
    }

    fn parse_view_item(+attrs: ~[attribute]) -> @view_item {
        let lo = self.span.lo, vis = self.parse_visibility();
        let node = if self.eat_keyword(~"use") {
            self.parse_use(vis)
        } else if self.eat_keyword(~"import") {
            view_item_import(self.parse_view_paths())
        } else if self.eat_keyword(~"export") {
            view_item_export(self.parse_view_paths())
        } else if self.eat_keyword(~"extern") {
            self.expect_keyword(~"mod");
            let ident = self.parse_ident();
            let metadata = self.parse_optional_meta();
            view_item_use(ident, metadata, self.get_id())
        } else {
            fail;
        };
        self.expect(token::SEMI);
        @{node: node, attrs: attrs,
          vis: vis, span: mk_sp(lo, self.last_span.hi)}
    }

    fn parse_items_and_view_items(+first_item_attrs: ~[attribute],
                                  mode: view_item_parse_mode)
                               -> {attrs_remaining: ~[attribute],
                                   view_items: ~[@view_item],
                                   items: ~[@item]} {
        let mut attrs = vec::append(first_item_attrs,
                                    self.parse_outer_attributes());

        let items_allowed;
        match mode {
            VIEW_ITEMS_AND_ITEMS_ALLOWED | IMPORTS_AND_ITEMS_ALLOWED =>
                items_allowed = true,
            VIEW_ITEMS_ALLOWED =>
                items_allowed = false
        }

        let (view_items, items) = (DVec(), DVec());
        loop {
            match self.parse_item_or_view_item(attrs, items_allowed) {
                iovi_none =>
                    break,
                iovi_view_item(view_item) => {
                    match mode {
                        VIEW_ITEMS_AND_ITEMS_ALLOWED |
                        VIEW_ITEMS_ALLOWED => {}
                        IMPORTS_AND_ITEMS_ALLOWED =>
                            match view_item.node {
                                view_item_import(_) => {}
                                view_item_export(_) | view_item_use(*) =>
                                    self.fatal(~"exports and \"extern mod\" \
                                                 declarations are not \
                                                 allowed here")
                            }
                    }
                    view_items.push(view_item);
                }
                iovi_item(item) => {
                    assert items_allowed;
                    items.push(item)
                }
            }
            attrs = self.parse_outer_attributes();
        }

        {attrs_remaining: attrs,
         view_items: vec::from_mut(dvec::unwrap(view_items)),
         items: vec::from_mut(dvec::unwrap(items))}
    }

    // Parses a source module as a crate
    fn parse_crate_mod(_cfg: crate_cfg) -> @crate {
        let lo = self.span.lo;
        let crate_attrs = self.parse_inner_attrs_and_next();
        let first_item_outer_attrs = crate_attrs.next;
        let m = self.parse_mod_items(token::EOF, first_item_outer_attrs);
        return @spanned(lo, self.span.lo,
                     {directives: ~[],
                      module: m,
                      attrs: crate_attrs.inner,
                      config: self.cfg});
    }

    fn parse_str() -> @~str {
        match copy self.token {
          token::LIT_STR(s) => { self.bump(); self.id_to_str(s) }
          _ =>  self.fatal(~"expected string literal")
        }
    }

    // Logic for parsing crate files (.rc)
    //
    // Each crate file is a sequence of directives.
    //
    // Each directive imperatively extends its environment with 0 or more
    // items.
    fn parse_crate_directive(first_outer_attr: ~[attribute]) ->
        crate_directive {

        // Collect the next attributes
        let outer_attrs = vec::append(first_outer_attr,
                                      self.parse_outer_attributes());
        // In a crate file outer attributes are only going to apply to mods
        let expect_mod = vec::len(outer_attrs) > 0u;

        let lo = self.span.lo;
        if expect_mod || self.is_keyword(~"mod") ||
            self.is_keyword(~"module") {

            if self.is_keyword(~"mod") {
                self.expect_keyword(~"mod");
            } else {
                self.expect_keyword(~"module");
            }
            let id = self.parse_ident();
            match self.token {
              // mod x = "foo.rs";
              token::SEMI => {
                let mut hi = self.span.hi;
                self.bump();
                return spanned(lo, hi, cdir_src_mod(id, outer_attrs));
              }
              // mod x = "foo_dir" { ...directives... }
              token::LBRACE => {
                self.bump();
                let inner_attrs = self.parse_inner_attrs_and_next();
                let mod_attrs = vec::append(outer_attrs, inner_attrs.inner);
                let next_outer_attr = inner_attrs.next;
                let cdirs = self.parse_crate_directives(token::RBRACE,
                                                        next_outer_attr);
                let mut hi = self.span.hi;
                self.expect(token::RBRACE);
                return spanned(lo, hi,
                            cdir_dir_mod(id, cdirs, mod_attrs));
              }
              _ => self.unexpected()
            }
        } else if self.is_view_item() {
            let vi = self.parse_view_item(outer_attrs);
            return spanned(lo, vi.span.hi, cdir_view_item(vi));
        }
        return self.fatal(~"expected crate directive");
    }

    fn parse_crate_directives(term: token::token,
                              first_outer_attr: ~[attribute]) ->
        ~[@crate_directive] {

        // This is pretty ugly. If we have an outer attribute then we can't
        // accept seeing the terminator next, so if we do see it then fail the
        // same way parse_crate_directive would
        if vec::len(first_outer_attr) > 0u && self.token == term {
            if self.is_keyword(~"mod") {
                self.expect_keyword(~"mod");
            } else {
                self.expect_keyword(~"module");
            }
        }

        let mut cdirs: ~[@crate_directive] = ~[];
        let mut first_outer_attr = first_outer_attr;
        while self.token != term {
            let cdir = @self.parse_crate_directive(first_outer_attr);
            vec::push(cdirs, cdir);
            first_outer_attr = ~[];
        }
        return cdirs;
    }
}

impl restriction : cmp::Eq {
    pure fn eq(&&other: restriction) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: restriction) -> bool { !self.eq(other) }
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

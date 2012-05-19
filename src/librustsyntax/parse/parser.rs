import result::result;
import either::{either, left, right};
import std::map::{hashmap, str_hash};
import token::{can_begin_expr, is_ident, is_plain_ident};
import codemap::{span,fss_none};
import util::interner;
import ast_util::{spanned, mk_sp, ident_to_path, operator_prec};
import ast::*;
import lexer::reader;
import prec::{as_prec, token_to_binop};
import attr::{parse_outer_attrs_or_ext,
              parse_inner_attrs_and_next,
              parse_outer_attributes,
              parse_optional_meta};
import common::*;
import dvec::{dvec, extensions};

export expect;
export file_type;
export parser;
export parse_expr;
export parse_pat;

// FIXME: #ast expects to find this here but it's actually defined in `parse`
// Fixing this will be easier when we have export decls on individual items --
// then parse can export this publicly, and everything else crate-visibly.
// (See #1893)
import parse_from_source_str;
export parse_from_source_str;

// TODO: remove these once we go around a snapshot cycle.
// These are here for the old way that #ast (qquote.rs) worked
fn parse_expr(p: parser) -> @ast::expr { p.parse_expr() }
fn parse_pat(p: parser) -> @ast::pat { p.parse_pat() }


enum restriction {
    UNRESTRICTED,
    RESTRICT_STMT_EXPR,
    RESTRICT_NO_CALL_EXPRS,
    RESTRICT_NO_BAR_OP,
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

/*
  So that we can distinguish a class ctor or dtor
  from other class members
 */
enum class_contents { ctor_decl(fn_decl, blk, codemap::span),
                      dtor_decl(blk, codemap::span),
                      members([@class_member]) }

type arg_or_capture_item = either<arg, capture_item>;
type item_info = (ident, item_, option<[attribute]>);

class parser {
    let sess: parse_sess;
    let cfg: crate_cfg;
    let file_type: file_type;
    let mut token: token::token;
    let mut span: span;
    let mut last_span: span;
    let buffer: dvec<{tok: token::token, span: span}>;
    let mut restriction: restriction;
    let reader: reader;
    let keywords: hashmap<str, ()>;
    let restricted_keywords: hashmap<str, ()>;

    new(sess: parse_sess, cfg: ast::crate_cfg, rdr: reader,
        ftype: file_type) {
        let tok0 = lexer::next_token(rdr);
        let span0 = ast_util::mk_sp(tok0.chpos, rdr.chpos);
        self.sess = sess;
        self.cfg = cfg;
        self.file_type = ftype;
        self.token = tok0.tok;
        self.span = span0;
        self.last_span = span0;
        self.buffer = dvec::dvec();
        self.restriction == UNRESTRICTED;
        self.reader = rdr;
        self.keywords = token::keyword_table();
        self.restricted_keywords = token::restricted_keyword_table();
    }

    //TODO: uncomment when destructors workd
    //drop {} /* do not copy the parser; its state is tied to outside state */

    fn bump() {
        self.last_span = self.span;
        if self.buffer.len() == 0u {
            let next = lexer::next_token(self.reader);
            self.token = next.tok;
            self.span = mk_sp(next.chpos, self.reader.chpos);
        } else {
            let next = self.buffer.shift();
            self.token = next.tok;
            self.span = next.span;
        }
    }
    fn swap(next: token::token, lo: uint, hi: uint) {
        self.token = next;
        self.span = mk_sp(lo, hi);
    }
    fn look_ahead(distance: uint) -> token::token {
        while self.buffer.len() < distance {
            let next = lexer::next_token(self.reader);
            let sp = mk_sp(next.chpos, self.reader.chpos);
            self.buffer.push({tok: next.tok, span: sp});
        }
        ret self.buffer[distance - 1u].tok;
    }
    fn fatal(m: str) -> ! {
        self.sess.span_diagnostic.span_fatal(self.span, m)
    }
    fn span_fatal(sp: span, m: str) -> ! {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }
    fn bug(m: str) -> ! {
        self.sess.span_diagnostic.span_bug(self.span, m)
    }
    fn warn(m: str) {
        self.sess.span_diagnostic.span_warn(self.span, m)
    }
    fn get_str(i: token::str_num) -> str {
        interner::get(*self.reader.interner, i)
    }
    fn get_id() -> node_id { next_node_id(self.sess) }

    fn parse_ty_fn() -> fn_decl {
        let inputs =
            parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                      self) { |p|
            let mode = p.parse_arg_mode();
            let name = if is_plain_ident(p.token)
                && p.look_ahead(1u) == token::COLON {

                let name = parse_value_ident(p);
                p.bump();
                name
            } else { "" };

            {mode: mode, ty: p.parse_ty(false), ident: name,
             id: p.get_id()}
        };
        // FIXME: constrs is empty because right now, higher-order functions
        // can't have constrained types.
        // Not sure whether that would be desirable anyway. See #34 for the
        // story on constrained types.
        let constrs: [@constr] = [];
        let (ret_style, ret_ty) = self.parse_ret_ty();
        ret {inputs: inputs.node, output: ret_ty,
             purity: impure_fn, cf: ret_style,
             constraints: constrs};
    }

    fn parse_ty_methods() -> [ty_method] {
        (parse_seq(token::LBRACE, token::RBRACE, seq_sep_none(), self) { |p|
            let attrs = parse_outer_attributes(p);
            let flo = p.span.lo;
            let pur = p.parse_fn_purity();
            let ident = p.parse_method_name();
            let tps = p.parse_ty_params();
            let d = p.parse_ty_fn(), fhi = p.last_span.hi;
            expect(p, token::SEMI);
            {ident: ident, attrs: attrs, decl: {purity: pur with d}, tps: tps,
             span: mk_sp(flo, fhi)}
        }).node
    }

    fn parse_mt() -> mt {
        let mutbl = self.parse_mutability();
        let t = self.parse_ty(false);
        ret {ty: t, mutbl: mutbl};
    }

    fn parse_ty_field() -> ty_field {
        let lo = self.span.lo;
        let mutbl = self.parse_mutability();
        let id = parse_ident(self);
        expect(self, token::COLON);
        let ty = self.parse_ty(false);
        ret spanned(lo, ty.span.hi, {ident: id, mt: {ty: ty, mutbl: mutbl}});
    }

    // if i is the jth ident in args, return j
    // otherwise, fail
    fn ident_index(args: [arg], i: ident) -> uint {
        let mut j = 0u;
        for args.each {|a| if a.ident == i { ret j; } j += 1u; }
        self.fatal("unbound variable `" + i + "` in constraint arg");
    }

    fn parse_type_constr_arg() -> @ty_constr_arg {
        let sp = self.span;
        let mut carg = carg_base;
        expect(self, token::BINOP(token::STAR));
        if self.token == token::DOT {
            // "*..." notation for record fields
            self.bump();
            let pth = self.parse_path_without_tps();
            carg = carg_ident(pth);
        }
        // No literals yet, I guess?
        ret @{node: carg, span: sp};
    }

    fn parse_constr_arg(args: [arg]) -> @constr_arg {
        let sp = self.span;
        let mut carg = carg_base;
        if self.token == token::BINOP(token::STAR) {
            self.bump();
        } else {
            let i: ident = parse_value_ident(self);
            carg = carg_ident(self.ident_index(args, i));
        }
        ret @{node: carg, span: sp};
    }

    fn parse_ty_constr(fn_args: [arg]) -> @constr {
        let lo = self.span.lo;
        let path = self.parse_path_without_tps();
        let args: {node: [@constr_arg], span: span} =
            parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                      self, {|p| p.parse_constr_arg(fn_args)});
        ret @spanned(lo, args.span.hi,
                     {path: path, args: args.node, id: self.get_id()});
    }

    fn parse_constr_in_type() -> @ty_constr {
        let lo = self.span.lo;
        let path = self.parse_path_without_tps();
        let args: [@ty_constr_arg] =
            parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                      self, {|p| p.parse_type_constr_arg()}).node;
        let hi = self.span.lo;
        let tc: ty_constr_ = {path: path, args: args, id: self.get_id()};
        ret @spanned(lo, hi, tc);
    }


    fn parse_constrs<T: copy>(pser: fn(parser) -> @constr_general<T>) ->
        [@constr_general<T>] {
        let mut constrs: [@constr_general<T>] = [];
        loop {
            let constr = pser(self);
            constrs += [constr];
            if self.token == token::COMMA { self.bump(); }
            else { ret constrs; }
        };
    }

    fn parse_type_constraints() -> [@ty_constr] {
        ret self.parse_constrs({|p| p.parse_constr_in_type()});
    }

    fn parse_ret_ty() -> (ret_style, @ty) {
        ret if eat(self, token::RARROW) {
            let lo = self.span.lo;
            if eat(self, token::NOT) {
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

    fn region_from_name(s: option<str>) -> @region {
        let r = alt s {
          some (string) { re_named(string) }
          none { re_anon }
        };

        @{id: self.get_id(), node: r}
    }

    // Parses something like "&x"
    fn parse_region() -> @region {
        expect(self, token::BINOP(token::AND));
        alt self.token {
          token::IDENT(sid, _) {
            self.bump();
            let n = self.get_str(sid);
            self.region_from_name(some(n))
          }
          _ {
            self.region_from_name(none)
          }
        }
    }

    // Parses something like "&x." (note the trailing dot)
    fn parse_region_dot() -> @region {
        let name =
            alt self.token {
              token::IDENT(sid, _) if self.look_ahead(1u) == token::DOT {
                self.bump(); self.bump();
                some(self.get_str(sid))
              }
              _ { none }
            };
        self.region_from_name(name)
    }

    fn parse_ty(colons_before_params: bool) -> @ty {
        let lo = self.span.lo;

        alt self.maybe_parse_dollar_mac() {
          some(e) {
            ret @{id: self.get_id(),
                  node: ty_mac(spanned(lo, self.span.hi, e)),
                  span: mk_sp(lo, self.span.hi)};
          }
          none {}
        }

        let t = if self.token == token::LPAREN {
            self.bump();
            if self.token == token::RPAREN {
                self.bump();
                ty_nil
            } else {
                let mut ts = [self.parse_ty(false)];
                while self.token == token::COMMA {
                    self.bump();
                    ts += [self.parse_ty(false)];
                }
                let t = if vec::len(ts) == 1u { ts[0].node }
                else { ty_tup(ts) };
                expect(self, token::RPAREN);
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
            let elems = parse_seq(token::LBRACE, token::RBRACE,
                                  seq_sep_opt(token::COMMA), self,
                                  {|p| p.parse_ty_field()});
            if vec::len(elems.node) == 0u {
                unexpected_last(self, token::RBRACE);
            }
            let hi = elems.span.hi;

            let t = ty_rec(elems.node);
            if self.token == token::COLON {
                self.bump();
                ty_constr(@{id: self.get_id(),
                            node: t,
                            span: mk_sp(lo, hi)},
                          self.parse_type_constraints())
            } else { t }
        } else if self.token == token::LBRACKET {
            expect(self, token::LBRACKET);
            let t = ty_vec(self.parse_mt());
            expect(self, token::RBRACKET);
            t
        } else if self.token == token::BINOP(token::AND) {
            self.bump();
            let region = self.parse_region_dot();
            let mt = self.parse_mt();
            ty_rptr(region, mt)
        } else if eat_keyword(self, "fn") {
            let proto = self.parse_fn_ty_proto();
            alt proto {
              proto_bare { self.warn("fn is deprecated, use native fn"); }
              _ { /* fallthrough */ }
            }
            ty_fn(proto, self.parse_ty_fn())
        } else if eat_keyword(self, "native") {
            expect_keyword(self, "fn");
            ty_fn(proto_bare, self.parse_ty_fn())
        } else if self.token == token::MOD_SEP || is_ident(self.token) {
            let path = self.parse_path_with_tps(colons_before_params);
            ty_path(path, self.get_id())
        } else { self.fatal("expecting type"); };

        let sp = mk_sp(lo, self.last_span.hi);
        ret @{id: self.get_id(),
              node: alt self.maybe_parse_vstore() {
                // Consider a vstore suffix like /@ or /~
                none { t }
                some(v) {
                  ty_vstore(@{id: self.get_id(), node:t, span: sp}, v)
                } },
              span: sp}
    }

    fn parse_arg_mode() -> mode {
        if eat(self, token::BINOP(token::AND)) {
            expl(by_mutbl_ref)
        } else if eat(self, token::BINOP(token::MINUS)) {
            expl(by_move)
        } else if eat(self, token::ANDAND) {
            expl(by_ref)
        } else if eat(self, token::BINOP(token::PLUS)) {
            if eat(self, token::BINOP(token::PLUS)) {
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
            let ident = parse_ident(p);
            @{id: p.get_id(), is_move: is_move, name: ident, span: sp}
        }

        if eat_keyword(self, "move") {
            either::right(parse_capture_item(self, true))
        } else if eat_keyword(self, "copy") {
            either::right(parse_capture_item(self, false))
        } else {
            parse_arg_fn(self)
        }
    }

    fn parse_arg() -> arg_or_capture_item {
        let m = self.parse_arg_mode();
        let i = parse_value_ident(self);
        expect(self, token::COLON);
        let t = self.parse_ty(false);
        either::left({mode: m, ty: t, ident: i, id: self.get_id()})
    }

    fn parse_arg_or_capture_item() -> arg_or_capture_item {
        self.parse_capture_item_or() {|p| p.parse_arg() }
    }

    fn parse_fn_block_arg() -> arg_or_capture_item {
        self.parse_capture_item_or() {|p|
            let m = p.parse_arg_mode();
            let i = parse_value_ident(p);
            let t = if eat(p, token::COLON) {
                p.parse_ty(false)
            } else {
                @{id: p.get_id(),
                  node: ty_infer,
                  span: mk_sp(p.span.lo, p.span.hi)}
            };
            either::left({mode: m, ty: t, ident: i, id: p.get_id()})
        }
    }

    fn maybe_parse_dollar_mac() -> option<mac_> {
        alt self.token {
          token::DOLLAR {
            let lo = self.span.lo;
            self.bump();
            alt self.token {
              token::LIT_INT(num, ty_i) {
                self.bump();
                some(mac_var(num as uint))
              }
              token::LPAREN {
                self.bump();
                let e = self.parse_expr();
                expect(self, token::RPAREN);
                let hi = self.last_span.hi;
                some(mac_aq(mk_sp(lo,hi), e))
              }
              _ {
                self.fatal("expected `(` or integer literal");
              }
            }
          }
          _ {none}
        }
    }

    fn maybe_parse_vstore() -> option<vstore> {
        if self.token == token::BINOP(token::SLASH) {
            self.bump();
            alt self.token {
              token::AT {
                self.bump(); some(vstore_box)
              }
              token::TILDE {
                self.bump(); some(vstore_uniq)
              }
              token::UNDERSCORE {
                self.bump(); some(vstore_fixed(none))
              }
              token::LIT_INT(i, ty_i) if i >= 0i64 {
                self.bump(); some(vstore_fixed(some(i as uint)))
              }
              token::BINOP(token::AND) {
                some(vstore_slice(self.parse_region()))
              }
              _ {
                none
              }
            }
        } else {
            none
        }
    }

    fn lit_from_token(tok: token::token) -> lit_ {
        alt tok {
          token::LIT_INT(i, it) { lit_int(i, it) }
          token::LIT_UINT(u, ut) { lit_uint(u, ut) }
          token::LIT_FLOAT(s, ft) { lit_float(self.get_str(s), ft) }
          token::LIT_STR(s) { lit_str(self.get_str(s)) }
          token::LPAREN { expect(self, token::RPAREN); lit_nil }
          _ { unexpected_last(self, tok); }
        }
    }

    fn parse_lit() -> lit {
        let lo = self.span.lo;
        let lit = if eat_keyword(self, "true") {
            lit_bool(true)
        } else if eat_keyword(self, "false") {
            lit_bool(false)
        } else {
            let tok = self.token;
            self.bump();
            self.lit_from_token(tok)
        };
        ret {node: lit, span: mk_sp(lo, self.last_span.hi)};
    }

    fn parse_path_without_tps() -> @path {
        self.parse_path_without_tps_(parse_ident, parse_ident)
    }

    fn parse_path_without_tps_(
        parse_ident: fn(parser) -> ident,
        parse_last_ident: fn(parser) -> ident) -> @path {

        let lo = self.span.lo;
        let global = eat(self, token::MOD_SEP);
        let mut ids = [];
        loop {
            let is_not_last =
                self.look_ahead(2u) != token::LT
                && self.look_ahead(1u) == token::MOD_SEP;

            if is_not_last {
                ids += [parse_ident(self)];
                expect(self, token::MOD_SEP);
            } else {
                ids += [parse_last_ident(self)];
                break;
            }
        }
        @{span: mk_sp(lo, self.last_span.hi), global: global,
          idents: ids, rp: none, types: []}
    }

    fn parse_value_path() -> @path {
        self.parse_path_without_tps_(parse_ident, parse_value_ident)
    }

    fn parse_path_with_tps(colons: bool) -> @path {
        #debug["parse_path_with_tps(colons=%b)", colons];

        let lo = self.span.lo;
        let path = self.parse_path_without_tps();
        if colons && !eat(self, token::MOD_SEP) {
            ret path;
        }

        // Parse the region parameter, if any, which will
        // be written "foo/&x"
        let rp = {
            // Hack: avoid parsing vstores like /@ and /~.  This is painful
            // because the notation for region bounds and the notation for
            // vstores is... um... the same.  I guess that's my fault.  This
            // is still not ideal as for str/& we end up parsing more than we
            // ought to and have to sort it out later.
            if self.token == token::BINOP(token::SLASH)
                && self.look_ahead(1u) == token::BINOP(token::AND) {

                expect(self, token::BINOP(token::SLASH));
                some(self.parse_region())
            } else {
                none
            }
        };

        // Parse any type parameters which may appear:
        let tps = {
            if self.token == token::LT {
                parse_seq_lt_gt(some(token::COMMA), self,
                                {|p| p.parse_ty(false)})
            } else {
                {node: [], span: path.span}
            }
        };

        ret @{span: mk_sp(lo, tps.span.hi),
              rp: rp,
              types: tps.node with *path};
    }

    fn parse_mutability() -> mutability {
        if eat_keyword(self, "mut") {
            m_mutbl
        } else if eat_keyword(self, "mut") {
            m_mutbl
        } else if eat_keyword(self, "const") {
            m_const
        } else {
            m_imm
        }
    }

    fn parse_field(sep: token::token) -> field {
        let lo = self.span.lo;
        let m = self.parse_mutability();
        let i = parse_ident(self);
        expect(self, sep);
        let e = self.parse_expr();
        ret spanned(lo, e.span.hi, {mutbl: m, ident: i, expr: e});
    }

    fn mk_expr(lo: uint, hi: uint, +node: expr_) -> @expr {
        ret @{id: self.get_id(), node: node, span: mk_sp(lo, hi)};
    }

    fn mk_mac_expr(lo: uint, hi: uint, m: mac_) -> @expr {
        ret @{id: self.get_id(),
              node: expr_mac({node: m, span: mk_sp(lo, hi)}),
              span: mk_sp(lo, hi)};
    }

    fn mk_lit_u32(i: u32) -> @expr {
        let span = self.span;
        let lv_lit = @{node: lit_uint(i as u64, ty_u32),
                       span: span};

        ret @{id: self.get_id(), node: expr_lit(lv_lit), span: span};
    }

    fn mk_pexpr(lo: uint, hi: uint, node: expr_) -> pexpr {
        ret pexpr(self.mk_expr(lo, hi, node));
    }

    fn to_expr(e: pexpr) -> @expr {
        alt e.node {
          expr_tup(es) if vec::len(es) == 1u { es[0u] }
          _ { *e }
        }
    }

    fn parse_bottom_expr() -> pexpr {
        let lo = self.span.lo;
        let mut hi = self.span.hi;

        let mut ex: expr_;

        alt self.maybe_parse_dollar_mac() {
          some(x) {ret pexpr(self.mk_mac_expr(lo, self.span.hi, x));}
          _ {}
        }

        if self.token == token::LPAREN {
            self.bump();
            if self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = @spanned(lo, hi, lit_nil);
                ret self.mk_pexpr(lo, hi, expr_lit(lit));
            }
            let mut es = [self.parse_expr()];
            while self.token == token::COMMA {
                self.bump(); es += [self.parse_expr()];
            }
            hi = self.span.hi;
            expect(self, token::RPAREN);

            // Note: we retain the expr_tup() even for simple
            // parenthesized expressions, but only for a "little while".
            // This is so that wrappers around parse_bottom_expr()
            // can tell whether the expression was parenthesized or not,
            // which affects expr_is_complete().
            ret self.mk_pexpr(lo, hi, expr_tup(es));
        } else if self.token == token::LBRACE {
            self.bump();
            if is_keyword(self, "mut") ||
                is_plain_ident(self.token)
                && self.look_ahead(1u) == token::COLON {
                let mut fields = [self.parse_field(token::COLON)];
                let mut base = none;
                while self.token != token::RBRACE {
                    if eat_keyword(self, "with") {
                        base = some(self.parse_expr()); break;
                    }
                    expect(self, token::COMMA);
                    if self.token == token::RBRACE {
                        // record ends by an optional trailing comma
                        break;
                    }
                    fields += [self.parse_field(token::COLON)];
                }
                hi = self.span.hi;
                expect(self, token::RBRACE);
                ex = expr_rec(fields, base);
            } else if token::is_bar(self.token) {
                ret pexpr(self.parse_fn_block_expr());
            } else {
                let blk = self.parse_block_tail(lo, default_blk);
                ret self.mk_pexpr(blk.span.lo, blk.span.hi, expr_block(blk));
            }
        } else if eat_keyword(self, "new") {
            expect(self, token::LPAREN);
            let r = self.parse_expr();
            expect(self, token::RPAREN);
            let v = self.parse_expr();
            ret self.mk_pexpr(lo, self.span.hi,
                              expr_new(r, self.get_id(), v));
        } else if eat_keyword(self, "if") {
            ret pexpr(self.parse_if_expr());
        } else if eat_keyword(self, "for") {
            ret pexpr(self.parse_for_expr());
        } else if eat_keyword(self, "while") {
            ret pexpr(self.parse_while_expr());
        } else if eat_keyword(self, "loop") {
            ret pexpr(self.parse_loop_expr());
        } else if eat_keyword(self, "alt") {
            ret pexpr(self.parse_alt_expr());
        } else if eat_keyword(self, "fn") {
            let proto = self.parse_fn_ty_proto();
            alt proto {
              proto_bare { self.fatal("fn expr are deprecated, use fn@"); }
              proto_any { self.fatal("fn* cannot be used in an expression"); }
              _ { /* fallthrough */ }
            }
            ret pexpr(self.parse_fn_expr(proto));
        } else if eat_keyword(self, "unchecked") {
            ret pexpr(self.parse_block_expr(lo, unchecked_blk));
        } else if eat_keyword(self, "unsafe") {
            ret pexpr(self.parse_block_expr(lo, unsafe_blk));
        } else if self.token == token::LBRACKET {
            self.bump();
            let mutbl = self.parse_mutability();
            let es =
                parse_seq_to_end(token::RBRACKET, seq_sep(token::COMMA), self,
                                 {|p| p.parse_expr()});
            hi = self.span.hi;
            ex = expr_vec(es, mutbl);
        } else if self.token == token::POUND
            && self.look_ahead(1u) == token::LT {
            self.bump();
            self.bump();
            let ty = self.parse_ty(false);
            expect(self, token::GT);

            /* hack: early return to take advantage of specialized function */
            ret pexpr(self.mk_mac_expr(lo, self.span.hi,
                                       mac_embed_type(ty)));
        } else if self.token == token::POUND
            && self.look_ahead(1u) == token::LBRACE {
            self.bump();
            self.bump();
            let blk = mac_embed_block(
                self.parse_block_tail(lo, default_blk));
            ret pexpr(self.mk_mac_expr(lo, self.span.hi, blk));
        } else if self.token == token::ELLIPSIS {
            self.bump();
            ret pexpr(self.mk_mac_expr(lo, self.span.hi, mac_ellipsis));
        } else if self.token == token::POUND {
            let ex_ext = self.parse_syntax_ext();
            hi = ex_ext.span.hi;
            ex = ex_ext.node;
        } else if eat_keyword(self, "bind") {
            let e = self.parse_expr_res(RESTRICT_NO_CALL_EXPRS);
            let es =
                parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                          self, {|p| p.parse_expr_or_hole()});
            hi = es.span.hi;
            ex = expr_bind(e, es.node);
        } else if eat_keyword(self, "fail") {
            if can_begin_expr(self.token) {
                let e = self.parse_expr();
                hi = e.span.hi;
                ex = expr_fail(some(e));
            } else { ex = expr_fail(none); }
        } else if eat_keyword(self, "log") {
            expect(self, token::LPAREN);
            let lvl = self.parse_expr();
            expect(self, token::COMMA);
            let e = self.parse_expr();
            ex = expr_log(2, lvl, e);
            hi = self.span.hi;
            expect(self, token::RPAREN);
        } else if eat_keyword(self, "assert") {
            let e = self.parse_expr();
            ex = expr_assert(e);
            hi = e.span.hi;
        } else if eat_keyword(self, "check") {
            /* Should be a predicate (pure boolean function) applied to
            arguments that are all either slot variables or literals.
            but the typechecker enforces that. */
            let e = self.parse_expr();
            hi = e.span.hi;
            ex = expr_check(checked_expr, e);
        } else if eat_keyword(self, "claim") {
            /* Same rules as check, except that if check-claims
            is enabled (a command-line flag), then the parser turns
            claims into check */

            let e = self.parse_expr();
            hi = e.span.hi;
            ex = expr_check(claimed_expr, e);
        } else if eat_keyword(self, "ret") {
            if can_begin_expr(self.token) {
                let e = self.parse_expr();
                hi = e.span.hi;
                ex = expr_ret(some(e));
            } else { ex = expr_ret(none); }
        } else if eat_keyword(self, "break") {
            ex = expr_break;
            hi = self.span.hi;
        } else if eat_keyword(self, "cont") {
            ex = expr_cont;
            hi = self.span.hi;
        } else if eat_keyword(self, "copy") {
            let e = self.parse_expr();
            ex = expr_copy(e);
            hi = e.span.hi;
        } else if self.token == token::MOD_SEP ||
            is_ident(self.token) && !is_keyword(self, "true") &&
            !is_keyword(self, "false") {
            let pth = self.parse_path_with_tps(true);
            hi = pth.span.hi;
            ex = expr_path(pth);
        } else {
            let lit = self.parse_lit();
            hi = lit.span.hi;
            ex = expr_lit(@lit);
        }

        // Vstore is legal following expr_lit(lit_str(...)) and expr_vec(...)
        // only.
        alt ex {
          expr_lit(@{node: lit_str(_), span: _}) |
          expr_vec(_, _)  {
            alt self.maybe_parse_vstore() {
              none { }
              some(v) {
                hi = self.span.hi;
                ex = expr_vstore(self.mk_expr(lo, hi, ex), v);
              }
            }
          }
          _ { }
        }

        ret self.mk_pexpr(lo, hi, ex);
    }

    fn parse_block_expr(lo: uint, blk_mode: blk_check_mode) -> @expr {
        expect(self, token::LBRACE);
        let blk = self.parse_block_tail(lo, blk_mode);
        ret self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk));
    }

    fn parse_syntax_ext() -> @expr {
        let lo = self.span.lo;
        expect(self, token::POUND);
        ret self.parse_syntax_ext_naked(lo);
    }

    fn parse_syntax_ext_naked(lo: uint) -> @expr {
        alt self.token {
          token::IDENT(_, _) {}
          _ { self.fatal("expected a syntax expander name"); }
        }
        let pth = self.parse_path_without_tps();
        //temporary for a backwards-compatible cycle:
        let sep = seq_sep(token::COMMA);
        let mut e = none;
        if (self.token == token::LPAREN || self.token == token::LBRACKET) {
            let es =
                if self.token == token::LPAREN {
                parse_seq(token::LPAREN, token::RPAREN,
                          sep, self, {|p| p.parse_expr()})
        } else {
            parse_seq(token::LBRACKET, token::RBRACKET,
                      sep, self, {|p| p.parse_expr()})
        };
        let hi = es.span.hi;
        e = some(self.mk_expr(es.span.lo, hi,
                              expr_vec(es.node, m_imm)));
    }
    let mut b = none;
    if self.token == token::LBRACE {
        self.bump();
        let lo = self.span.lo;
        let mut depth = 1u;
        while (depth > 0u) {
            alt (self.token) {
              token::LBRACE {depth += 1u;}
              token::RBRACE {depth -= 1u;}
              token::EOF {self.fatal("unexpected EOF in macro body");}
              _ {}
            }
            self.bump();
        }
        let hi = self.last_span.lo;
        b = some({span: mk_sp(lo,hi)});
    }
    ret self.mk_mac_expr(lo, self.span.hi, mac_invoc(pth, e, b));
}

    fn parse_dot_or_call_expr() -> pexpr {
        let b = self.parse_bottom_expr();
        self.parse_dot_or_call_expr_with(b)
    }

    fn permits_call() -> bool {
        ret self.restriction != RESTRICT_NO_CALL_EXPRS;
    }

    fn parse_dot_or_call_expr_with(e0: pexpr) -> pexpr {
        let mut e = e0;
        let lo = e.span.lo;
        let mut hi = e.span.hi;
        loop {
            // expr.f
            if eat(self, token::DOT) {
                alt self.token {
                  token::IDENT(i, _) {
                    hi = self.span.hi;
                    self.bump();
                    let tys = if eat(self, token::MOD_SEP) {
                        expect(self, token::LT);
                        parse_seq_to_gt(some(token::COMMA), self,
                                        {|p| p.parse_ty(false)})
                    } else { [] };
                    e = self.mk_pexpr(lo, hi, expr_field(self.to_expr(e),
                                                         self.get_str(i),
                                                         tys));
                  }
                  _ { unexpected(self); }
                }
                cont;
            }
            if self.expr_is_complete(e) { break; }
            alt self.token {
              // expr(...)
              token::LPAREN if self.permits_call() {
                let es_opt =
                    parse_seq(token::LPAREN, token::RPAREN,
                              seq_sep(token::COMMA), self,
                              {|p| p.parse_expr_or_hole()});
                hi = es_opt.span.hi;

                let nd =
                    if vec::any(es_opt.node, {|e| option::is_none(e) }) {
                    expr_bind(self.to_expr(e), es_opt.node)
            } else {
                let es = vec::map(es_opt.node) {|e| option::get(e) };
                expr_call(self.to_expr(e), es, false)
            };
            e = self.mk_pexpr(lo, hi, nd);
          }

          // expr {|| ... }
          token::LBRACE if (token::is_bar(self.look_ahead(1u))
                            && self.permits_call()) {
            self.bump();
            let blk = self.parse_fn_block_expr();
            alt e.node {
              expr_call(f, args, false) {
                e = pexpr(@{node: expr_call(f, args + [blk], true)
                            with *self.to_expr(e)});
              }
              _ {
                e = self.mk_pexpr(lo, self.last_span.hi,
                                  expr_call(self.to_expr(e), [blk], true));
              }
            }
          }

          // expr[...]
          token::LBRACKET {
            self.bump();
            let ix = self.parse_expr();
            hi = ix.span.hi;
            expect(self, token::RBRACKET);
            self.get_id(); // see ast_util::op_expr_callee_id
            e = self.mk_pexpr(lo, hi, expr_index(self.to_expr(e), ix));
          }

          _ { ret e; }
        }
    }
    ret e;
}

    fn parse_prefix_expr() -> pexpr {
        let lo = self.span.lo;
        let mut hi = self.span.hi;

        let mut ex;
        alt self.token {
          token::NOT {
            self.bump();
            let e = self.to_expr(self.parse_prefix_expr());
            hi = e.span.hi;
            self.get_id(); // see ast_util::op_expr_callee_id
            ex = expr_unary(not, e);
          }
          token::BINOP(b) {
            alt b {
              token::MINUS {
                self.bump();
                let e = self.to_expr(self.parse_prefix_expr());
                hi = e.span.hi;
                self.get_id(); // see ast_util::op_expr_callee_id
                ex = expr_unary(neg, e);
              }
              token::STAR {
                self.bump();
                let e = self.to_expr(self.parse_prefix_expr());
                hi = e.span.hi;
                ex = expr_unary(deref, e);
              }
              token::AND {
                self.bump();
                let m = self.parse_mutability();
                let e = self.to_expr(self.parse_prefix_expr());
                hi = e.span.hi;
                ex = expr_addr_of(m, e);
              }
              _ { ret self.parse_dot_or_call_expr(); }
            }
          }
          token::AT {
            self.bump();
            let m = self.parse_mutability();
            let e = self.to_expr(self.parse_prefix_expr());
            hi = e.span.hi;
            ex = expr_unary(box(m), e);
          }
          token::TILDE {
            self.bump();
            let m = self.parse_mutability();
            let e = self.to_expr(self.parse_prefix_expr());
            hi = e.span.hi;
            ex = expr_unary(uniq(m), e);
          }
          _ { ret self.parse_dot_or_call_expr(); }
        }
        ret self.mk_pexpr(lo, hi, ex);
    }


    fn parse_binops() -> @expr {
        ret self.parse_more_binops(self.parse_prefix_expr(), 0u);
    }

    fn parse_more_binops(plhs: pexpr, min_prec: uint) ->
        @expr {
        let lhs = self.to_expr(plhs);
        if self.expr_is_complete(plhs) { ret lhs; }
        let peeked = self.token;
        if peeked == token::BINOP(token::OR) &&
            self.restriction == RESTRICT_NO_BAR_OP { ret lhs; }
        let cur_opt   = token_to_binop(peeked);
        alt cur_opt {
          some(cur_op) {
            let cur_prec = operator_prec(cur_op);
            if cur_prec > min_prec {
                self.bump();
                let expr = self.parse_prefix_expr();
                let rhs = self.parse_more_binops(expr, cur_prec);
                self.get_id(); // see ast_util::op_expr_callee_id
                let bin = self.mk_pexpr(lhs.span.lo, rhs.span.hi,
                                        expr_binary(cur_op, lhs, rhs));
                ret self.parse_more_binops(bin, min_prec);
            }
          }
          _ {}
        }
        if as_prec > min_prec && eat_keyword(self, "as") {
            let rhs = self.parse_ty(true);
            let _as =
                self.mk_pexpr(lhs.span.lo, rhs.span.hi, expr_cast(lhs, rhs));
            ret self.parse_more_binops(_as, min_prec);
        }
        ret lhs;
    }

    fn parse_assign_expr() -> @expr {
        let lo = self.span.lo;
        let lhs = self.parse_binops();
        alt self.token {
          token::EQ {
            self.bump();
            let rhs = self.parse_expr();
            ret self.mk_expr(lo, rhs.span.hi, expr_assign(lhs, rhs));
          }
          token::BINOPEQ(op) {
            self.bump();
            let rhs = self.parse_expr();
            let mut aop;
            alt op {
              token::PLUS { aop = add; }
              token::MINUS { aop = subtract; }
              token::STAR { aop = mul; }
              token::SLASH { aop = div; }
              token::PERCENT { aop = rem; }
              token::CARET { aop = bitxor; }
              token::AND { aop = bitand; }
              token::OR { aop = bitor; }
              token::SHL { aop = shl; }
              token::SHR { aop = shr; }
            }
            self.get_id(); // see ast_util::op_expr_callee_id
            ret self.mk_expr(lo, rhs.span.hi, expr_assign_op(aop, lhs, rhs));
          }
          token::LARROW {
            self.bump();
            let rhs = self.parse_expr();
            ret self.mk_expr(lo, rhs.span.hi, expr_move(lhs, rhs));
          }
          token::DARROW {
            self.bump();
            let rhs = self.parse_expr();
            ret self.mk_expr(lo, rhs.span.hi, expr_swap(lhs, rhs));
          }
          _ {/* fall through */ }
        }
        ret lhs;
    }

    fn parse_if_expr_1() ->
        {cond: @expr,
         then: blk,
         els: option<@expr>,
         lo: uint,
         hi: uint} {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let thn = self.parse_block();
        let mut els: option<@expr> = none;
        let mut hi = thn.span.hi;
        if eat_keyword(self, "else") {
            let elexpr = self.parse_else_expr();
            els = some(elexpr);
            hi = elexpr.span.hi;
        }
        ret {cond: cond, then: thn, els: els, lo: lo, hi: hi};
    }

    fn parse_if_expr() -> @expr {
        if eat_keyword(self, "check") {
            let q = self.parse_if_expr_1();
            ret self.mk_expr(q.lo, q.hi,
                             expr_if_check(q.cond, q.then, q.els));
        } else {
            let q = self.parse_if_expr_1();
            ret self.mk_expr(q.lo, q.hi, expr_if(q.cond, q.then, q.els));
        }
    }

    fn parse_fn_expr(proto: proto) -> @expr {
        let lo = self.last_span.lo;

        let cc_old = self.parse_old_skool_capture_clause();

        // if we want to allow fn expression argument types to be inferred in
        // the future, just have to change parse_arg to parse_fn_block_arg.
        let (decl, capture_clause) =
            self.parse_fn_decl(impure_fn,
                               {|p| p.parse_arg_or_capture_item()});

        let body = self.parse_block();
        ret self.mk_expr(lo, body.span.hi,
                         expr_fn(proto, decl, body,
                                 @(*capture_clause + cc_old)));
    }

    fn parse_fn_block_expr() -> @expr {
        let lo = self.last_span.lo;
        let (decl, captures) = self.parse_fn_block_decl();
        let body = self.parse_block_tail(lo, default_blk);
        ret self.mk_expr(lo, body.span.hi,
                         expr_fn_block(decl, body, captures));
    }

    fn parse_else_expr() -> @expr {
        if eat_keyword(self, "if") {
            ret self.parse_if_expr();
        } else {
            let blk = self.parse_block();
            ret self.mk_expr(blk.span.lo, blk.span.hi, expr_block(blk));
        }
    }

    fn parse_for_expr() -> @expr {
        let lo = self.last_span;
        let call = self.parse_expr_res(RESTRICT_STMT_EXPR);
        alt call.node {
          expr_call(f, args, true) {
            let b_arg = vec::last(args);
            let last = self.mk_expr(b_arg.span.lo, b_arg.span.hi,
                                    expr_loop_body(b_arg));
            @{node: expr_call(f, vec::init(args) + [last], true)
              with *call}
          }
          _ {
            self.span_fatal(lo, "`for` must be followed by a block call");
          }
        }
    }

    fn parse_while_expr() -> @expr {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let body = self.parse_block_no_value();
        let mut hi = body.span.hi;
        ret self.mk_expr(lo, hi, expr_while(cond, body));
    }

    fn parse_loop_expr() -> @expr {
        let lo = self.last_span.lo;
        let body = self.parse_block_no_value();
        let mut hi = body.span.hi;
        ret self.mk_expr(lo, hi, expr_loop(body));
    }

    fn parse_alt_expr() -> @expr {
        let lo = self.last_span.lo;
        let mode = if eat_keyword(self, "check") { alt_check }
        else { alt_exhaustive };
        let discriminant = self.parse_expr();
        expect(self, token::LBRACE);
        let mut arms: [arm] = [];
        while self.token != token::RBRACE {
            let pats = self.parse_pats();
            let mut guard = none;
            if eat_keyword(self, "if") { guard = some(self.parse_expr()); }
            let blk = self.parse_block();
            arms += [{pats: pats, guard: guard, body: blk}];
        }
        let mut hi = self.span.hi;
        self.bump();
        ret self.mk_expr(lo, hi, expr_alt(discriminant, arms, mode));
    }

    fn parse_expr() -> @expr {
        ret self.parse_expr_res(UNRESTRICTED);
    }

    fn parse_expr_or_hole() -> option<@expr> {
        alt self.token {
          token::UNDERSCORE { self.bump(); ret none; }
          _ { ret some(self.parse_expr()); }
        }
    }

    fn parse_expr_res(r: restriction) -> @expr {
        let old = self.restriction;
        self.restriction = r;
        let e = self.parse_assign_expr();
        self.restriction = old;
        ret e;
    }

    fn parse_initializer() -> option<initializer> {
        alt self.token {
          token::EQ {
            self.bump();
            ret some({op: init_assign, expr: self.parse_expr()});
          }
          token::LARROW {
            self.bump();
            ret some({op: init_move, expr: self.parse_expr()});
          }
          // Now that the the channel is the first argument to receive,
          // combining it with an initializer doesn't really make sense.
          // case (token::RECV) {
          //     self.bump();
          //     ret some(rec(op = init_recv,
          //                  expr = self.parse_expr()));
          // }
          _ {
            ret none;
          }
        }
    }

    fn parse_pats() -> [@pat] {
        let mut pats = [];
        loop {
            pats += [self.parse_pat()];
            if self.token == token::BINOP(token::OR) { self.bump(); }
            else { ret pats; }
        };
    }

    fn parse_pat() -> @pat {
        let lo = self.span.lo;
        let mut hi = self.span.hi;
        let mut pat;
        alt self.token {
          token::UNDERSCORE { self.bump(); pat = pat_wild; }
          token::AT {
            self.bump();
            let sub = self.parse_pat();
            pat = pat_box(sub);
            hi = sub.span.hi;
          }
          token::TILDE {
            self.bump();
            let sub = self.parse_pat();
            pat = pat_uniq(sub);
            hi = sub.span.hi;
          }
          token::LBRACE {
            self.bump();
            let mut fields = [];
            let mut etc = false;
            let mut first = true;
            while self.token != token::RBRACE {
                if first { first = false; }
                else { expect(self, token::COMMA); }

                if self.token == token::UNDERSCORE {
                    self.bump();
                    if self.token != token::RBRACE {
                        self.fatal("expecting }, found " +
                                   token_to_str(self.reader, self.token));
                    }
                    etc = true;
                    break;
                }

                let lo1 = self.last_span.lo;
                let fieldname = if self.look_ahead(1u) == token::COLON {
                    parse_ident(self)
                } else {
                    parse_value_ident(self)
                };
                let hi1 = self.last_span.lo;
                let fieldpath = ast_util::ident_to_path(mk_sp(lo1, hi1),
                                                        fieldname);
                let mut subpat;
                if self.token == token::COLON {
                    self.bump();
                    subpat = self.parse_pat();
                } else {
                    subpat = @{id: self.get_id(),
                               node: pat_ident(fieldpath, none),
                               span: mk_sp(lo, hi)};
                }
                fields += [{ident: fieldname, pat: subpat}];
            }
            hi = self.span.hi;
            self.bump();
            pat = pat_rec(fields, etc);
          }
          token::LPAREN {
            self.bump();
            if self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = @{node: lit_nil, span: mk_sp(lo, hi)};
                let expr = self.mk_expr(lo, hi, expr_lit(lit));
                pat = pat_lit(expr);
            } else {
                let mut fields = [self.parse_pat()];
                while self.token == token::COMMA {
                    self.bump();
                    fields += [self.parse_pat()];
                }
                if vec::len(fields) == 1u { expect(self, token::COMMA); }
                hi = self.span.hi;
                expect(self, token::RPAREN);
                pat = pat_tup(fields);
            }
          }
          tok {
            if !is_ident(tok) || is_keyword(self, "true")
                || is_keyword(self, "false") {
                let val = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                if eat_keyword(self, "to") {
                    let end = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                    hi = end.span.hi;
                    pat = pat_range(val, end);
                } else {
                    hi = val.span.hi;
                    pat = pat_lit(val);
                }
            } else if is_plain_ident(self.token) &&
                alt self.look_ahead(1u) {
                  token::LPAREN | token::LBRACKET | token::LT { false }
                  _ { true }
                } {
                let name = self.parse_value_path();
                let sub = if eat(self, token::AT) { some(self.parse_pat()) }
                else { none };
                pat = pat_ident(name, sub);
            } else {
                let enum_path = self.parse_path_with_tps(true);
                hi = enum_path.span.hi;
                let mut args: [@pat] = [];
                let mut star_pat = false;
                alt self.token {
                  token::LPAREN {
                    alt self.look_ahead(1u) {
                      token::BINOP(token::STAR) {
                        // This is a "top constructor only" pat
                        self.bump(); self.bump();
                        star_pat = true;
                        expect(self, token::RPAREN);
                      }
                      _ {
                        let a = parse_seq(token::LPAREN, token::RPAREN,
                                          seq_sep(token::COMMA), self,
                                          {|p| p.parse_pat()});
                        args = a.node;
                        hi = a.span.hi;
                      }
                    }
                  }
                  _ { }
                }
                // at this point, we're not sure whether it's a enum or a bind
                if star_pat {
                    pat = pat_enum(enum_path, none);
                }
                else if vec::is_empty(args) &&
                    vec::len(enum_path.idents) == 1u {
                    pat = pat_ident(enum_path, none);
                }
                else {
                    pat = pat_enum(enum_path, some(args));
                }
            }
          }
        }
        ret @{id: self.get_id(), node: pat, span: mk_sp(lo, hi)};
    }

    fn parse_local(is_mutbl: bool,
                   allow_init: bool) -> @local {
        let lo = self.span.lo;
        let pat = self.parse_pat();
        let mut ty = @{id: self.get_id(),
                       node: ty_infer,
                       span: mk_sp(lo, lo)};
        if eat(self, token::COLON) { ty = self.parse_ty(false); }
        let init = if allow_init { self.parse_initializer() } else { none };
        ret @spanned(lo, self.last_span.hi,
                     {is_mutbl: is_mutbl, ty: ty, pat: pat,
                      init: init, id: self.get_id()});
    }

    fn parse_let() -> @decl {
        let is_mutbl = eat_keyword(self, "mut");
        let lo = self.span.lo;
        let mut locals = [self.parse_local(is_mutbl, true)];
        while eat(self, token::COMMA) {
            locals += [self.parse_local(is_mutbl, true)];
        }
        ret @spanned(lo, self.last_span.hi, decl_local(locals));
    }

    /* assumes "let" token has already been consumed */
    fn parse_instance_var(pr: visibility) -> @class_member {
        let mut is_mutbl = class_immutable;
        let lo = self.span.lo;
        if eat_keyword(self, "mut") {
            is_mutbl = class_mutable;
        }
        if !is_plain_ident(self.token) {
            self.fatal("expecting ident");
        }
        let name = parse_ident(self);
        expect(self, token::COLON);
        let ty = self.parse_ty(false);
        ret @{node: instance_var(name, ty, is_mutbl, self.get_id(), pr),
              span: mk_sp(lo, self.last_span.hi)};
    }

    fn parse_stmt(+first_item_attrs: [attribute]) -> @stmt {
        fn check_expected_item(p: parser, current_attrs: [attribute]) {
            // If we have attributes then we should have an item
            if vec::is_not_empty(current_attrs) {
                p.fatal("expected item");
            }
        }

        let lo = self.span.lo;
        if is_keyword(self, "let") {
            check_expected_item(self, first_item_attrs);
            expect_keyword(self, "let");
            let decl = self.parse_let();
            ret @spanned(lo, decl.span.hi, stmt_decl(decl, self.get_id()));
        } else {
            let mut item_attrs;
            alt parse_outer_attrs_or_ext(self, first_item_attrs) {
              none { item_attrs = []; }
              some(left(attrs)) { item_attrs = attrs; }
              some(right(ext)) {
                ret @spanned(lo, ext.span.hi, stmt_expr(ext, self.get_id()));
              }
            }

            let item_attrs = first_item_attrs + item_attrs;

            alt self.parse_item(item_attrs, public) {
              some(i) {
                let mut hi = i.span.hi;
                let decl = @spanned(lo, hi, decl_item(i));
                ret @spanned(lo, hi, stmt_decl(decl, self.get_id()));
              }
              none() { /* fallthrough */ }
            }

            check_expected_item(self, item_attrs);

            // Remainder are line-expr stmts.
            let e = self.parse_expr_res(RESTRICT_STMT_EXPR);
            ret @spanned(lo, e.span.hi, stmt_expr(e, self.get_id()));
        }
    }

    fn expr_is_complete(e: pexpr) -> bool {
        log(debug, ("expr_is_complete", self.restriction,
                    print::pprust::expr_to_str(*e),
                    classify::expr_requires_semi_to_be_stmt(*e)));
        ret self.restriction == RESTRICT_STMT_EXPR &&
            !classify::expr_requires_semi_to_be_stmt(*e);
    }

    fn parse_block() -> blk {
        let (attrs, blk) = self.parse_inner_attrs_and_block(false);
        assert vec::is_empty(attrs);
        ret blk;
    }

    fn parse_inner_attrs_and_block(parse_attrs: bool) -> ([attribute], blk) {

        fn maybe_parse_inner_attrs_and_next(p: parser, parse_attrs: bool) ->
            {inner: [attribute], next: [attribute]} {
            if parse_attrs {
                parse_inner_attrs_and_next(p)
            } else {
                {inner: [], next: []}
            }
        }

        let lo = self.span.lo;
        if eat_keyword(self, "unchecked") {
            expect(self, token::LBRACE);
            let {inner, next} = maybe_parse_inner_attrs_and_next(self,
                                                                 parse_attrs);
            ret (inner, self.parse_block_tail_(lo, unchecked_blk, next));
        } else if eat_keyword(self, "unsafe") {
            expect(self, token::LBRACE);
            let {inner, next} = maybe_parse_inner_attrs_and_next(self,
                                                                 parse_attrs);
            ret (inner, self.parse_block_tail_(lo, unsafe_blk, next));
        } else {
            expect(self, token::LBRACE);
            let {inner, next} = maybe_parse_inner_attrs_and_next(self,
                                                                 parse_attrs);
            ret (inner, self.parse_block_tail_(lo, default_blk, next));
        }
    }

    fn parse_block_no_value() -> blk {
        // We parse blocks that cannot have a value the same as any other
        // block; the type checker will make sure that the tail expression (if
        // any) has unit type.
        ret self.parse_block();
    }

    // Precondition: already parsed the '{' or '#{'
    // I guess that also means "already parsed the 'impure'" if
    // necessary, and this should take a qualifier.
    // some blocks start with "#{"...
    fn parse_block_tail(lo: uint, s: blk_check_mode) -> blk {
        self.parse_block_tail_(lo, s, [])
    }

    fn parse_block_tail_(lo: uint, s: blk_check_mode,
                         +first_item_attrs: [attribute]) -> blk {
        let mut stmts = [];
        let mut expr = none;
        let {attrs_remaining, view_items} =
            self.parse_view(first_item_attrs, true);
        let mut initial_attrs = attrs_remaining;

        if self.token == token::RBRACE && !vec::is_empty(initial_attrs) {
            self.fatal("expected item");
        }

        while self.token != token::RBRACE {
            alt self.token {
              token::SEMI {
                self.bump(); // empty
              }
              _ {
                let stmt = self.parse_stmt(initial_attrs);
                initial_attrs = [];
                alt stmt.node {
                  stmt_expr(e, stmt_id) { // Expression without semicolon:
                    alt self.token {
                      token::SEMI {
                        self.bump();
                        stmts += [@{node: stmt_semi(e, stmt_id) with *stmt}];
                      }
                      token::RBRACE {
                        expr = some(e);
                      }
                      t {
                        if classify::stmt_ends_with_semi(*stmt) {
                            self.fatal("expected ';' or '}' after expression \
                                        but found '"
                                       + token_to_str(self.reader, t) + "'");
                        }
                        stmts += [stmt];
                      }
                    }
                  }

                  _ { // All other kinds of statements:
                    stmts += [stmt];

                    if classify::stmt_ends_with_semi(*stmt) {
                        expect(self, token::SEMI);
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
        ret spanned(lo, hi, bloc);
    }

    fn parse_ty_param() -> ty_param {
        let mut bounds = [];
        let ident = parse_ident(self);
        if eat(self, token::COLON) {
            while self.token != token::COMMA && self.token != token::GT {
                if eat_keyword(self, "send") { bounds += [bound_send]; }
                else if eat_keyword(self, "copy") { bounds += [bound_copy]; }
                else { bounds += [bound_iface(self.parse_ty(false))]; }
            }
        }
        ret {ident: ident, id: self.get_id(), bounds: @bounds};
    }

    fn parse_ty_params() -> [ty_param] {
        if eat(self, token::LT) {
            parse_seq_to_gt(some(token::COMMA), self,
                            {|p| p.parse_ty_param()})
        } else { [] }
    }

    // FIXME Remove after snapshot
    fn parse_old_skool_capture_clause() -> [capture_item] {
        fn expect_opt_trailing_semi(p: parser) {
            if !eat(p, token::SEMI) {
                if p.token != token::RBRACKET {
                    p.fatal("expecting ; or ]");
                }
            }
        }

        fn eat_ident_list(p: parser, is_move: bool) -> [capture_item] {
            let mut res = [];
            loop {
                alt p.token {
                  token::IDENT(_, _) {
                    let id = p.get_id();
                    let sp = mk_sp(p.span.lo, p.span.hi);
                    let ident = parse_ident(p);
                    res += [@{id:id, is_move: is_move, name:ident, span:sp}];
                    if !eat(p, token::COMMA) {
                        ret res;
                    }
                  }

                  _ { ret res; }
                }
            };
        }

        let mut cap_items = [];

        if eat(self, token::LBRACKET) {
            while !eat(self, token::RBRACKET) {
                if eat_keyword(self, "copy") {
                    cap_items += eat_ident_list(self, false);
                    expect_opt_trailing_semi(self);
                } else if eat_keyword(self, "move") {
                    cap_items += eat_ident_list(self, true);
                    expect_opt_trailing_semi(self);
                } else {
                    let s: str = "expecting send, copy, or move clause";
                    self.fatal(s);
                }
            }
        }

        ret cap_items;
    }

    fn parse_fn_decl(purity: purity,
                     parse_arg_fn: fn(parser) -> arg_or_capture_item)
        -> (fn_decl, capture_clause) {

        let args_or_capture_items: [arg_or_capture_item] =
            parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                      self, parse_arg_fn).node;

        let inputs = either::lefts(args_or_capture_items);
        let capture_clause = @either::rights(args_or_capture_items);

        // Use the args list to translate each bound variable
        // mentioned in a constraint to an arg index.
        // Seems weird to do this in the parser, but I'm not sure how else to.
        let mut constrs = [];
        if self.token == token::COLON {
            self.bump();
            constrs = self.parse_constrs({|p| p.parse_ty_constr(inputs) });
        }
        let (ret_style, ret_ty) = self.parse_ret_ty();
        ret ({inputs: inputs,
              output: ret_ty,
              purity: purity,
              cf: ret_style,
              constraints: constrs}, capture_clause);
    }

    fn parse_fn_block_decl() -> (fn_decl, capture_clause) {
        let inputs_captures = {
            if eat(self, token::OROR) {
                []
            } else {
                parse_seq(token::BINOP(token::OR), token::BINOP(token::OR),
                          seq_sep(token::COMMA), self,
                          {|p| p.parse_fn_block_arg()}).node
            }
        };
        let output = if eat(self, token::RARROW) {
            self.parse_ty(false)
        } else {
            @{id: self.get_id(), node: ty_infer, span: self.span}
        };
        ret ({inputs: either::lefts(inputs_captures),
              output: output,
              purity: impure_fn,
              cf: return_val,
              constraints: []},
             @either::rights(inputs_captures));
    }

    fn parse_fn_header() -> {ident: ident, tps: [ty_param]} {
        let id = parse_value_ident(self);
        let ty_params = self.parse_ty_params();
        ret {ident: id, tps: ty_params};
    }

    fn mk_item(lo: uint, hi: uint, +ident: ident,
               +node: item_, vis: visibility,
               +attrs: [attribute]) -> @item {
        ret @{ident: ident,
              attrs: attrs,
              id: self.get_id(),
              node: node,
              vis: vis,
              span: mk_sp(lo, hi)};
    }

    fn parse_item_fn(purity: purity) -> item_info {
        let t = self.parse_fn_header();
        let (decl, _) = self.parse_fn_decl(purity, {|p| p.parse_arg()});
        let (inner_attrs, body) = self.parse_inner_attrs_and_block(true);
        (t.ident, item_fn(decl, t.tps, body), some(inner_attrs))
    }

    fn parse_method_name() -> ident {
        alt self.token {
          token::BINOP(op) { self.bump(); token::binop_to_str(op) }
          token::NOT { self.bump(); "!" }
          token::LBRACKET { self.bump(); expect(self, token::RBRACKET); "[]" }
          _ {
            let id = parse_value_ident(self);
            if id == "unary" && eat(self, token::BINOP(token::MINUS)) {
                "unary-"
            }
            else { id }
          }
        }
    }

    fn parse_method(pr: visibility) -> @method {
        let attrs = parse_outer_attributes(self);
        let lo = self.span.lo, pur = self.parse_fn_purity();
        let ident = self.parse_method_name();
        let tps = self.parse_ty_params();
        let (decl, _) = self.parse_fn_decl(pur, {|p| p.parse_arg()});
        let (inner_attrs, body) = self.parse_inner_attrs_and_block(true);
        let attrs = attrs + inner_attrs;
        @{ident: ident, attrs: attrs, tps: tps, decl: decl, body: body,
          id: self.get_id(), span: mk_sp(lo, body.span.hi),
          self_id: self.get_id(), vis: pr}
    }

    fn parse_item_iface() -> item_info {
        let ident = parse_ident(self);
        let rp = self.parse_region_param();
        let tps = self.parse_ty_params();
        let meths = self.parse_ty_methods();
        (ident, item_iface(tps, rp, meths), none)
    }

    // Parses three variants (with the region/type params always optional):
    //    impl /&<T: copy> of to_str for [T] { ... }
    //    impl name/&<T> of to_str for [T] { ... }
    //    impl name/&<T> for [T] { ... }
    fn parse_item_impl() -> item_info {
        fn wrap_path(p: parser, pt: @path) -> @ty {
            @{id: p.get_id(), node: ty_path(pt, p.get_id()), span: pt.span}
        }
        let mut (ident, rp, tps) = {
            if self.token == token::LT {
                (none, rp_none, self.parse_ty_params())
            } else if self.token == token::BINOP(token::SLASH) {
                (none, self.parse_region_param(), self.parse_ty_params())
            }
            else if is_keyword(self, "of") {
                (none, rp_none, [])
            } else {
                let id = parse_ident(self);
                let rp = self.parse_region_param();
                (some(id), rp, self.parse_ty_params())
            }
        };
        let ifce = if eat_keyword(self, "of") {
            let path = self.parse_path_with_tps(false);
            if option::is_none(ident) {
                ident = some(vec::last(path.idents));
            }
            some(@{path: path, id: self.get_id()})
        } else { none };
        let ident = alt ident {
          some(name) { name }
          none { expect_keyword(self, "of"); fail; }
        };
        expect_keyword(self, "for");
        let ty = self.parse_ty(false);
        let mut meths = [];
        expect(self, token::LBRACE);
        while !eat(self, token::RBRACE) {
            meths += [self.parse_method(public)];
        }
        (ident, item_impl(tps, rp, ifce, ty, meths), none)
    }

    fn parse_item_res() -> item_info {
        let ident = parse_value_ident(self);
        let rp = self.parse_region_param();
        let ty_params = self.parse_ty_params();
        expect(self, token::LPAREN);
        let arg_ident = parse_value_ident(self);
        expect(self, token::COLON);
        let t = self.parse_ty(false);
        expect(self, token::RPAREN);
        let dtor = self.parse_block_no_value();
        let decl = {
            inputs: [{mode: expl(by_ref), ty: t,
                      ident: arg_ident, id: self.get_id()}],
            output: @{id: self.get_id(), node: ty_nil,
                      span: ast_util::dummy_sp()},
            purity: impure_fn,
            cf: return_val,
            constraints: []
        };
        (ident, item_res(decl, ty_params, dtor,
                         self.get_id(), self.get_id(), rp), none)
    }

    // Instantiates ident <i> with references to <typarams> as arguments.
    // Used to create a path that refers to a class which will be defined as
    // the return type of the ctor function.
    fn ident_to_path_tys(i: ident,
                         rp: region_param,
                         typarams: [ty_param]) -> @path {
        let s = self.last_span;

        // Hack.  But then, this whole function is in service of a hack.
        let a_r = alt rp {
          rp_none { none }
          rp_self { some(self.region_from_name(some("self"))) }
        };

        @{span: s, global: false, idents: [i],
          rp: a_r,
          types: vec::map(typarams, {|tp|
              @{id: self.get_id(),
                node: ty_path(ident_to_path(s, tp.ident), self.get_id()),
                span: s}})
         }
    }

    fn parse_iface_ref() -> @iface_ref {
        @{path: self.parse_path_with_tps(false),
          id: self.get_id()}
    }

    fn parse_iface_ref_list() -> [@iface_ref] {
        parse_seq_to_before_end(token::LBRACE, seq_sep(token::COMMA), self,
                                {|p| p.parse_iface_ref()})
    }

    fn parse_item_class() -> item_info {
        let class_name = parse_value_ident(self);
        let rp = self.parse_region_param();
        let ty_params = self.parse_ty_params();
        let class_path = self.ident_to_path_tys(class_name, rp, ty_params);
        let ifaces : [@iface_ref] = if eat_keyword(self, "implements")
            { self.parse_iface_ref_list() }
        else { [] };
        expect(self, token::LBRACE);
        let mut ms: [@class_member] = [];
        let ctor_id = self.get_id();
        let mut the_ctor : option<(fn_decl, blk, codemap::span)> = none;
        let mut the_dtor : option<(blk, codemap::span)> = none;
        while self.token != token::RBRACE {
            alt self.parse_class_item(class_path) {
              ctor_decl(a_fn_decl, blk, s) {
                the_ctor = some((a_fn_decl, blk, s));
              }
              dtor_decl(blk, s) {
                the_dtor = some((blk, s));
              }
              members(mms) { ms += mms; }
            }
        }
        let actual_dtor = option::map(the_dtor) {|dtor|
            let (d_body, d_s) = dtor;
            {node: {id: self.get_id(),
                    self_id: self.get_id(),
                    body: d_body},
             span: d_s}};
        self.bump();
        alt the_ctor {
          some((ct_d, ct_b, ct_s)) {
            (class_name,
             item_class(ty_params, ifaces, ms, {
                 node: {id: ctor_id,
                        self_id: self.get_id(),
                        dec: ct_d,
                        body: ct_b},
                 span: ct_s}, actual_dtor, rp),
             none)
          }
          /*
          Is it strange for the parser to check this?
          */
          none {
            self.fatal("class with no ctor");
          }
        }
    }

    fn parse_single_class_item(vis: visibility)
        -> @class_member {
        if eat_keyword(self, "let") {
            let a_var = self.parse_instance_var(vis);
            expect(self, token::SEMI);
            ret a_var;
        }
        else {
            let m = self.parse_method(vis);
            ret @{node: class_method(m), span: m.span};
        }
    }

    fn parse_ctor(result_ty: ast::ty_) -> class_contents {
        // Can ctors/dtors have attrs? FIXME
        let lo = self.last_span.lo;
        let (decl_, _) = self.parse_fn_decl(impure_fn, {|p| p.parse_arg()});
        let decl = {output: @{id: self.get_id(),
                              node: result_ty, span: decl_.output.span}
                    with decl_};
        let body = self.parse_block();
        ctor_decl(decl, body, mk_sp(lo, self.last_span.hi))
    }

    fn parse_dtor() -> class_contents {
        // Can ctors/dtors have attrs? FIXME
        let lo = self.last_span.lo;
        let body = self.parse_block();
        dtor_decl(body, mk_sp(lo, self.last_span.hi))
    }

    fn parse_class_item(class_name_with_tps: @path)
        -> class_contents {
        if eat_keyword(self, "new") {
            // result type is always the type of the class
            ret self.parse_ctor(ty_path(class_name_with_tps,
                                        self.get_id()));
        }
        else if eat_keyword(self, "drop") {
            ret self.parse_dtor();
        }
        else if eat_keyword(self, "priv") {
            expect(self, token::LBRACE);
        let mut results = [];
        while self.token != token::RBRACE {
            results += [self.parse_single_class_item(private)];
        }
        self.bump();
        ret members(results);
    }
    else {
        // Probably need to parse attrs
        ret members([self.parse_single_class_item(public)]);
    }
}

    fn parse_visibility(def: visibility) -> visibility {
        if eat_keyword(self, "pub") { public }
        else if eat_keyword(self, "priv") { private }
        else { def }
    }

    fn parse_mod_items(term: token::token,
                       +first_item_attrs: [attribute]) -> _mod {
        // Shouldn't be any view items since we've already parsed an item attr
        let {attrs_remaining, view_items} =
            self.parse_view(first_item_attrs, false);
        let mut items: [@item] = [];
        let mut first = true;
        while self.token != term {
            let mut attrs = parse_outer_attributes(self);
            if first { attrs = attrs_remaining + attrs; first = false; }
            #debug["parse_mod_items: parse_item(attrs=%?)", attrs];
            let vis = self.parse_visibility(private);
            alt self.parse_item(attrs, vis) {
              some(i) { items += [i]; }
              _ {
                self.fatal("expected item but found '" +
                           token_to_str(self.reader, self.token) + "'");
              }
            }
            #debug["parse_mod_items: attrs=%?", attrs];
        }

        if first && attrs_remaining.len() > 0u {
            // We parsed attributes for the first item but didn't find it
            self.fatal("expected item");
        }

        ret {view_items: view_items, items: items};
    }

    fn parse_item_const() -> item_info {
        let id = parse_value_ident(self);
        expect(self, token::COLON);
        let ty = self.parse_ty(false);
        expect(self, token::EQ);
        let e = self.parse_expr();
        expect(self, token::SEMI);
        (id, item_const(ty, e), none)
    }

    fn parse_item_mod() -> item_info {
        let id = parse_ident(self);
        expect(self, token::LBRACE);
        let inner_attrs = parse_inner_attrs_and_next(self);
        let m = self.parse_mod_items(token::RBRACE, inner_attrs.next);
        expect(self, token::RBRACE);
        (id, item_mod(m), some(inner_attrs.inner))
    }

    fn parse_item_native_fn(+attrs: [attribute],
                            purity: purity) -> @native_item {
        let lo = self.last_span.lo;
        let t = self.parse_fn_header();
        let (decl, _) = self.parse_fn_decl(purity, {|p| p.parse_arg()});
        let mut hi = self.span.hi;
        expect(self, token::SEMI);
        ret @{ident: t.ident,
              attrs: attrs,
              node: native_item_fn(decl, t.tps),
              id: self.get_id(),
              span: mk_sp(lo, hi)};
    }

    fn parse_fn_purity() -> purity {
        if eat_keyword(self, "fn") { impure_fn }
        else if eat_keyword(self, "pure") {
            expect_keyword(self, "fn");
            pure_fn
        } else if eat_keyword(self, "unsafe") {
            expect_keyword(self, "fn");
            unsafe_fn
        }
        else { unexpected(self); }
    }

    fn parse_native_item(+attrs: [attribute]) ->
        @native_item {
        self.parse_item_native_fn(attrs, self.parse_fn_purity())
    }

    fn parse_native_mod_items(+first_item_attrs: [attribute]) ->
        native_mod {
        // Shouldn't be any view items since we've already parsed an item attr
        let {attrs_remaining, view_items} =
            self.parse_view(first_item_attrs, false);
        let mut items: [@native_item] = [];
        let mut initial_attrs = attrs_remaining;
        while self.token != token::RBRACE {
            let attrs = initial_attrs + parse_outer_attributes(self);
            initial_attrs = [];
            items += [self.parse_native_item(attrs)];
        }
        ret {view_items: view_items,
             items: items};
    }

    fn parse_item_native_mod() -> item_info {
        expect_keyword(self, "mod");
        let id = parse_ident(self);
        expect(self, token::LBRACE);
        let more_attrs = parse_inner_attrs_and_next(self);
        let m = self.parse_native_mod_items(more_attrs.next);
        expect(self, token::RBRACE);
        (id, item_native_mod(m), some(more_attrs.inner))
    }

    fn parse_type_decl() -> {lo: uint, ident: ident} {
        let lo = self.last_span.lo;
        let id = parse_ident(self);
        ret {lo: lo, ident: id};
    }

    fn parse_item_type() -> item_info {
        let t = self.parse_type_decl();
        let rp = self.parse_region_param();
        let tps = self.parse_ty_params();
        expect(self, token::EQ);
        let ty = self.parse_ty(false);
        expect(self, token::SEMI);
        (t.ident, item_ty(ty, tps, rp), none)
    }

    fn parse_region_param() -> region_param {
        if eat(self, token::BINOP(token::SLASH)) {
            expect(self, token::BINOP(token::AND));
            rp_self
        } else {
            rp_none
        }
    }

    fn parse_item_enum(default_vis: visibility) -> item_info {
        let id = parse_ident(self);
        let rp = self.parse_region_param();
        let ty_params = self.parse_ty_params();
        let mut variants: [variant] = [];
        // Newtype syntax
        if self.token == token::EQ {
            check_restricted_keywords_(self, id);
            self.bump();
            let ty = self.parse_ty(false);
            expect(self, token::SEMI);
            let variant =
                spanned(ty.span.lo, ty.span.hi,
                        {name: id,
                         attrs: [],
                         args: [{ty: ty, id: self.get_id()}],
                         id: self.get_id(),
                         disr_expr: none,
                         vis: public});
            ret (id, item_enum([variant], ty_params, rp), none);
        }
        expect(self, token::LBRACE);

        let mut all_nullary = true, have_disr = false;

        while self.token != token::RBRACE {
            let variant_attrs = parse_outer_attributes(self);
            let vlo = self.span.lo;
            let vis = self.parse_visibility(default_vis);
            let ident = parse_value_ident(self);
            let mut args = [], disr_expr = none;
            if self.token == token::LPAREN {
                all_nullary = false;
                let arg_tys = parse_seq(token::LPAREN, token::RPAREN,
                                        seq_sep(token::COMMA), self,
                                        {|p| p.parse_ty(false)});
                for arg_tys.node.each {|ty|
                    args += [{ty: ty, id: self.get_id()}];
                }
            } else if eat(self, token::EQ) {
                have_disr = true;
                disr_expr = some(self.parse_expr());
            }

            let vr = {name: ident, attrs: variant_attrs,
                      args: args, id: self.get_id(),
                      disr_expr: disr_expr, vis: vis};
            variants += [spanned(vlo, self.last_span.hi, vr)];

            if !eat(self, token::COMMA) { break; }
        }
        expect(self, token::RBRACE);
        if (have_disr && !all_nullary) {
            self.fatal("discriminator values can only be used with a c-like \
                        enum");
        }
        (id, item_enum(variants, ty_params, rp), none)
    }

    fn parse_fn_ty_proto() -> proto {
        alt self.token {
          token::AT {
            self.bump();
            proto_box
          }
          token::TILDE {
            self.bump();
            proto_uniq
          }
          token::BINOP(token::AND) {
            self.bump();
            proto_block
          }
          _ {
            proto_any
          }
        }
    }

    fn fn_expr_lookahead(tok: token::token) -> bool {
        alt tok {
          token::LPAREN | token::AT | token::TILDE | token::BINOP(_) {
            true
          }
          _ {
            false
          }
        }
    }

    fn parse_item(+attrs: [attribute], vis: visibility)
        -> option<@item> {
        let lo = self.span.lo;
        let (ident, item_, extra_attrs) = if eat_keyword(self, "const") {
            self.parse_item_const()
        } else if is_keyword(self, "fn") &&
            !self.fn_expr_lookahead(self.look_ahead(1u)) {
            self.bump();
            self.parse_item_fn(impure_fn)
        } else if eat_keyword(self, "pure") {
            expect_keyword(self, "fn");
            self.parse_item_fn(pure_fn)
        } else if is_keyword(self, "unsafe")
            && self.look_ahead(1u) != token::LBRACE {
            self.bump();
            expect_keyword(self, "fn");
            self.parse_item_fn(unsafe_fn)
        } else if eat_keyword(self, "crust") {
            expect_keyword(self, "fn");
            self.parse_item_fn(crust_fn)
        } else if eat_keyword(self, "mod") {
            self.parse_item_mod()
        } else if eat_keyword(self, "native") {
            self.parse_item_native_mod()
        } else if eat_keyword(self, "type") {
            self.parse_item_type()
        } else if eat_keyword(self, "enum") {
            self.parse_item_enum(vis)
        } else if eat_keyword(self, "iface") {
            self.parse_item_iface()
        } else if eat_keyword(self, "impl") {
            self.parse_item_impl()
        } else if eat_keyword(self, "resource") {
            self.parse_item_res()
        } else if eat_keyword(self, "class") {
            self.parse_item_class()
        } else { ret none; };
        some(self.mk_item(lo, self.last_span.hi, ident, item_, vis,
                          alt extra_attrs {
                              some(as) { attrs + as }
                              none { attrs }
                          }))
    }

    fn parse_use() -> view_item_ {
        let ident = parse_ident(self);
        let metadata = parse_optional_meta(self);
        ret view_item_use(ident, metadata, self.get_id());
    }

    fn parse_view_path() -> @view_path {
        let lo = self.span.lo;
        let first_ident = parse_ident(self);
        let mut path = [first_ident];
        #debug("parsed view_path: %s", first_ident);
        alt self.token {
          token::EQ {
            // x = foo::bar
            self.bump();
            path = [parse_ident(self)];
            while self.token == token::MOD_SEP {
                self.bump();
                let id = parse_ident(self);
                path += [id];
            }
            let path = @{span: mk_sp(lo, self.span.hi), global: false,
                         idents: path, rp: none, types: []};
            ret @spanned(lo, self.span.hi,
                         view_path_simple(first_ident, path, self.get_id()));
          }

          token::MOD_SEP {
            // foo::bar or foo::{a,b,c} or foo::*
            while self.token == token::MOD_SEP {
                self.bump();

                alt self.token {

                  token::IDENT(i, _) {
                    self.bump();
                    path += [self.get_str(i)];
                  }

                  // foo::bar::{a,b,c}
                  token::LBRACE {
                    let idents =
                        parse_seq(token::LBRACE, token::RBRACE,
                                  seq_sep(token::COMMA), self,
                                  {|p| parse_path_list_ident(p)}).node;
                    let path = @{span: mk_sp(lo, self.span.hi),
                                 global: false, idents: path,
                                 rp: none, types: []};
                    ret @spanned(lo, self.span.hi,
                                 view_path_list(path, idents, self.get_id()));
                  }

                  // foo::bar::*
                  token::BINOP(token::STAR) {
                    self.bump();
                    let path = @{span: mk_sp(lo, self.span.hi),
                                 global: false, idents: path,
                                 rp: none, types: []};
                    ret @spanned(lo, self.span.hi,
                                 view_path_glob(path, self.get_id()));
                  }

                  _ { break; }
                }
            }
          }
          _ { }
        }
        let last = path[vec::len(path) - 1u];
        let path = @{span: mk_sp(lo, self.span.hi), global: false,
                     idents: path, rp: none, types: []};
        ret @spanned(lo, self.span.hi,
                     view_path_simple(last, path, self.get_id()));
    }

    fn parse_view_paths() -> [@view_path] {
        let mut vp = [self.parse_view_path()];
        while self.token == token::COMMA {
            self.bump();
            vp += [self.parse_view_path()];
        }
        ret vp;
    }

    fn is_view_item() -> bool {
        let tok = if !is_keyword(self, "pub") && !is_keyword(self, "priv") {
            self.token
        } else { self.look_ahead(1u) };
        token_is_keyword(self, "use", tok)
            || token_is_keyword(self, "import", tok)
            || token_is_keyword(self, "export", tok)
    }

    fn parse_view_item(+attrs: [attribute]) -> @view_item {
        let lo = self.span.lo, vis = self.parse_visibility(private);
        let node = if eat_keyword(self, "use") {
            self.parse_use()
        } else if eat_keyword(self, "import") {
            view_item_import(self.parse_view_paths())
        } else if eat_keyword(self, "export") {
            view_item_export(self.parse_view_paths())
        } else { fail; };
        expect(self, token::SEMI);
        @{node: node, attrs: attrs,
          vis: vis, span: mk_sp(lo, self.last_span.hi)}
    }

    fn parse_view(+first_item_attrs: [attribute],
                  only_imports: bool) -> {attrs_remaining: [attribute],
                                          view_items: [@view_item]} {
        let mut attrs = first_item_attrs + parse_outer_attributes(self);
        let mut items = [];
        while if only_imports { is_keyword(self, "import") }
        else { self.is_view_item() } {
            items += [self.parse_view_item(attrs)];
            attrs = parse_outer_attributes(self);
        }
        {attrs_remaining: attrs, view_items: items}
    }

    // Parses a source module as a crate
    fn parse_crate_mod(_cfg: crate_cfg) -> @crate {
        let lo = self.span.lo;
        let crate_attrs = parse_inner_attrs_and_next(self);
        let first_item_outer_attrs = crate_attrs.next;
        let m = self.parse_mod_items(token::EOF, first_item_outer_attrs);
        ret @spanned(lo, self.span.lo,
                     {directives: [],
                      module: m,
                      attrs: crate_attrs.inner,
                      config: self.cfg});
    }

    fn parse_str() -> str {
        alt self.token {
          token::LIT_STR(s) { self.bump(); self.get_str(s) }
          _ {
            self.fatal("expected string literal")
          }
        }
    }

    // Logic for parsing crate files (.rc)
    //
    // Each crate file is a sequence of directives.
    //
    // Each directive imperatively extends its environment with 0 or more
    // items.
    fn parse_crate_directive(first_outer_attr: [attribute]) ->
        crate_directive {

        // Collect the next attributes
        let outer_attrs = first_outer_attr + parse_outer_attributes(self);
        // In a crate file outer attributes are only going to apply to mods
        let expect_mod = vec::len(outer_attrs) > 0u;

        let lo = self.span.lo;
        if expect_mod || is_keyword(self, "mod") {
            expect_keyword(self, "mod");
            let id = parse_ident(self);
            alt self.token {
              // mod x = "foo.rs";
              token::SEMI {
                let mut hi = self.span.hi;
                self.bump();
                ret spanned(lo, hi, cdir_src_mod(id, outer_attrs));
              }
              // mod x = "foo_dir" { ...directives... }
              token::LBRACE {
                self.bump();
                let inner_attrs = parse_inner_attrs_and_next(self);
                let mod_attrs = outer_attrs + inner_attrs.inner;
                let next_outer_attr = inner_attrs.next;
                let cdirs = self.parse_crate_directives(token::RBRACE,
                                                        next_outer_attr);
                let mut hi = self.span.hi;
                expect(self, token::RBRACE);
                ret spanned(lo, hi,
                            cdir_dir_mod(id, cdirs, mod_attrs));
              }
              _ { unexpected(self); }
            }
        } else if self.is_view_item() {
            let vi = self.parse_view_item(outer_attrs);
            ret spanned(lo, vi.span.hi, cdir_view_item(vi));
        } else { ret self.fatal("expected crate directive"); }
    }

    fn parse_crate_directives(term: token::token,
                              first_outer_attr: [attribute]) ->
        [@crate_directive] {

        // This is pretty ugly. If we have an outer attribute then we can't
        // accept seeing the terminator next, so if we do see it then fail the
        // same way parse_crate_directive would
        if vec::len(first_outer_attr) > 0u && self.token == term {
            expect_keyword(self, "mod");
        }

        let mut cdirs: [@crate_directive] = [];
        let mut first_outer_attr = first_outer_attr;
        while self.token != term {
            let cdir = @self.parse_crate_directive(first_outer_attr);
            cdirs += [cdir];
            first_outer_attr = [];
        }
        ret cdirs;
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

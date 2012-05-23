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
export mk_item;
export restriction;
export parser;
export parse_crate_directives;
export parse_crate_mod;
export parse_expr;
export parse_item;
export parse_mod_items;
export parse_pat;
export parse_seq;
export parse_stmt;
export parse_ty;
export parse_lit;
export parse_syntax_ext_naked;

// FIXME: #ast expects to find this here but it's actually defined in `parse`
// Fixing this will be easier when we have export decls on individual items --
// then parse can export this publicly, and everything else crate-visibly.
// (See #1893)
import parse_from_source_str;
export parse_from_source_str;

enum restriction {
    UNRESTRICTED,
    RESTRICT_STMT_EXPR,
    RESTRICT_NO_CALL_EXPRS,
    RESTRICT_NO_BAR_OP,
}

enum file_type { CRATE_FILE, SOURCE_FILE, }

type parser = @{
    sess: parse_sess,
    cfg: crate_cfg,
    file_type: file_type,
    mut token: token::token,
    mut span: span,
    mut last_span: span,
    buffer: dvec<{tok: token::token, span: span}>,
    mut restriction: restriction,
    reader: reader,
    keywords: hashmap<str, ()>,
    restricted_keywords: hashmap<str, ()>
};

impl parser for parser {
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
}

fn parse_ty_fn(p: parser) -> fn_decl {
    fn parse_fn_input_ty(p: parser) -> arg {
        let mode = parse_arg_mode(p);
        let name = if is_plain_ident(p.token)
            && p.look_ahead(1u) == token::COLON {

            let name = parse_value_ident(p);
            p.bump();
            name
        } else { "" };
        ret {mode: mode, ty: parse_ty(p, false), ident: name, id: p.get_id()};
    }
    let inputs =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_fn_input_ty, p);
    // FIXME: constrs is empty because right now, higher-order functions
    // can't have constrained types.
    // Not sure whether that would be desirable anyway. See #34 for the
    // story on constrained types.
    let constrs: [@constr] = [];
    let (ret_style, ret_ty) = parse_ret_ty(p);
    ret {inputs: inputs.node, output: ret_ty,
         purity: impure_fn, cf: ret_style,
         constraints: constrs};
}

fn parse_ty_methods(p: parser) -> [ty_method] {
    parse_seq(token::LBRACE, token::RBRACE, seq_sep_none(), {|p|
        let attrs = parse_outer_attributes(p);
        let flo = p.span.lo;
        let pur = parse_fn_purity(p);
        let ident = parse_method_name(p);
        let tps = parse_ty_params(p);
        let d = parse_ty_fn(p), fhi = p.last_span.hi;
        expect(p, token::SEMI);
        {ident: ident, attrs: attrs, decl: {purity: pur with d}, tps: tps,
         span: mk_sp(flo, fhi)}
    }, p).node
}

fn parse_mt(p: parser) -> mt {
    let mutbl = parse_mutability(p);
    let t = parse_ty(p, false);
    ret {ty: t, mutbl: mutbl};
}

fn parse_ty_field(p: parser) -> ty_field {
    let lo = p.span.lo;
    let mutbl = parse_mutability(p);
    let id = parse_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    ret spanned(lo, ty.span.hi, {ident: id, mt: {ty: ty, mutbl: mutbl}});
}

// if i is the jth ident in args, return j
// otherwise, fail
fn ident_index(p: parser, args: [arg], i: ident) -> uint {
    let mut j = 0u;
    for args.each {|a| if a.ident == i { ret j; } j += 1u; }
    p.fatal("unbound variable `" + i + "` in constraint arg");
}

fn parse_type_constr_arg(p: parser) -> @ty_constr_arg {
    let sp = p.span;
    let mut carg = carg_base;
    expect(p, token::BINOP(token::STAR));
    if p.token == token::DOT {
        // "*..." notation for record fields
        p.bump();
        let pth = parse_path_without_tps(p);
        carg = carg_ident(pth);
    }
    // No literals yet, I guess?
    ret @{node: carg, span: sp};
}

fn parse_constr_arg(args: [arg], p: parser) -> @constr_arg {
    let sp = p.span;
    let mut carg = carg_base;
    if p.token == token::BINOP(token::STAR) {
        p.bump();
    } else {
        let i: ident = parse_value_ident(p);
        carg = carg_ident(ident_index(p, args, i));
    }
    ret @{node: carg, span: sp};
}

fn parse_ty_constr(fn_args: [arg], p: parser) -> @constr {
    let lo = p.span.lo;
    let path = parse_path_without_tps(p);
    let args: {node: [@constr_arg], span: span} =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  {|p| parse_constr_arg(fn_args, p)}, p);
    ret @spanned(lo, args.span.hi,
                 {path: path, args: args.node, id: p.get_id()});
}

fn parse_constr_in_type(p: parser) -> @ty_constr {
    let lo = p.span.lo;
    let path = parse_path_without_tps(p);
    let args: [@ty_constr_arg] =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_type_constr_arg, p).node;
    let hi = p.span.lo;
    let tc: ty_constr_ = {path: path, args: args, id: p.get_id()};
    ret @spanned(lo, hi, tc);
}


fn parse_constrs<T: copy>(pser: fn(parser) -> @constr_general<T>,
                         p: parser) ->
   [@constr_general<T>] {
    let mut constrs: [@constr_general<T>] = [];
    loop {
        let constr = pser(p);
        constrs += [constr];
        if p.token == token::COMMA { p.bump(); } else { ret constrs; }
    };
}

fn parse_type_constraints(p: parser) -> [@ty_constr] {
    ret parse_constrs(parse_constr_in_type, p);
}

fn parse_ret_ty(p: parser) -> (ret_style, @ty) {
    ret if eat(p, token::RARROW) {
        let lo = p.span.lo;
        if eat(p, token::NOT) {
            (noreturn, @{id: p.get_id(),
                              node: ty_bot,
                              span: mk_sp(lo, p.last_span.hi)})
        } else {
            (return_val, parse_ty(p, false))
        }
    } else {
        let pos = p.span.lo;
        (return_val, @{id: p.get_id(),
                            node: ty_nil,
                            span: mk_sp(pos, pos)})
    }
}

fn region_from_name(p: parser, s: option<str>) -> @region {
    let r = alt s {
      some (string) { re_named(string) }
      none { re_anon }
    };

    @{id: p.get_id(), node: r}
}

// Parses something like "&x"
fn parse_region(p: parser) -> @region {
    expect(p, token::BINOP(token::AND));
    alt p.token {
      token::IDENT(sid, _) {
        p.bump();
        let n = p.get_str(sid);
        region_from_name(p, some(n))
      }
      _ {
        region_from_name(p, none)
      }
    }
}

// Parses something like "&x." (note the trailing dot)
fn parse_region_dot(p: parser) -> @region {
    let name =
        alt p.token {
          token::IDENT(sid, _) if p.look_ahead(1u) == token::DOT {
            p.bump(); p.bump();
            some(p.get_str(sid))
          }
          _ { none }
        };
    region_from_name(p, name)
}

fn parse_ty(p: parser, colons_before_params: bool) -> @ty {
    let lo = p.span.lo;

    alt maybe_parse_dollar_mac(p) {
      some(e) {
        ret @{id: p.get_id(),
              node: ty_mac(spanned(lo, p.span.hi, e)),
              span: mk_sp(lo, p.span.hi)};
      }
      none {}
    }

    let t = if p.token == token::LPAREN {
        p.bump();
        if p.token == token::RPAREN {
            p.bump();
            ty_nil
        } else {
            let mut ts = [parse_ty(p, false)];
            while p.token == token::COMMA {
                p.bump();
                ts += [parse_ty(p, false)];
            }
            let t = if vec::len(ts) == 1u { ts[0].node }
                    else { ty_tup(ts) };
            expect(p, token::RPAREN);
            t
        }
    } else if p.token == token::AT {
        p.bump();
        ty_box(parse_mt(p))
    } else if p.token == token::TILDE {
        p.bump();
        ty_uniq(parse_mt(p))
    } else if p.token == token::BINOP(token::STAR) {
        p.bump();
        ty_ptr(parse_mt(p))
    } else if p.token == token::LBRACE {
        let elems =
            parse_seq(token::LBRACE, token::RBRACE, seq_sep_opt(token::COMMA),
                      parse_ty_field, p);
        if vec::len(elems.node) == 0u { unexpected_last(p, token::RBRACE); }
        let hi = elems.span.hi;

        let t = ty_rec(elems.node);
        if p.token == token::COLON {
            p.bump();
            ty_constr(@{id: p.get_id(),
                             node: t,
                             span: mk_sp(lo, hi)},
                           parse_type_constraints(p))
        } else { t }
    } else if p.token == token::LBRACKET {
        expect(p, token::LBRACKET);
        let t = ty_vec(parse_mt(p));
        expect(p, token::RBRACKET);
        t
    } else if p.token == token::BINOP(token::AND) {
        p.bump();
        let region = parse_region_dot(p);
        let mt = parse_mt(p);
        ty_rptr(region, mt)
    } else if eat_keyword(p, "fn") {
        let proto = parse_fn_ty_proto(p);
        alt proto {
          proto_bare { p.warn("fn is deprecated, use native fn"); }
          _ { /* fallthrough */ }
        }
        ty_fn(proto, parse_ty_fn(p))
    } else if eat_keyword(p, "native") {
        expect_keyword(p, "fn");
        ty_fn(proto_bare, parse_ty_fn(p))
    } else if p.token == token::MOD_SEP || is_ident(p.token) {
        let path = parse_path_with_tps(p, colons_before_params);
        ty_path(path, p.get_id())
    } else { p.fatal("expecting type"); };

    fn mk_ty(p: parser, t: ty_, lo: uint, hi: uint) -> @ty {
        @{id: p.get_id(),
          node: t,
          span: mk_sp(lo, hi)}
    }

    let ty = mk_ty(p, t, lo, p.last_span.hi);

    // Consider a vstore suffix like /@ or /~
    alt maybe_parse_vstore(p) {
      none {
        ret ty;
      }
      some(v) {
        let t1 = ty_vstore(ty, v);
        ret mk_ty(p, t1, lo, p.last_span.hi);
      }
    }
}

fn parse_arg_mode(p: parser) -> mode {
    if eat(p, token::BINOP(token::AND)) {
        expl(by_mutbl_ref)
    } else if eat(p, token::BINOP(token::MINUS)) {
        expl(by_move)
    } else if eat(p, token::ANDAND) {
        expl(by_ref)
    } else if eat(p, token::BINOP(token::PLUS)) {
        if eat(p, token::BINOP(token::PLUS)) {
            expl(by_val)
        } else {
            expl(by_copy)
        }
    } else { infer(p.get_id()) }
}

fn parse_capture_item_or(
    p: parser,
    parse_arg_fn: fn() -> arg_or_capture_item) -> arg_or_capture_item {

    fn parse_capture_item(p: parser, is_move: bool) -> capture_item {
        let id = p.get_id();
        let sp = mk_sp(p.span.lo, p.span.hi);
        let ident = parse_ident(p);
        {id: id, is_move: is_move, name: ident, span: sp}
    }

    if eat_keyword(p, "move") {
        either::right(parse_capture_item(p, true))
    } else if eat_keyword(p, "copy") {
        either::right(parse_capture_item(p, false))
    } else {
        parse_arg_fn()
    }
}

fn parse_arg(p: parser) -> arg_or_capture_item {
    let m = parse_arg_mode(p);
    let i = parse_value_ident(p);
    expect(p, token::COLON);
    let t = parse_ty(p, false);
    either::left({mode: m, ty: t, ident: i, id: p.get_id()})
}

fn parse_arg_or_capture_item(p: parser) -> arg_or_capture_item {
    parse_capture_item_or(p) {|| parse_arg(p) }
}

fn parse_fn_block_arg(p: parser) -> arg_or_capture_item {
    parse_capture_item_or(p) {||
        let m = parse_arg_mode(p);
        let i = parse_value_ident(p);
        let t = if eat(p, token::COLON) {
                    parse_ty(p, false)
                } else {
                    @{id: p.get_id(),
                      node: ty_infer,
                      span: mk_sp(p.span.lo, p.span.hi)}
                };
        either::left({mode: m, ty: t, ident: i, id: p.get_id()})
    }
}

fn maybe_parse_dollar_mac(p: parser) -> option<mac_> {
    alt p.token {
      token::DOLLAR {
        let lo = p.span.lo;
        p.bump();
        alt p.token {
          token::LIT_INT(num, ty_i) {
            p.bump();
            some(mac_var(num as uint))
          }
          token::LPAREN {
            p.bump();
            let e = parse_expr(p);
            expect(p, token::RPAREN);
            let hi = p.last_span.hi;
            some(mac_aq(mk_sp(lo,hi), e))
          }
          _ {
            p.fatal("expected `(` or integer literal");
          }
        }
      }
      _ {none}
    }
}

fn maybe_parse_vstore(p: parser) -> option<vstore> {
    if p.token == token::BINOP(token::SLASH) {
        p.bump();
        alt p.token {
          token::AT {
            p.bump(); some(vstore_box)
          }
          token::TILDE {
            p.bump(); some(vstore_uniq)
          }
          token::UNDERSCORE {
            p.bump(); some(vstore_fixed(none))
          }
          token::LIT_INT(i, ty_i) if i >= 0i64 {
            p.bump(); some(vstore_fixed(some(i as uint)))
          }
          token::BINOP(token::AND) {
            some(vstore_slice(parse_region(p)))
          }
          _ {
            none
          }
        }
    } else {
        none
    }
}

fn lit_from_token(p: parser, tok: token::token) -> lit_ {
    alt tok {
      token::LIT_INT(i, it) { lit_int(i, it) }
      token::LIT_UINT(u, ut) { lit_uint(u, ut) }
      token::LIT_FLOAT(s, ft) { lit_float(p.get_str(s), ft) }
      token::LIT_STR(s) { lit_str(p.get_str(s)) }
      token::LPAREN { expect(p, token::RPAREN); lit_nil }
      _ { unexpected_last(p, tok); }
    }
}

fn parse_lit(p: parser) -> lit {
    let lo = p.span.lo;
    let lit = if eat_keyword(p, "true") {
        lit_bool(true)
    } else if eat_keyword(p, "false") {
        lit_bool(false)
    } else {
        let tok = p.token;
        p.bump();
        lit_from_token(p, tok)
    };
    ret {node: lit, span: mk_sp(lo, p.last_span.hi)};
}

fn parse_path_without_tps(p: parser) -> @path {
    parse_path_without_tps_(p, parse_ident, parse_ident)
}

fn parse_path_without_tps_(
    p: parser, parse_ident: fn(parser) -> ident,
    parse_last_ident: fn(parser) -> ident) -> @path {

    let lo = p.span.lo;
    let global = eat(p, token::MOD_SEP);
    let mut ids = [];
    loop {
        let is_not_last =
            p.look_ahead(2u) != token::LT
            && p.look_ahead(1u) == token::MOD_SEP;

        if is_not_last {
            ids += [parse_ident(p)];
            expect(p, token::MOD_SEP);
        } else {
            ids += [parse_last_ident(p)];
            break;
        }
    }
    @{span: mk_sp(lo, p.last_span.hi), global: global,
      idents: ids, rp: none, types: []}
}

fn parse_value_path(p: parser) -> @path {
    parse_path_without_tps_(p, parse_ident, parse_value_ident)
}

fn parse_path_with_tps(p: parser, colons: bool) -> @path {
    #debug["parse_path_with_tps(colons=%b)", colons];

    let lo = p.span.lo;
    let path = parse_path_without_tps(p);
    if colons && !eat(p, token::MOD_SEP) {
        ret path;
    }

    // Parse the region parameter, if any, which will
    // be written "foo/&x"
    let rp = {
        // Hack: avoid parsing vstores like /@ and /~.  This is painful
        // because the notation for region bounds and the notation for vstores
        // is... um... the same.  I guess that's my fault.  This is still not
        // ideal as for str/& we end up parsing more than we ought to and have
        // to sort it out later.
        if p.token == token::BINOP(token::SLASH)
            && p.look_ahead(1u) == token::BINOP(token::AND) {

            expect(p, token::BINOP(token::SLASH));
            some(parse_region(p))
        } else {
            none
        }
    };

    // Parse any type parameters which may appear:
    let tps = {
        if p.token == token::LT {
            parse_seq_lt_gt(some(token::COMMA), {|p| parse_ty(p, false)}, p)
        } else {
            {node: [], span: path.span}
        }
    };

    ret @{span: mk_sp(lo, tps.span.hi),
          rp: rp,
          types: tps.node with *path};
}

fn parse_mutability(p: parser) -> mutability {
    if eat_keyword(p, "mut") {
        m_mutbl
    } else if eat_keyword(p, "mut") {
        m_mutbl
    } else if eat_keyword(p, "const") {
        m_const
    } else {
        m_imm
    }
}

fn parse_field(p: parser, sep: token::token) -> field {
    let lo = p.span.lo;
    let m = parse_mutability(p);
    let i = parse_ident(p);
    expect(p, sep);
    let e = parse_expr(p);
    ret spanned(lo, e.span.hi, {mutbl: m, ident: i, expr: e});
}

fn mk_expr(p: parser, lo: uint, hi: uint, +node: expr_) -> @expr {
    ret @{id: p.get_id(), node: node, span: mk_sp(lo, hi)};
}

fn mk_mac_expr(p: parser, lo: uint, hi: uint, m: mac_) -> @expr {
    ret @{id: p.get_id(),
          node: expr_mac({node: m, span: mk_sp(lo, hi)}),
          span: mk_sp(lo, hi)};
}

fn mk_lit_u32(p: parser, i: u32) -> @expr {
    let span = p.span;
    let lv_lit = @{node: lit_uint(i as u64, ty_u32),
                   span: span};

    ret @{id: p.get_id(), node: expr_lit(lv_lit), span: span};
}

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

fn mk_pexpr(p: parser, lo: uint, hi: uint, node: expr_) -> pexpr {
    ret pexpr(mk_expr(p, lo, hi, node));
}

fn to_expr(e: pexpr) -> @expr {
    alt e.node {
      expr_tup(es) if vec::len(es) == 1u { es[0u] }
      _ { *e }
    }
}

fn parse_bottom_expr(p: parser) -> pexpr {
    let lo = p.span.lo;
    let mut hi = p.span.hi;

    let mut ex: expr_;

    alt maybe_parse_dollar_mac(p) {
      some(x) {ret pexpr(mk_mac_expr(p, lo, p.span.hi, x));}
      _ {}
    }

    if p.token == token::LPAREN {
        p.bump();
        if p.token == token::RPAREN {
            hi = p.span.hi;
            p.bump();
            let lit = @spanned(lo, hi, lit_nil);
            ret mk_pexpr(p, lo, hi, expr_lit(lit));
        }
        let mut es = [parse_expr(p)];
        while p.token == token::COMMA { p.bump(); es += [parse_expr(p)]; }
        hi = p.span.hi;
        expect(p, token::RPAREN);

        // Note: we retain the expr_tup() even for simple
        // parenthesized expressions, but only for a "little while".
        // This is so that wrappers around parse_bottom_expr()
        // can tell whether the expression was parenthesized or not,
        // which affects expr_is_complete().
        ret mk_pexpr(p, lo, hi, expr_tup(es));
    } else if p.token == token::LBRACE {
        p.bump();
        if is_keyword(p, "mut") ||
               is_plain_ident(p.token) && p.look_ahead(1u) == token::COLON {
            let mut fields = [parse_field(p, token::COLON)];
            let mut base = none;
            while p.token != token::RBRACE {
                if eat_keyword(p, "with") {
                    base = some(parse_expr(p)); break;
                }
                expect(p, token::COMMA);
                if p.token == token::RBRACE {
                    // record ends by an optional trailing comma
                    break;
                }
                fields += [parse_field(p, token::COLON)];
            }
            hi = p.span.hi;
            expect(p, token::RBRACE);
            ex = expr_rec(fields, base);
        } else if token::is_bar(p.token) {
            ret pexpr(parse_fn_block_expr(p));
        } else {
            let blk = parse_block_tail(p, lo, default_blk);
            ret mk_pexpr(p, blk.span.lo, blk.span.hi, expr_block(blk));
        }
    } else if eat_keyword(p, "new") {
        expect(p, token::LPAREN);
        let r = parse_expr(p);
        expect(p, token::RPAREN);
        let v = parse_expr(p);
        ret mk_pexpr(p, lo, p.span.hi,
                     expr_new(r, p.get_id(), v));
    } else if eat_keyword(p, "if") {
        ret pexpr(parse_if_expr(p));
    } else if eat_keyword(p, "for") {
        ret pexpr(parse_for_expr(p));
    } else if eat_keyword(p, "while") {
        ret pexpr(parse_while_expr(p));
    } else if eat_keyword(p, "loop") {
        ret pexpr(parse_loop_expr(p));
    } else if eat_keyword(p, "alt") {
        ret pexpr(parse_alt_expr(p));
    } else if eat_keyword(p, "fn") {
        let proto = parse_fn_ty_proto(p);
        alt proto {
          proto_bare { p.fatal("fn expr are deprecated, use fn@"); }
          proto_any { p.fatal("fn* cannot be used in an expression"); }
          _ { /* fallthrough */ }
        }
        ret pexpr(parse_fn_expr(p, proto));
    } else if eat_keyword(p, "unchecked") {
        ret pexpr(parse_block_expr(p, lo, unchecked_blk));
    } else if eat_keyword(p, "unsafe") {
        ret pexpr(parse_block_expr(p, lo, unsafe_blk));
    } else if p.token == token::LBRACKET {
        p.bump();
        let mutbl = parse_mutability(p);
        let es =
            parse_seq_to_end(token::RBRACKET, seq_sep(token::COMMA),
                             parse_expr, p);
        hi = p.span.hi;
        ex = expr_vec(es, mutbl);
    } else if p.token == token::POUND && p.look_ahead(1u) == token::LT {
        p.bump();
        p.bump();
        let ty = parse_ty(p, false);
        expect(p, token::GT);

        /* hack: early return to take advantage of specialized function */
        ret pexpr(mk_mac_expr(p, lo, p.span.hi,
                              mac_embed_type(ty)));
    } else if p.token == token::POUND && p.look_ahead(1u) == token::LBRACE {
        p.bump();
        p.bump();
        let blk = mac_embed_block(
            parse_block_tail(p, lo, default_blk));
        ret pexpr(mk_mac_expr(p, lo, p.span.hi, blk));
    } else if p.token == token::ELLIPSIS {
        p.bump();
        ret pexpr(mk_mac_expr(p, lo, p.span.hi, mac_ellipsis));
    } else if p.token == token::POUND {
        let ex_ext = parse_syntax_ext(p);
        hi = ex_ext.span.hi;
        ex = ex_ext.node;
    } else if eat_keyword(p, "bind") {
        let e = parse_expr_res(p, RESTRICT_NO_CALL_EXPRS);
        let es =
            parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                      parse_expr_or_hole, p);
        hi = es.span.hi;
        ex = expr_bind(e, es.node);
    } else if eat_keyword(p, "fail") {
        if can_begin_expr(p.token) {
            let e = parse_expr(p);
            hi = e.span.hi;
            ex = expr_fail(some(e));
        } else { ex = expr_fail(none); }
    } else if eat_keyword(p, "log") {
        expect(p, token::LPAREN);
        let lvl = parse_expr(p);
        expect(p, token::COMMA);
        let e = parse_expr(p);
        ex = expr_log(2, lvl, e);
        hi = p.span.hi;
        expect(p, token::RPAREN);
    } else if eat_keyword(p, "assert") {
        let e = parse_expr(p);
        ex = expr_assert(e);
        hi = e.span.hi;
    } else if eat_keyword(p, "check") {
        /* Should be a predicate (pure boolean function) applied to
           arguments that are all either slot variables or literals.
           but the typechecker enforces that. */
        let e = parse_expr(p);
        hi = e.span.hi;
        ex = expr_check(checked_expr, e);
    } else if eat_keyword(p, "claim") {
        /* Same rules as check, except that if check-claims
         is enabled (a command-line flag), then the parser turns
        claims into check */

        let e = parse_expr(p);
        hi = e.span.hi;
        ex = expr_check(claimed_expr, e);
    } else if eat_keyword(p, "ret") {
        if can_begin_expr(p.token) {
            let e = parse_expr(p);
            hi = e.span.hi;
            ex = expr_ret(some(e));
        } else { ex = expr_ret(none); }
    } else if eat_keyword(p, "break") {
        ex = expr_break;
        hi = p.span.hi;
    } else if eat_keyword(p, "cont") {
        ex = expr_cont;
        hi = p.span.hi;
    } else if eat_keyword(p, "copy") {
        let e = parse_expr(p);
        ex = expr_copy(e);
        hi = e.span.hi;
    } else if p.token == token::MOD_SEP ||
                  is_ident(p.token) && !is_keyword(p, "true") &&
                      !is_keyword(p, "false") {
        let pth = parse_path_with_tps(p, true);
        hi = pth.span.hi;
        ex = expr_path(pth);
    } else {
        let lit = parse_lit(p);
        hi = lit.span.hi;
        ex = expr_lit(@lit);
    }

    // Vstore is legal following expr_lit(lit_str(...)) and expr_vec(...)
    // only.
    alt ex {
      expr_lit(@{node: lit_str(_), span: _}) |
      expr_vec(_, _)  {
        alt maybe_parse_vstore(p) {
          none { }
          some(v) {
            hi = p.span.hi;
            ex = expr_vstore(mk_expr(p, lo, hi, ex), v);
          }
        }
      }
      _ { }
    }

    ret mk_pexpr(p, lo, hi, ex);
}

fn parse_block_expr(p: parser,
                    lo: uint,
                    blk_mode: blk_check_mode) -> @expr {
    expect(p, token::LBRACE);
    let blk = parse_block_tail(p, lo, blk_mode);
    ret mk_expr(p, blk.span.lo, blk.span.hi, expr_block(blk));
}

fn parse_syntax_ext(p: parser) -> @expr {
    let lo = p.span.lo;
    expect(p, token::POUND);
    ret parse_syntax_ext_naked(p, lo);
}

fn parse_syntax_ext_naked(p: parser, lo: uint) -> @expr {
    alt p.token {
      token::IDENT(_, _) {}
      _ { p.fatal("expected a syntax expander name"); }
    }
    let pth = parse_path_without_tps(p);
    //temporary for a backwards-compatible cycle:
    let sep = seq_sep(token::COMMA);
    let mut e = none;
    if (p.token == token::LPAREN || p.token == token::LBRACKET) {
        let es =
            if p.token == token::LPAREN {
                parse_seq(token::LPAREN, token::RPAREN,
                          sep, parse_expr, p)
            } else {
                parse_seq(token::LBRACKET, token::RBRACKET,
                          sep, parse_expr, p)
            };
        let hi = es.span.hi;
        e = some(mk_expr(p, es.span.lo, hi,
                         expr_vec(es.node, m_imm)));
    }
    let mut b = none;
    if p.token == token::LBRACE {
        p.bump();
        let lo = p.span.lo;
        let mut depth = 1u;
        while (depth > 0u) {
            alt (p.token) {
              token::LBRACE {depth += 1u;}
              token::RBRACE {depth -= 1u;}
              token::EOF {p.fatal("unexpected EOF in macro body");}
              _ {}
            }
            p.bump();
        }
        let hi = p.last_span.lo;
        b = some({span: mk_sp(lo,hi)});
    }
    ret mk_mac_expr(p, lo, p.span.hi, mac_invoc(pth, e, b));
}

fn parse_dot_or_call_expr(p: parser) -> pexpr {
    let b = parse_bottom_expr(p);
    parse_dot_or_call_expr_with(p, b)
}

fn permits_call(p: parser) -> bool {
    ret p.restriction != RESTRICT_NO_CALL_EXPRS;
}

fn parse_dot_or_call_expr_with(p: parser, e0: pexpr) -> pexpr {
    let mut e = e0;
    let lo = e.span.lo;
    let mut hi = e.span.hi;
    loop {
        // expr.f
        if eat(p, token::DOT) {
            alt p.token {
              token::IDENT(i, _) {
                hi = p.span.hi;
                p.bump();
                let tys = if eat(p, token::MOD_SEP) {
                    expect(p, token::LT);
                    parse_seq_to_gt(some(token::COMMA),
                                    {|p| parse_ty(p, false)}, p)
                } else { [] };
                e = mk_pexpr(p, lo, hi,
                             expr_field(to_expr(e),
                                             p.get_str(i),
                                             tys));
              }
              _ { unexpected(p); }
            }
            cont;
        }
        if expr_is_complete(p, e) { break; }
        alt p.token {
          // expr(...)
          token::LPAREN if permits_call(p) {
            let es_opt =
                parse_seq(token::LPAREN, token::RPAREN,
                          seq_sep(token::COMMA), parse_expr_or_hole, p);
            hi = es_opt.span.hi;

            let nd =
                if vec::any(es_opt.node, {|e| option::is_none(e) }) {
                    expr_bind(to_expr(e), es_opt.node)
                } else {
                    let es = vec::map(es_opt.node) {|e| option::get(e) };
                    expr_call(to_expr(e), es, false)
                };
            e = mk_pexpr(p, lo, hi, nd);
          }

          // expr {|| ... }
          token::LBRACE if (token::is_bar(p.look_ahead(1u))
                            && permits_call(p)) {
            p.bump();
            let blk = parse_fn_block_expr(p);
            alt e.node {
              expr_call(f, args, false) {
                e = pexpr(@{node: expr_call(f, args + [blk], true)
                            with *to_expr(e)});
              }
              _ {
                e = mk_pexpr(p, lo, p.last_span.hi,
                            expr_call(to_expr(e), [blk], true));
              }
            }
          }

          // expr[...]
          token::LBRACKET {
            p.bump();
            let ix = parse_expr(p);
            hi = ix.span.hi;
            expect(p, token::RBRACKET);
            p.get_id(); // see ast_util::op_expr_callee_id
            e = mk_pexpr(p, lo, hi, expr_index(to_expr(e), ix));
          }

          _ { ret e; }
        }
    }
    ret e;
}

fn parse_prefix_expr(p: parser) -> pexpr {
    let lo = p.span.lo;
    let mut hi = p.span.hi;

    let mut ex;
    alt p.token {
      token::NOT {
        p.bump();
        let e = to_expr(parse_prefix_expr(p));
        hi = e.span.hi;
        p.get_id(); // see ast_util::op_expr_callee_id
        ex = expr_unary(not, e);
      }
      token::BINOP(b) {
        alt b {
          token::MINUS {
            p.bump();
            let e = to_expr(parse_prefix_expr(p));
            hi = e.span.hi;
            p.get_id(); // see ast_util::op_expr_callee_id
            ex = expr_unary(neg, e);
          }
          token::STAR {
            p.bump();
            let e = to_expr(parse_prefix_expr(p));
            hi = e.span.hi;
            ex = expr_unary(deref, e);
          }
          token::AND {
            p.bump();
            let m = parse_mutability(p);
            let e = to_expr(parse_prefix_expr(p));
            hi = e.span.hi;
            ex = expr_addr_of(m, e);
          }
          _ { ret parse_dot_or_call_expr(p); }
        }
      }
      token::AT {
        p.bump();
        let m = parse_mutability(p);
        let e = to_expr(parse_prefix_expr(p));
        hi = e.span.hi;
        ex = expr_unary(box(m), e);
      }
      token::TILDE {
        p.bump();
        let m = parse_mutability(p);
        let e = to_expr(parse_prefix_expr(p));
        hi = e.span.hi;
        ex = expr_unary(uniq(m), e);
      }
      _ { ret parse_dot_or_call_expr(p); }
    }
    ret mk_pexpr(p, lo, hi, ex);
}


fn parse_binops(p: parser) -> @expr {
    ret parse_more_binops(p, parse_prefix_expr(p), 0u);
}

fn parse_more_binops(p: parser, plhs: pexpr, min_prec: uint) ->
   @expr {
    let lhs = to_expr(plhs);
    if expr_is_complete(p, plhs) { ret lhs; }
    let peeked = p.token;
    if peeked == token::BINOP(token::OR) &&
       p.restriction == RESTRICT_NO_BAR_OP { ret lhs; }
    let cur_opt   = token_to_binop(peeked);
    alt cur_opt {
     some(cur_op) {
       let cur_prec = operator_prec(cur_op);
       if cur_prec > min_prec {
          p.bump();
          let expr = parse_prefix_expr(p);
          let rhs = parse_more_binops(p, expr, cur_prec);
          p.get_id(); // see ast_util::op_expr_callee_id
          let bin = mk_pexpr(p, lhs.span.lo, rhs.span.hi,
                            expr_binary(cur_op, lhs, rhs));
          ret parse_more_binops(p, bin, min_prec);
       }
     }
     _ {}
    }
    if as_prec > min_prec && eat_keyword(p, "as") {
        let rhs = parse_ty(p, true);
        let _as =
            mk_pexpr(p, lhs.span.lo, rhs.span.hi, expr_cast(lhs, rhs));
        ret parse_more_binops(p, _as, min_prec);
    }
    ret lhs;
}

fn parse_assign_expr(p: parser) -> @expr {
    let lo = p.span.lo;
    let lhs = parse_binops(p);
    alt p.token {
      token::EQ {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, expr_assign(lhs, rhs));
      }
      token::BINOPEQ(op) {
        p.bump();
        let rhs = parse_expr(p);
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
        p.get_id(); // see ast_util::op_expr_callee_id
        ret mk_expr(p, lo, rhs.span.hi, expr_assign_op(aop, lhs, rhs));
      }
      token::LARROW {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, expr_move(lhs, rhs));
      }
      token::DARROW {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, expr_swap(lhs, rhs));
      }
      _ {/* fall through */ }
    }
    ret lhs;
}

fn parse_if_expr_1(p: parser) ->
   {cond: @expr,
    then: blk,
    els: option<@expr>,
    lo: uint,
    hi: uint} {
    let lo = p.last_span.lo;
    let cond = parse_expr(p);
    let thn = parse_block(p);
    let mut els: option<@expr> = none;
    let mut hi = thn.span.hi;
    if eat_keyword(p, "else") {
        let elexpr = parse_else_expr(p);
        els = some(elexpr);
        hi = elexpr.span.hi;
    }
    ret {cond: cond, then: thn, els: els, lo: lo, hi: hi};
}

fn parse_if_expr(p: parser) -> @expr {
    if eat_keyword(p, "check") {
        let q = parse_if_expr_1(p);
        ret mk_expr(p, q.lo, q.hi, expr_if_check(q.cond, q.then, q.els));
    } else {
        let q = parse_if_expr_1(p);
        ret mk_expr(p, q.lo, q.hi, expr_if(q.cond, q.then, q.els));
    }
}

fn parse_fn_expr(p: parser, proto: proto) -> @expr {
    let lo = p.last_span.lo;

    let cc_old = parse_old_skool_capture_clause(p);

    // if we want to allow fn expression argument types to be inferred in the
    // future, just have to change parse_arg to parse_fn_block_arg.
    let (decl, capture_clause) =
        parse_fn_decl(p, impure_fn, parse_arg_or_capture_item);

    let body = parse_block(p);
    ret mk_expr(p, lo, body.span.hi,
                expr_fn(proto, decl, body,
                             @(*capture_clause + cc_old)));
}

fn parse_fn_block_expr(p: parser) -> @expr {
    let lo = p.last_span.lo;
    let (decl, captures) = parse_fn_block_decl(p);
    let body = parse_block_tail(p, lo, default_blk);
    ret mk_expr(p, lo, body.span.hi,
                expr_fn_block(decl, body, captures));
}

fn parse_else_expr(p: parser) -> @expr {
    if eat_keyword(p, "if") {
        ret parse_if_expr(p);
    } else {
        let blk = parse_block(p);
        ret mk_expr(p, blk.span.lo, blk.span.hi, expr_block(blk));
    }
}

fn parse_for_expr(p: parser) -> @expr {
    let lo = p.last_span;
    let call = parse_expr_res(p, RESTRICT_STMT_EXPR);
    alt call.node {
      expr_call(f, args, true) {
        let b_arg = vec::last(args);
        let last = mk_expr(p, b_arg.span.lo, b_arg.span.hi,
                           expr_loop_body(b_arg));
        @{node: expr_call(f, vec::init(args) + [last], true)
          with *call}
      }
      _ {
        p.span_fatal(lo, "`for` must be followed by a block call");
      }
    }
}

fn parse_while_expr(p: parser) -> @expr {
    let lo = p.last_span.lo;
    let cond = parse_expr(p);
    let body = parse_block_no_value(p);
    let mut hi = body.span.hi;
    ret mk_expr(p, lo, hi, expr_while(cond, body));
}

fn parse_loop_expr(p: parser) -> @expr {
    let lo = p.last_span.lo;
    let body = parse_block_no_value(p);
    let mut hi = body.span.hi;
    ret mk_expr(p, lo, hi, expr_loop(body));
}

fn parse_alt_expr(p: parser) -> @expr {
    let lo = p.last_span.lo;
    let mode = if eat_keyword(p, "check") { alt_check }
               else { alt_exhaustive };
    let discriminant = parse_expr(p);
    expect(p, token::LBRACE);
    let mut arms: [arm] = [];
    while p.token != token::RBRACE {
        let pats = parse_pats(p);
        let mut guard = none;
        if eat_keyword(p, "if") { guard = some(parse_expr(p)); }
        let blk = parse_block(p);
        arms += [{pats: pats, guard: guard, body: blk}];
    }
    let mut hi = p.span.hi;
    p.bump();
    ret mk_expr(p, lo, hi, expr_alt(discriminant, arms, mode));
}

fn parse_expr(p: parser) -> @expr {
    ret parse_expr_res(p, UNRESTRICTED);
}

fn parse_expr_or_hole(p: parser) -> option<@expr> {
    alt p.token {
      token::UNDERSCORE { p.bump(); ret none; }
      _ { ret some(parse_expr(p)); }
    }
}

fn parse_expr_res(p: parser, r: restriction) -> @expr {
    let old = p.restriction;
    p.restriction = r;
    let e = parse_assign_expr(p);
    p.restriction = old;
    ret e;
}

fn parse_initializer(p: parser) -> option<initializer> {
    alt p.token {
      token::EQ {
        p.bump();
        ret some({op: init_assign, expr: parse_expr(p)});
      }
      token::LARROW {
        p.bump();
        ret some({op: init_move, expr: parse_expr(p)});
      }
      // Now that the the channel is the first argument to receive,
      // combining it with an initializer doesn't really make sense.
      // case (token::RECV) {
      //     p.bump();
      //     ret some(rec(op = init_recv,
      //                  expr = parse_expr(p)));
      // }
      _ {
        ret none;
      }
    }
}

fn parse_pats(p: parser) -> [@pat] {
    let mut pats = [];
    loop {
        pats += [parse_pat(p)];
        if p.token == token::BINOP(token::OR) { p.bump(); } else { ret pats; }
    };
}

fn parse_pat(p: parser) -> @pat {
    let lo = p.span.lo;
    let mut hi = p.span.hi;
    let mut pat;
    alt p.token {
      token::UNDERSCORE { p.bump(); pat = pat_wild; }
      token::AT {
        p.bump();
        let sub = parse_pat(p);
        pat = pat_box(sub);
        hi = sub.span.hi;
      }
      token::TILDE {
        p.bump();
        let sub = parse_pat(p);
        pat = pat_uniq(sub);
        hi = sub.span.hi;
      }
      token::LBRACE {
        p.bump();
        let mut fields = [];
        let mut etc = false;
        let mut first = true;
        while p.token != token::RBRACE {
            if first { first = false; } else { expect(p, token::COMMA); }

            if p.token == token::UNDERSCORE {
                p.bump();
                if p.token != token::RBRACE {
                    p.fatal("expecting }, found " +
                                token_to_str(p.reader, p.token));
                }
                etc = true;
                break;
            }

            let lo1 = p.last_span.lo;
            let fieldname = if p.look_ahead(1u) == token::COLON {
                parse_ident(p)
            } else {
                parse_value_ident(p)
            };
            let hi1 = p.last_span.lo;
            let fieldpath = ast_util::ident_to_path(mk_sp(lo1, hi1),
                                          fieldname);
            let mut subpat;
            if p.token == token::COLON {
                p.bump();
                subpat = parse_pat(p);
            } else {
                subpat = @{id: p.get_id(),
                           node: pat_ident(fieldpath, none),
                           span: mk_sp(lo, hi)};
            }
            fields += [{ident: fieldname, pat: subpat}];
        }
        hi = p.span.hi;
        p.bump();
        pat = pat_rec(fields, etc);
      }
      token::LPAREN {
        p.bump();
        if p.token == token::RPAREN {
            hi = p.span.hi;
            p.bump();
            let lit = @{node: lit_nil, span: mk_sp(lo, hi)};
            let expr = mk_expr(p, lo, hi, expr_lit(lit));
            pat = pat_lit(expr);
        } else {
            let mut fields = [parse_pat(p)];
            while p.token == token::COMMA {
                p.bump();
                fields += [parse_pat(p)];
            }
            if vec::len(fields) == 1u { expect(p, token::COMMA); }
            hi = p.span.hi;
            expect(p, token::RPAREN);
            pat = pat_tup(fields);
        }
      }
      tok {
        if !is_ident(tok) || is_keyword(p, "true") || is_keyword(p, "false") {
            let val = parse_expr_res(p, RESTRICT_NO_BAR_OP);
            if eat_keyword(p, "to") {
                let end = parse_expr_res(p, RESTRICT_NO_BAR_OP);
                hi = end.span.hi;
                pat = pat_range(val, end);
            } else {
                hi = val.span.hi;
                pat = pat_lit(val);
            }
        } else if is_plain_ident(p.token) &&
            alt p.look_ahead(1u) {
              token::LPAREN | token::LBRACKET | token::LT { false }
              _ { true }
            } {
            let name = parse_value_path(p);
            let sub = if eat(p, token::AT) { some(parse_pat(p)) }
                      else { none };
            pat = pat_ident(name, sub);
        } else {
            let enum_path = parse_path_with_tps(p, true);
            hi = enum_path.span.hi;
            let mut args: [@pat] = [];
            let mut star_pat = false;
            alt p.token {
              token::LPAREN {
                alt p.look_ahead(1u) {
                  token::BINOP(token::STAR) {
                    // This is a "top constructor only" pat
                    p.bump(); p.bump();
                    star_pat = true;
                    expect(p, token::RPAREN);
                  }
                  _ {
                   let a =
                       parse_seq(token::LPAREN, token::RPAREN,
                                seq_sep(token::COMMA), parse_pat, p);
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
    ret @{id: p.get_id(), node: pat, span: mk_sp(lo, hi)};
}

fn parse_local(p: parser, is_mutbl: bool,
               allow_init: bool) -> @local {
    let lo = p.span.lo;
    let pat = parse_pat(p);
    let mut ty = @{id: p.get_id(),
                   node: ty_infer,
                   span: mk_sp(lo, lo)};
    if eat(p, token::COLON) { ty = parse_ty(p, false); }
    let init = if allow_init { parse_initializer(p) } else { none };
    ret @spanned(lo, p.last_span.hi,
                 {is_mutbl: is_mutbl, ty: ty, pat: pat,
                  init: init, id: p.get_id()});
}

fn parse_let(p: parser) -> @decl {
    let is_mutbl = eat_keyword(p, "mut");
    let lo = p.span.lo;
    let mut locals = [parse_local(p, is_mutbl, true)];
    while eat(p, token::COMMA) {
        locals += [parse_local(p, is_mutbl, true)];
    }
    ret @spanned(lo, p.last_span.hi, decl_local(locals));
}

/* assumes "let" token has already been consumed */
fn parse_instance_var(p:parser, pr: visibility) -> @class_member {
    let mut is_mutbl = class_immutable;
    let lo = p.span.lo;
    if eat_keyword(p, "mut") {
        is_mutbl = class_mutable;
    }
    if !is_plain_ident(p.token) {
        p.fatal("expecting ident");
    }
    let name = parse_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    ret @{node: instance_var(name, ty, is_mutbl, p.get_id(), pr),
          span: mk_sp(lo, p.last_span.hi)};
}

fn parse_stmt(p: parser, +first_item_attrs: [attribute]) -> @stmt {
    fn check_expected_item(p: parser, current_attrs: [attribute]) {
        // If we have attributes then we should have an item
        if vec::is_not_empty(current_attrs) {
            p.fatal("expected item");
        }
    }

    let lo = p.span.lo;
    if is_keyword(p, "let") {
        check_expected_item(p, first_item_attrs);
        expect_keyword(p, "let");
        let decl = parse_let(p);
        ret @spanned(lo, decl.span.hi, stmt_decl(decl, p.get_id()));
    } else {
        let mut item_attrs;
        alt parse_outer_attrs_or_ext(p, first_item_attrs) {
          none { item_attrs = []; }
          some(left(attrs)) { item_attrs = attrs; }
          some(right(ext)) {
            ret @spanned(lo, ext.span.hi, stmt_expr(ext, p.get_id()));
          }
        }

        let item_attrs = first_item_attrs + item_attrs;

        alt parse_item(p, item_attrs, public) {
          some(i) {
            let mut hi = i.span.hi;
            let decl = @spanned(lo, hi, decl_item(i));
            ret @spanned(lo, hi, stmt_decl(decl, p.get_id()));
          }
          none() { /* fallthrough */ }
        }

        check_expected_item(p, item_attrs);

        // Remainder are line-expr stmts.
        let e = parse_expr_res(p, RESTRICT_STMT_EXPR);
        ret @spanned(lo, e.span.hi, stmt_expr(e, p.get_id()));
    }
}

fn expr_is_complete(p: parser, e: pexpr) -> bool {
    log(debug, ("expr_is_complete", p.restriction,
                print::pprust::expr_to_str(*e),
                classify::expr_requires_semi_to_be_stmt(*e)));
    ret p.restriction == RESTRICT_STMT_EXPR &&
        !classify::expr_requires_semi_to_be_stmt(*e);
}

fn parse_block(p: parser) -> blk {
    let (attrs, blk) = parse_inner_attrs_and_block(p, false);
    assert vec::is_empty(attrs);
    ret blk;
}

fn parse_inner_attrs_and_block(
    p: parser, parse_attrs: bool) -> ([attribute], blk) {

    fn maybe_parse_inner_attrs_and_next(
        p: parser, parse_attrs: bool) ->
        {inner: [attribute], next: [attribute]} {
        if parse_attrs {
            parse_inner_attrs_and_next(p)
        } else {
            {inner: [], next: []}
        }
    }

    let lo = p.span.lo;
    if eat_keyword(p, "unchecked") {
        expect(p, token::LBRACE);
        let {inner, next} = maybe_parse_inner_attrs_and_next(p, parse_attrs);
        ret (inner, parse_block_tail_(p, lo, unchecked_blk, next));
    } else if eat_keyword(p, "unsafe") {
        expect(p, token::LBRACE);
        let {inner, next} = maybe_parse_inner_attrs_and_next(p, parse_attrs);
        ret (inner, parse_block_tail_(p, lo, unsafe_blk, next));
    } else {
        expect(p, token::LBRACE);
        let {inner, next} = maybe_parse_inner_attrs_and_next(p, parse_attrs);
        ret (inner, parse_block_tail_(p, lo, default_blk, next));
    }
}

fn parse_block_no_value(p: parser) -> blk {
    // We parse blocks that cannot have a value the same as any other block;
    // the type checker will make sure that the tail expression (if any) has
    // unit type.
    ret parse_block(p);
}

// Precondition: already parsed the '{' or '#{'
// I guess that also means "already parsed the 'impure'" if
// necessary, and this should take a qualifier.
// some blocks start with "#{"...
fn parse_block_tail(p: parser, lo: uint, s: blk_check_mode) -> blk {
    parse_block_tail_(p, lo, s, [])
}

fn parse_block_tail_(p: parser, lo: uint, s: blk_check_mode,
                     +first_item_attrs: [attribute]) -> blk {
    let mut stmts = [];
    let mut expr = none;
    let {attrs_remaining, view_items} = parse_view(p, first_item_attrs, true);
    let mut initial_attrs = attrs_remaining;

    if p.token == token::RBRACE && !vec::is_empty(initial_attrs) {
        p.fatal("expected item");
    }

    while p.token != token::RBRACE {
        alt p.token {
          token::SEMI {
            p.bump(); // empty
          }
          _ {
            let stmt = parse_stmt(p, initial_attrs);
            initial_attrs = [];
            alt stmt.node {
              stmt_expr(e, stmt_id) { // Expression without semicolon:
                alt p.token {
                  token::SEMI {
                    p.bump();
                    stmts += [@{node: stmt_semi(e, stmt_id) with *stmt}];
                  }
                  token::RBRACE {
                    expr = some(e);
                  }
                  t {
                    if classify::stmt_ends_with_semi(*stmt) {
                        p.fatal("expected ';' or '}' after expression but \
                                 found '" + token_to_str(p.reader, t) +
                                "'");
                    }
                    stmts += [stmt];
                  }
                }
              }

              _ { // All other kinds of statements:
                stmts += [stmt];

                if classify::stmt_ends_with_semi(*stmt) {
                    expect(p, token::SEMI);
                }
              }
            }
          }
        }
    }
    let mut hi = p.span.hi;
    p.bump();
    let bloc = {view_items: view_items, stmts: stmts, expr: expr,
                id: p.get_id(), rules: s};
    ret spanned(lo, hi, bloc);
}

fn parse_ty_param(p: parser) -> ty_param {
    let mut bounds = [];
    let ident = parse_ident(p);
    if eat(p, token::COLON) {
        while p.token != token::COMMA && p.token != token::GT {
            if eat_keyword(p, "send") { bounds += [bound_send]; }
            else if eat_keyword(p, "copy") { bounds += [bound_copy]; }
            else { bounds += [bound_iface(parse_ty(p, false))]; }
        }
    }
    ret {ident: ident, id: p.get_id(), bounds: @bounds};
}

fn parse_ty_params(p: parser) -> [ty_param] {
    if eat(p, token::LT) {
        parse_seq_to_gt(some(token::COMMA), parse_ty_param, p)
    } else { [] }
}

// FIXME Remove after snapshot
fn parse_old_skool_capture_clause(p: parser) -> [capture_item] {
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
                res += [{id:id, is_move: is_move, name:ident, span:sp}];
                if !eat(p, token::COMMA) {
                    ret res;
                }
              }

              _ { ret res; }
            }
        };
    }

    let mut cap_items = [];

    if eat(p, token::LBRACKET) {
        while !eat(p, token::RBRACKET) {
            if eat_keyword(p, "copy") {
                cap_items += eat_ident_list(p, false);
                expect_opt_trailing_semi(p);
            } else if eat_keyword(p, "move") {
                cap_items += eat_ident_list(p, true);
                expect_opt_trailing_semi(p);
            } else {
                let s: str = "expecting send, copy, or move clause";
                p.fatal(s);
            }
        }
    }

    ret cap_items;
}

type arg_or_capture_item = either<arg, capture_item>;


fn parse_fn_decl(p: parser, purity: purity,
                 parse_arg_fn: fn(parser) -> arg_or_capture_item)
    -> (fn_decl, capture_clause) {

    let args_or_capture_items: [arg_or_capture_item] =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_arg_fn, p).node;

    let inputs = either::lefts(args_or_capture_items);
    let capture_clause = @either::rights(args_or_capture_items);

    // Use the args list to translate each bound variable
    // mentioned in a constraint to an arg index.
    // Seems weird to do this in the parser, but I'm not sure how else to.
    let mut constrs = [];
    if p.token == token::COLON {
        p.bump();
        constrs = parse_constrs({|x| parse_ty_constr(inputs, x) }, p);
    }
    let (ret_style, ret_ty) = parse_ret_ty(p);
    ret ({inputs: inputs,
          output: ret_ty,
          purity: purity,
          cf: ret_style,
          constraints: constrs}, capture_clause);
}

fn parse_fn_block_decl(p: parser) -> (fn_decl, capture_clause) {
    let inputs_captures = {
        if eat(p, token::OROR) {
            []
        } else {
            parse_seq(token::BINOP(token::OR),
                      token::BINOP(token::OR),
                      seq_sep(token::COMMA),
                      parse_fn_block_arg, p).node
        }
    };
    let output = if eat(p, token::RARROW) {
                     parse_ty(p, false)
                 } else {
                     @{id: p.get_id(), node: ty_infer, span: p.span}
                 };
    ret ({inputs: either::lefts(inputs_captures),
          output: output,
          purity: impure_fn,
          cf: return_val,
          constraints: []},
         @either::rights(inputs_captures));
}

fn parse_fn_header(p: parser) -> {ident: ident, tps: [ty_param]} {
    let id = parse_value_ident(p);
    let ty_params = parse_ty_params(p);
    ret {ident: id, tps: ty_params};
}

fn mk_item(p: parser, lo: uint, hi: uint, +ident: ident,
           +node: item_, vis: visibility,
           +attrs: [attribute]) -> @item {
    ret @{ident: ident,
          attrs: attrs,
          id: p.get_id(),
          node: node,
          vis: vis,
          span: mk_sp(lo, hi)};
}

type item_info = (ident, item_, option<[attribute]>);

fn parse_item_fn(p: parser, purity: purity) -> item_info {
    let t = parse_fn_header(p);
    let (decl, _) = parse_fn_decl(p, purity, parse_arg);
    let (inner_attrs, body) = parse_inner_attrs_and_block(p, true);
    (t.ident, item_fn(decl, t.tps, body), some(inner_attrs))
}

fn parse_method_name(p: parser) -> ident {
    alt p.token {
      token::BINOP(op) { p.bump(); token::binop_to_str(op) }
      token::NOT { p.bump(); "!" }
      token::LBRACKET { p.bump(); expect(p, token::RBRACKET); "[]" }
      _ {
          let id = parse_value_ident(p);
          if id == "unary" && eat(p, token::BINOP(token::MINUS)) { "unary-" }
          else { id }
      }
    }
}

fn parse_method(p: parser, pr: visibility) -> @method {
    let attrs = parse_outer_attributes(p);
    let lo = p.span.lo, pur = parse_fn_purity(p);
    let ident = parse_method_name(p);
    let tps = parse_ty_params(p);
    let (decl, _) = parse_fn_decl(p, pur, parse_arg);
    let (inner_attrs, body) = parse_inner_attrs_and_block(p, true);
    let attrs = attrs + inner_attrs;
    @{ident: ident, attrs: attrs, tps: tps, decl: decl, body: body,
      id: p.get_id(), span: mk_sp(lo, body.span.hi),
      self_id: p.get_id(), vis: pr}
}

fn parse_item_iface(p: parser) -> item_info {
    let ident = parse_ident(p);
    let rp = parse_region_param(p);
    let tps = parse_ty_params(p);
    let meths = parse_ty_methods(p);
    (ident, item_iface(tps, rp, meths), none)
}

// Parses three variants (with the region/type params always optional):
//    impl /&<T: copy> of to_str for [T] { ... }
//    impl name/&<T> of to_str for [T] { ... }
//    impl name/&<T> for [T] { ... }
fn parse_item_impl(p: parser) -> item_info {
    fn wrap_path(p: parser, pt: @path) -> @ty {
        @{id: p.get_id(), node: ty_path(pt, p.get_id()), span: pt.span}
    }
    let mut (ident, rp, tps) = {
        if p.token == token::LT {
            (none, rp_none, parse_ty_params(p))
        } else if p.token == token::BINOP(token::SLASH) {
            (none, parse_region_param(p), parse_ty_params(p))
        }
        else if is_keyword(p, "of") {
            (none, rp_none, [])
        } else {
            let id = parse_ident(p);
            let rp = parse_region_param(p);
            (some(id), rp, parse_ty_params(p))
        }
    };
    let ifce = if eat_keyword(p, "of") {
        let path = parse_path_with_tps(p, false);
        if option::is_none(ident) {
            ident = some(vec::last(path.idents));
        }
        some(@{path: path, id: p.get_id()})
    } else { none };
    let ident = alt ident {
        some(name) { name }
        none { expect_keyword(p, "of"); fail; }
    };
    expect_keyword(p, "for");
    let ty = parse_ty(p, false);
    let mut meths = [];
    expect(p, token::LBRACE);
    while !eat(p, token::RBRACE) { meths += [parse_method(p, public)]; }
    (ident, item_impl(tps, rp, ifce, ty, meths), none)
}

fn parse_item_res(p: parser) -> item_info {
    let ident = parse_value_ident(p);
    let rp = parse_region_param(p);
    let ty_params = parse_ty_params(p);
    expect(p, token::LPAREN);
    let arg_ident = parse_value_ident(p);
    expect(p, token::COLON);
    let t = parse_ty(p, false);
    expect(p, token::RPAREN);
    let dtor = parse_block_no_value(p);
    let decl = {
        inputs: [{mode: expl(by_ref), ty: t,
                  ident: arg_ident, id: p.get_id()}],
        output: @{id: p.get_id(), node: ty_nil,
                  span: ast_util::dummy_sp()},
        purity: impure_fn,
        cf: return_val,
        constraints: []
    };
    (ident, item_res(decl, ty_params, dtor,
                          p.get_id(), p.get_id(), rp), none)
}

// Instantiates ident <i> with references to <typarams> as arguments.  Used to
// create a path that refers to a class which will be defined as the return
// type of the ctor function.
fn ident_to_path_tys(p: parser, i: ident,
                     rp: region_param,
                     typarams: [ty_param]) -> @path {
    let s = p.last_span;

    // Hack.  But then, this whole function is in service of a hack.
    let a_r = alt rp {
      rp_none { none }
      rp_self { some(region_from_name(p, some("self"))) }
    };

    @{span: s, global: false, idents: [i],
      rp: a_r,
      types: vec::map(typarams, {|tp|
          @{id: p.get_id(),
            node: ty_path(ident_to_path(s, tp.ident), p.get_id()),
            span: s}})
     }
}

fn parse_iface_ref(p:parser) -> @iface_ref {
    @{path: parse_path_with_tps(p, false),
      id: p.get_id()}
}

fn parse_iface_ref_list(p:parser) -> [@iface_ref] {
    parse_seq_to_before_end(token::LBRACE, seq_sep(token::COMMA),
                            parse_iface_ref, p)
}

fn parse_item_class(p: parser) -> item_info {
    let class_name = parse_value_ident(p);
    let rp = parse_region_param(p);
    let ty_params = parse_ty_params(p);
    let class_path = ident_to_path_tys(p, class_name, rp, ty_params);
    let ifaces : [@iface_ref] = if eat_keyword(p, "implements")
                                       { parse_iface_ref_list(p) }
                                    else { [] };
    expect(p, token::LBRACE);
    let mut ms: [@class_member] = [];
    let ctor_id = p.get_id();
    let mut the_ctor : option<(fn_decl, blk, codemap::span)> = none;
    let mut the_dtor : option<(blk, codemap::span)> = none;
    while p.token != token::RBRACE {
        alt parse_class_item(p, class_path) {
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
       {node: {id: p.get_id(),
               self_id: p.get_id(),
               body: d_body},
               span: d_s}};
    p.bump();
    alt the_ctor {
      some((ct_d, ct_b, ct_s)) {
        (class_name,
         item_class(ty_params, ifaces, ms, {
             node: {id: ctor_id,
                    self_id: p.get_id(),
                    dec: ct_d,
                    body: ct_b},
                     span: ct_s}, actual_dtor, rp),
        none)
      }
       /*
         Is it strange for the parser to check this?
       */
       none {
         p.fatal("class with no ctor");
       }
    }
}

fn parse_single_class_item(p: parser, vis: visibility)
    -> @class_member {
   if eat_keyword(p, "let") {
      let a_var = parse_instance_var(p, vis);
      expect(p, token::SEMI);
      ret a_var;
   }
   else {
       let m = parse_method(p, vis);
       ret @{node: class_method(m), span: m.span};
   }
}

/*
  So that we can distinguish a class ctor or dtor
  from other class members
 */
enum class_contents { ctor_decl(fn_decl, blk, codemap::span),
                      dtor_decl(blk, codemap::span),
                      members([@class_member]) }

fn parse_ctor(p: parser, result_ty: ast::ty_) -> class_contents {
  // Can ctors/dtors have attrs? FIXME
  let lo = p.last_span.lo;
  let (decl_, _) = parse_fn_decl(p, impure_fn, parse_arg);
  let decl = {output: @{id: p.get_id(),
                        node: result_ty, span: decl_.output.span}
              with decl_};
  let body = parse_block(p);
  ctor_decl(decl, body, mk_sp(lo, p.last_span.hi))
}

fn parse_dtor(p: parser) -> class_contents {
  // Can ctors/dtors have attrs? FIXME
  let lo = p.last_span.lo;
  let body = parse_block(p);
  dtor_decl(body, mk_sp(lo, p.last_span.hi))
}

fn parse_class_item(p:parser, class_name_with_tps: @path)
    -> class_contents {
    if eat_keyword(p, "new") {
       // result type is always the type of the class
       ret parse_ctor(p, ty_path(class_name_with_tps,
                              p.get_id()));
    }
    else if eat_keyword(p, "drop") {
      ret parse_dtor(p);
    }
    else if eat_keyword(p, "priv") {
            expect(p, token::LBRACE);
            let mut results = [];
            while p.token != token::RBRACE {
                    results += [parse_single_class_item(p, private)];
            }
            p.bump();
            ret members(results);
    }
    else {
        // Probably need to parse attrs
        ret members([parse_single_class_item(p, public)]);
    }
}

fn parse_visibility(p: parser, def: visibility) -> visibility {
    if eat_keyword(p, "pub") { public }
    else if eat_keyword(p, "priv") { private }
    else { def }
}

fn parse_mod_items(p: parser, term: token::token,
                   +first_item_attrs: [attribute]) -> _mod {
    // Shouldn't be any view items since we've already parsed an item attr
    let {attrs_remaining, view_items} =
        parse_view(p, first_item_attrs, false);
    let mut items: [@item] = [];
    let mut first = true;
    while p.token != term {
        let mut attrs = parse_outer_attributes(p);
        if first { attrs = attrs_remaining + attrs; first = false; }
        #debug["parse_mod_items: parse_item(attrs=%?)", attrs];
        let vis = parse_visibility(p, private);
        alt parse_item(p, attrs, vis) {
          some(i) { items += [i]; }
          _ {
            p.fatal("expected item but found '" +
                    token_to_str(p.reader, p.token) + "'");
          }
        }
        #debug["parse_mod_items: attrs=%?", attrs];
    }

    if first && attrs_remaining.len() > 0u {
        // We parsed attributes for the first item but didn't find the item
        p.fatal("expected item");
    }

    ret {view_items: view_items, items: items};
}

fn parse_item_const(p: parser) -> item_info {
    let id = parse_value_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    expect(p, token::EQ);
    let e = parse_expr(p);
    expect(p, token::SEMI);
    (id, item_const(ty, e), none)
}

fn parse_item_mod(p: parser) -> item_info {
    let id = parse_ident(p);
    expect(p, token::LBRACE);
    let inner_attrs = parse_inner_attrs_and_next(p);
    let m = parse_mod_items(p, token::RBRACE, inner_attrs.next);
    expect(p, token::RBRACE);
    (id, item_mod(m), some(inner_attrs.inner))
}

fn parse_item_native_fn(p: parser, +attrs: [attribute],
                        purity: purity) -> @native_item {
    let lo = p.last_span.lo;
    let t = parse_fn_header(p);
    let (decl, _) = parse_fn_decl(p, purity, parse_arg);
    let mut hi = p.span.hi;
    expect(p, token::SEMI);
    ret @{ident: t.ident,
          attrs: attrs,
          node: native_item_fn(decl, t.tps),
          id: p.get_id(),
          span: mk_sp(lo, hi)};
}

fn parse_fn_purity(p: parser) -> purity {
    if eat_keyword(p, "fn") { impure_fn }
    else if eat_keyword(p, "pure") { expect_keyword(p, "fn"); pure_fn }
    else if eat_keyword(p, "unsafe") {
        expect_keyword(p, "fn");
        unsafe_fn
    }
    else { unexpected(p); }
}

fn parse_native_item(p: parser, +attrs: [attribute]) ->
   @native_item {
    parse_item_native_fn(p, attrs, parse_fn_purity(p))
}

fn parse_native_mod_items(p: parser, +first_item_attrs: [attribute]) ->
   native_mod {
    // Shouldn't be any view items since we've already parsed an item attr
    let {attrs_remaining, view_items} =
        parse_view(p, first_item_attrs, false);
    let mut items: [@native_item] = [];
    let mut initial_attrs = attrs_remaining;
    while p.token != token::RBRACE {
        let attrs = initial_attrs + parse_outer_attributes(p);
        initial_attrs = [];
        items += [parse_native_item(p, attrs)];
    }
    ret {view_items: view_items,
         items: items};
}

fn parse_item_native_mod(p: parser) -> item_info {
    expect_keyword(p, "mod");
    let id = parse_ident(p);
    expect(p, token::LBRACE);
    let more_attrs = parse_inner_attrs_and_next(p);
    let m = parse_native_mod_items(p, more_attrs.next);
    expect(p, token::RBRACE);
    (id, item_native_mod(m), some(more_attrs.inner))
}

fn parse_type_decl(p: parser) -> {lo: uint, ident: ident} {
    let lo = p.last_span.lo;
    let id = parse_ident(p);
    ret {lo: lo, ident: id};
}

fn parse_item_type(p: parser) -> item_info {
    let t = parse_type_decl(p);
    let rp = parse_region_param(p);
    let tps = parse_ty_params(p);
    expect(p, token::EQ);
    let ty = parse_ty(p, false);
    expect(p, token::SEMI);
    (t.ident, item_ty(ty, tps, rp), none)
}

fn parse_region_param(p: parser) -> region_param {
    if eat(p, token::BINOP(token::SLASH)) {
        expect(p, token::BINOP(token::AND));
        rp_self
    } else {
        rp_none
    }
}

fn parse_item_enum(p: parser, default_vis: visibility) -> item_info {
    let id = parse_ident(p);
    let rp = parse_region_param(p);
    let ty_params = parse_ty_params(p);
    let mut variants: [variant] = [];
    // Newtype syntax
    if p.token == token::EQ {
        check_restricted_keywords_(p, id);
        p.bump();
        let ty = parse_ty(p, false);
        expect(p, token::SEMI);
        let variant =
            spanned(ty.span.lo, ty.span.hi,
                    {name: id,
                     attrs: [],
                     args: [{ty: ty, id: p.get_id()}],
                     id: p.get_id(),
                     disr_expr: none,
                     vis: public});
        ret (id, item_enum([variant], ty_params, rp), none);
    }
    expect(p, token::LBRACE);

    let mut all_nullary = true, have_disr = false;

    while p.token != token::RBRACE {
        let variant_attrs = parse_outer_attributes(p);
        let vlo = p.span.lo;
        let vis = parse_visibility(p, default_vis);
        let ident = parse_value_ident(p);
        let mut args = [], disr_expr = none;
        if p.token == token::LPAREN {
            all_nullary = false;
            let arg_tys = parse_seq(token::LPAREN, token::RPAREN,
                                    seq_sep(token::COMMA),
                                    {|p| parse_ty(p, false)}, p);
            for arg_tys.node.each {|ty|
                args += [{ty: ty, id: p.get_id()}];
            }
        } else if eat(p, token::EQ) {
            have_disr = true;
            disr_expr = some(parse_expr(p));
        }

        let vr = {name: ident, attrs: variant_attrs,
                  args: args, id: p.get_id(),
                  disr_expr: disr_expr, vis: vis};
        variants += [spanned(vlo, p.last_span.hi, vr)];

        if !eat(p, token::COMMA) { break; }
    }
    expect(p, token::RBRACE);
    if (have_disr && !all_nullary) {
        p.fatal("discriminator values can only be used with a c-like enum");
    }
    (id, item_enum(variants, ty_params, rp), none)
}

fn parse_fn_ty_proto(p: parser) -> proto {
    alt p.token {
      token::AT {
        p.bump();
        proto_box
      }
      token::TILDE {
        p.bump();
        proto_uniq
      }
      token::BINOP(token::AND) {
        p.bump();
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

fn parse_item(p: parser, +attrs: [attribute], vis: visibility)
    -> option<@item> {
    let lo = p.span.lo;
    let (ident, item_, extra_attrs) = if eat_keyword(p, "const") {
        parse_item_const(p)
    } else if is_keyword(p, "fn") && !fn_expr_lookahead(p.look_ahead(1u)) {
        p.bump();
        parse_item_fn(p, impure_fn)
    } else if eat_keyword(p, "pure") {
        expect_keyword(p, "fn");
        parse_item_fn(p, pure_fn)
    } else if is_keyword(p, "unsafe") && p.look_ahead(1u) != token::LBRACE {
        p.bump();
        expect_keyword(p, "fn");
        parse_item_fn(p, unsafe_fn)
    } else if eat_keyword(p, "crust") {
        expect_keyword(p, "fn");
        parse_item_fn(p, crust_fn)
    } else if eat_keyword(p, "mod") {
        parse_item_mod(p)
    } else if eat_keyword(p, "native") {
        parse_item_native_mod(p)
    } else if eat_keyword(p, "type") {
        parse_item_type(p)
    } else if eat_keyword(p, "enum") {
        parse_item_enum(p, vis)
    } else if eat_keyword(p, "iface") {
        parse_item_iface(p)
    } else if eat_keyword(p, "impl") {
        parse_item_impl(p)
    } else if eat_keyword(p, "resource") {
        parse_item_res(p)
    } else if eat_keyword(p, "class") {
        parse_item_class(p)
    } else { ret none; };
    some(mk_item(p, lo, p.last_span.hi, ident, item_, vis,
                 alt extra_attrs {
                     some(as) { attrs + as }
                     none { attrs }
                 }))
}

fn parse_use(p: parser) -> view_item_ {
    let ident = parse_ident(p);
    let metadata = parse_optional_meta(p);
    ret view_item_use(ident, metadata, p.get_id());
}

fn parse_view_path(p: parser) -> @view_path {
    let lo = p.span.lo;
    let first_ident = parse_ident(p);
    let mut path = [first_ident];
    #debug("parsed view_path: %s", first_ident);
    alt p.token {
      token::EQ {
        // x = foo::bar
        p.bump();
        path = [parse_ident(p)];
        while p.token == token::MOD_SEP {
            p.bump();
            let id = parse_ident(p);
            path += [id];
        }
        let path = @{span: mk_sp(lo, p.span.hi), global: false,
                     idents: path, rp: none, types: []};
        ret @spanned(lo, p.span.hi,
                     view_path_simple(first_ident, path, p.get_id()));
      }

      token::MOD_SEP {
        // foo::bar or foo::{a,b,c} or foo::*
        while p.token == token::MOD_SEP {
            p.bump();

            alt p.token {

              token::IDENT(i, _) {
                p.bump();
                path += [p.get_str(i)];
              }

              // foo::bar::{a,b,c}
              token::LBRACE {
                let idents =
                    parse_seq(token::LBRACE, token::RBRACE,
                              seq_sep(token::COMMA),
                              parse_path_list_ident, p).node;
                let path = @{span: mk_sp(lo, p.span.hi),
                             global: false, idents: path,
                             rp: none, types: []};
                ret @spanned(lo, p.span.hi,
                             view_path_list(path, idents, p.get_id()));
              }

              // foo::bar::*
              token::BINOP(token::STAR) {
                p.bump();
                let path = @{span: mk_sp(lo, p.span.hi),
                             global: false, idents: path,
                             rp: none, types: []};
                ret @spanned(lo, p.span.hi,
                             view_path_glob(path, p.get_id()));
              }

              _ { break; }
            }
        }
      }
      _ { }
    }
    let last = path[vec::len(path) - 1u];
    let path = @{span: mk_sp(lo, p.span.hi), global: false,
                 idents: path, rp: none, types: []};
    ret @spanned(lo, p.span.hi,
                 view_path_simple(last, path, p.get_id()));
}

fn parse_view_paths(p: parser) -> [@view_path] {
    let mut vp = [parse_view_path(p)];
    while p.token == token::COMMA {
        p.bump();
        vp += [parse_view_path(p)];
    }
    ret vp;
}

fn is_view_item(p: parser) -> bool {
    let tok = if !is_keyword(p, "pub") && !is_keyword(p, "priv") { p.token }
              else { p.look_ahead(1u) };
    token_is_keyword(p, "use", tok) || token_is_keyword(p, "import", tok) ||
        token_is_keyword(p, "export", tok)
}

fn parse_view_item(p: parser, +attrs: [attribute]) -> @view_item {
    let lo = p.span.lo, vis = parse_visibility(p, private);
    let node = if eat_keyword(p, "use") {
        parse_use(p)
    } else if eat_keyword(p, "import") {
        view_item_import(parse_view_paths(p))
    } else if eat_keyword(p, "export") {
        view_item_export(parse_view_paths(p))
    } else { fail; };
    expect(p, token::SEMI);
    @{node: node, attrs: attrs,
      vis: vis, span: mk_sp(lo, p.last_span.hi)}
}

fn parse_view(p: parser, +first_item_attrs: [attribute],
              only_imports: bool) -> {attrs_remaining: [attribute],
                                      view_items: [@view_item]} {
    let mut attrs = first_item_attrs + parse_outer_attributes(p);
    let mut items = [];
    while if only_imports { is_keyword(p, "import") }
          else { is_view_item(p) } {
        items += [parse_view_item(p, attrs)];
        attrs = parse_outer_attributes(p);
    }
    {attrs_remaining: attrs, view_items: items}
}

// Parses a source module as a crate
fn parse_crate_mod(p: parser, _cfg: crate_cfg) -> @crate {
    let lo = p.span.lo;
    let crate_attrs = parse_inner_attrs_and_next(p);
    let first_item_outer_attrs = crate_attrs.next;
    let m = parse_mod_items(p, token::EOF, first_item_outer_attrs);
    ret @spanned(lo, p.span.lo,
                 {directives: [],
                  module: m,
                  attrs: crate_attrs.inner,
                  config: p.cfg});
}

fn parse_str(p: parser) -> str {
    alt p.token {
      token::LIT_STR(s) { p.bump(); p.get_str(s) }
      _ {
        p.fatal("expected string literal")
      }
    }
}

// Logic for parsing crate files (.rc)
//
// Each crate file is a sequence of directives.
//
// Each directive imperatively extends its environment with 0 or more items.
fn parse_crate_directive(p: parser, first_outer_attr: [attribute]) ->
   crate_directive {

    // Collect the next attributes
    let outer_attrs = first_outer_attr + parse_outer_attributes(p);
    // In a crate file outer attributes are only going to apply to mods
    let expect_mod = vec::len(outer_attrs) > 0u;

    let lo = p.span.lo;
    if expect_mod || is_keyword(p, "mod") {
        expect_keyword(p, "mod");
        let id = parse_ident(p);
        alt p.token {
          // mod x = "foo.rs";
          token::SEMI {
            let mut hi = p.span.hi;
            p.bump();
            ret spanned(lo, hi, cdir_src_mod(id, outer_attrs));
          }
          // mod x = "foo_dir" { ...directives... }
          token::LBRACE {
            p.bump();
            let inner_attrs = parse_inner_attrs_and_next(p);
            let mod_attrs = outer_attrs + inner_attrs.inner;
            let next_outer_attr = inner_attrs.next;
            let cdirs =
                parse_crate_directives(p, token::RBRACE, next_outer_attr);
            let mut hi = p.span.hi;
            expect(p, token::RBRACE);
            ret spanned(lo, hi,
                        cdir_dir_mod(id, cdirs, mod_attrs));
          }
          _ { unexpected(p); }
        }
    } else if is_view_item(p) {
        let vi = parse_view_item(p, outer_attrs);
        ret spanned(lo, vi.span.hi, cdir_view_item(vi));
    } else { ret p.fatal("expected crate directive"); }
}

fn parse_crate_directives(p: parser, term: token::token,
                          first_outer_attr: [attribute]) ->
   [@crate_directive] {

    // This is pretty ugly. If we have an outer attribute then we can't accept
    // seeing the terminator next, so if we do see it then fail the same way
    // parse_crate_directive would
    if vec::len(first_outer_attr) > 0u && p.token == term {
        expect_keyword(p, "mod");
    }

    let mut cdirs: [@crate_directive] = [];
    let mut first_outer_attr = first_outer_attr;
    while p.token != term {
        let cdir = @parse_crate_directive(p, first_outer_attr);
        cdirs += [cdir];
        first_outer_attr = [];
    }
    ret cdirs;
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

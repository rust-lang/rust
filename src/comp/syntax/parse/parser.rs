import core::{vec, str, option, either, result};
import std::{io, fs};
import option::{some, none};
import either::{left, right};
import std::map::{hashmap, new_str_hash};
import token::can_begin_expr;
import codemap::span;
import util::interner;
import ast::{node_id, spanned};
import ast_util::{mk_sp, ident_to_path};
import front::attr;
import lexer::reader;
import driver::diagnostic;

enum restriction {
    UNRESTRICTED,
    RESTRICT_STMT_EXPR,
    RESTRICT_NO_CALL_EXPRS,
    RESTRICT_NO_BAR_OP,
}

enum file_type { CRATE_FILE, SOURCE_FILE, }

type parse_sess = @{
    cm: codemap::codemap,
    mutable next_id: node_id,
    span_diagnostic: diagnostic::span_handler,
    // these two must be kept up to date
    mutable chpos: uint,
    mutable byte_pos: uint
};

fn next_node_id(sess: parse_sess) -> node_id {
    let rv = sess.next_id;
    sess.next_id += 1;
    // ID 0 is reserved for the crate and doesn't actually exist in the AST
    assert rv != 0;
    ret rv;
}

type parser = @{
    sess: parse_sess,
    cfg: ast::crate_cfg,
    file_type: file_type,
    mutable token: token::token,
    mutable span: span,
    mutable last_span: span,
    mutable buffer: [{tok: token::token, span: span}],
    mutable restriction: restriction,
    reader: reader,
    precs: @[op_spec],
    bad_expr_words: hashmap<str, ()>
};

impl parser for parser {
    fn bump() {
        self.last_span = self.span;
        if vec::len(self.buffer) == 0u {
            let next = lexer::next_token(self.reader);
            self.token = next.tok;
            self.span = ast_util::mk_sp(next.chpos, self.reader.chpos);
        } else {
            let next = vec::pop(self.buffer);
            self.token = next.tok;
            self.span = next.span;
        }
    }
    fn swap(next: token::token, lo: uint, hi: uint) {
        self.token = next;
        self.span = ast_util::mk_sp(lo, hi);
    }
    fn look_ahead(distance: uint) -> token::token {
        while vec::len(self.buffer) < distance {
            let next = lexer::next_token(self.reader);
            let sp = ast_util::mk_sp(next.chpos, self.reader.chpos);
            self.buffer = [{tok: next.tok, span: sp}] + self.buffer;
        }
        ret self.buffer[distance - 1u].tok;
    }
    fn fatal(m: str) -> ! {
        self.sess.span_diagnostic.span_fatal(self.span, m)
    }
    fn span_fatal(sp: span, m: str) -> ! {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }
    fn warn(m: str) {
        self.sess.span_diagnostic.span_warn(self.span, m)
    }
    fn get_str(i: token::str_num) -> str {
        interner::get(*self.reader.interner, i)
    }
    fn get_id() -> node_id { next_node_id(self.sess) }
}

fn new_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg, path: str,
                        ftype: file_type) ->
   parser {
    let src = alt io::read_whole_file_str(path) {
      result::ok(src) {
        // FIXME: This copy is unfortunate
        @src
      }
      result::err(e) {
        sess.span_diagnostic.handler().fatal(e)
      }
    };
    let filemap = codemap::new_filemap(path, src,
                                       sess.chpos, sess.byte_pos);
    sess.cm.files += [filemap];
    let itr = @interner::mk(str::hash, str::eq);
    let rdr = lexer::new_reader(sess.cm, sess.span_diagnostic, filemap, itr);
    ret new_parser(sess, cfg, rdr, ftype);
}

fn new_parser_from_source_str(sess: parse_sess, cfg: ast::crate_cfg,
                              name: str, ss: codemap::file_substr,
                              source: @str) -> parser {
    let ftype = SOURCE_FILE;
    let filemap = codemap::new_filemap_w_substr
        (name, ss, source, sess.chpos, sess.byte_pos);
    sess.cm.files += [filemap];
    let itr = @interner::mk(str::hash, str::eq);
    let rdr = lexer::new_reader(sess.cm, sess.span_diagnostic,
                                filemap, itr);
    ret new_parser(sess, cfg, rdr, ftype);
}

fn new_parser(sess: parse_sess, cfg: ast::crate_cfg, rdr: reader,
              ftype: file_type) -> parser {
    let tok0 = lexer::next_token(rdr);
    let span0 = ast_util::mk_sp(tok0.chpos, rdr.chpos);
    @{sess: sess,
      cfg: cfg,
      file_type: ftype,
      mutable token: tok0.tok,
      mutable span: span0,
      mutable last_span: span0,
      mutable buffer: [],
      mutable restriction: UNRESTRICTED,
      reader: rdr,
      precs: prec_table(),
      bad_expr_words: bad_expr_word_table()}
}

// These are the words that shouldn't be allowed as value identifiers,
// because, if used at the start of a line, they will cause the line to be
// interpreted as a specific kind of statement, which would be confusing.
fn bad_expr_word_table() -> hashmap<str, ()> {
    let words = new_str_hash();
    for word in ["mod", "if", "else", "while", "do", "alt", "for", "break",
                 "cont", "ret", "be", "fail", "type", "resource", "check",
                 "assert", "claim", "native", "fn", "pure",
                 "unsafe", "import", "export", "let", "const",
                 "log", "copy", "sendfn", "impl", "iface", "enum",
                 "m32", "m64", "m128", "f80", "f16", "f128",
                 "class", "trait"] {
        words.insert(word, ());
    }
    words
}

fn unexpected(p: parser, t: token::token) -> ! {
    let s: str = "unexpected token: '" + token::to_str(p.reader, t) +
        "'";
    p.fatal(s);
}

fn expect(p: parser, t: token::token) {
    if p.token == t {
        p.bump();
    } else {
        let s: str = "expecting '";
        s += token::to_str(p.reader, t);
        s += "' but found '";
        s += token::to_str(p.reader, p.token);
        p.fatal(s + "'");
    }
}

fn expect_gt(p: parser) {
    if p.token == token::GT {
        p.bump();
    } else if p.token == token::BINOP(token::LSR) {
        p.swap(token::GT, p.span.lo + 1u, p.span.hi);
    } else if p.token == token::BINOP(token::ASR) {
        p.swap(token::BINOP(token::LSR), p.span.lo + 1u, p.span.hi);
    } else {
        let s: str = "expecting ";
        s += token::to_str(p.reader, token::GT);
        s += ", found ";
        s += token::to_str(p.reader, p.token);
        p.fatal(s);
    }
}

fn spanned<T: copy>(lo: uint, hi: uint, node: T) -> spanned<T> {
    ret {node: node, span: ast_util::mk_sp(lo, hi)};
}

fn parse_ident(p: parser) -> ast::ident {
    alt p.token {
      token::IDENT(i, _) { p.bump(); ret p.get_str(i); }
      _ { p.fatal("expecting ident, found "
                  + token::to_str(p.reader, p.token)); }
    }
}

fn parse_import_ident(p: parser) -> ast::import_ident {
    let lo = p.span.lo;
    let ident = parse_ident(p);
    let hi = p.span.hi;
    ret spanned(lo, hi, {name: ident, id: p.get_id()});
}

fn parse_value_ident(p: parser) -> ast::ident {
    check_bad_word(p);
    ret parse_ident(p);
}

fn eat(p: parser, tok: token::token) -> bool {
    ret if p.token == tok { p.bump(); true } else { false };
}

fn is_word(p: parser, word: str) -> bool {
    ret alt p.token {
          token::IDENT(sid, false) { str::eq(word, p.get_str(sid)) }
          _ { false }
        };
}

fn eat_word(p: parser, word: str) -> bool {
    alt p.token {
      token::IDENT(sid, false) {
        if str::eq(word, p.get_str(sid)) {
            p.bump();
            ret true;
        } else { ret false; }
      }
      _ { ret false; }
    }
}

fn expect_word(p: parser, word: str) {
    if !eat_word(p, word) {
        p.fatal("expecting " + word + ", found " +
                    token::to_str(p.reader, p.token));
    }
}

fn check_bad_word(p: parser) {
    alt p.token {
      token::IDENT(sid, false) {
        let w = p.get_str(sid);
        if p.bad_expr_words.contains_key(w) {
            p.fatal("found " + w + " in expression position");
        }
      }
      _ { }
    }
}

fn parse_ty_fn(p: parser) -> ast::fn_decl {
    fn parse_fn_input_ty(p: parser) -> ast::arg {
        let mode = parse_arg_mode(p);
        let name = if is_plain_ident(p) && p.look_ahead(1u) == token::COLON {
            let name = parse_value_ident(p);
            p.bump();
            name
        } else { "" };
        ret {mode: mode, ty: parse_ty(p, false), ident: name, id: p.get_id()};
    }
    let inputs =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_fn_input_ty, p);
    // FIXME: there's no syntax for this right now anyway
    //  auto constrs = parse_constrs(~[], p);
    let constrs: [@ast::constr] = [];
    let (ret_style, ret_ty) = parse_ret_ty(p);
    ret {inputs: inputs.node, output: ret_ty,
                           purity: ast::impure_fn, cf: ret_style,
                           constraints: constrs};
}

fn parse_ty_methods(p: parser) -> [ast::ty_method] {
    parse_seq(token::LBRACE, token::RBRACE, seq_sep_none(), {|p|
        let attrs = parse_outer_attributes(p);
        let flo = p.span.lo;
        expect_word(p, "fn");
        let ident = parse_method_name(p);
        let tps = parse_ty_params(p);
        let d = parse_ty_fn(p), fhi = p.last_span.hi;
        expect(p, token::SEMI);
            {ident: ident, attrs: attrs, decl: d, tps: tps,
                    span: ast_util::mk_sp(flo, fhi)}}, p).node
}

fn parse_mt(p: parser) -> ast::mt {
    let mut = parse_mutability(p);
    let t = parse_ty(p, false);
    ret {ty: t, mut: mut};
}

fn parse_ty_field(p: parser) -> ast::ty_field {
    let lo = p.span.lo;
    let mut = parse_mutability(p);
    let id = parse_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    ret spanned(lo, ty.span.hi, {ident: id, mt: {ty: ty, mut: mut}});
}

// if i is the jth ident in args, return j
// otherwise, fail
fn ident_index(p: parser, args: [ast::arg], i: ast::ident) -> uint {
    let j = 0u;
    for a: ast::arg in args { if a.ident == i { ret j; } j += 1u; }
    p.fatal("Unbound variable " + i + " in constraint arg");
}

fn parse_type_constr_arg(p: parser) -> @ast::ty_constr_arg {
    let sp = p.span;
    let carg = ast::carg_base;
    expect(p, token::BINOP(token::STAR));
    if p.token == token::DOT {
        // "*..." notation for record fields
        p.bump();
        let pth = parse_path(p);
        carg = ast::carg_ident(pth);
    }
    // No literals yet, I guess?
    ret @{node: carg, span: sp};
}

fn parse_constr_arg(args: [ast::arg], p: parser) -> @ast::constr_arg {
    let sp = p.span;
    let carg = ast::carg_base;
    if p.token == token::BINOP(token::STAR) {
        p.bump();
    } else {
        let i: ast::ident = parse_value_ident(p);
        carg = ast::carg_ident(ident_index(p, args, i));
    }
    ret @{node: carg, span: sp};
}

fn parse_ty_constr(fn_args: [ast::arg], p: parser) -> @ast::constr {
    let lo = p.span.lo;
    let path = parse_path(p);
    let args: {node: [@ast::constr_arg], span: span} =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  {|p| parse_constr_arg(fn_args, p)}, p);
    ret @spanned(lo, args.span.hi,
                 {path: path, args: args.node, id: p.get_id()});
}

fn parse_constr_in_type(p: parser) -> @ast::ty_constr {
    let lo = p.span.lo;
    let path = parse_path(p);
    let args: [@ast::ty_constr_arg] =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_type_constr_arg, p).node;
    let hi = p.span.lo;
    let tc: ast::ty_constr_ = {path: path, args: args, id: p.get_id()};
    ret @spanned(lo, hi, tc);
}


fn parse_constrs<T: copy>(pser: fn(parser) -> @ast::constr_general<T>,
                         p: parser) ->
   [@ast::constr_general<T>] {
    let constrs: [@ast::constr_general<T>] = [];
    while true {
        let constr = pser(p);
        constrs += [constr];
        if p.token == token::COMMA { p.bump(); } else { break; }
    }
    constrs
}

fn parse_type_constraints(p: parser) -> [@ast::ty_constr] {
    ret parse_constrs(parse_constr_in_type, p);
}

fn parse_ty_postfix(orig_t: ast::ty_, p: parser, colons_before_params: bool,
                    lo: uint) -> @ast::ty {
    if colons_before_params && p.token == token::MOD_SEP {
        p.bump();
        expect(p, token::LT);
    } else if !colons_before_params && p.token == token::LT {
        p.bump();
    } else { ret @spanned(lo, p.last_span.hi, orig_t); }

    // If we're here, we have explicit type parameter instantiation.
    let seq = parse_seq_to_gt(some(token::COMMA), {|p| parse_ty(p, false)},
                              p);

    alt orig_t {
      ast::ty_path(pth, ann) {
        ret @spanned(lo, p.last_span.hi,
                     ast::ty_path(@spanned(lo, p.last_span.hi,
                                           {global: pth.node.global,
                                            idents: pth.node.idents,
                                            types: seq}), ann));
      }
      _ { p.fatal("type parameter instantiation only allowed for paths"); }
    }
}

fn parse_ret_ty(p: parser) -> (ast::ret_style, @ast::ty) {
    ret if eat(p, token::RARROW) {
        let lo = p.span.lo;
        if eat(p, token::NOT) {
            (ast::noreturn, @spanned(lo, p.last_span.hi, ast::ty_bot))
        } else { (ast::return_val, parse_ty(p, false)) }
    } else {
        let pos = p.span.lo;
        (ast::return_val, @spanned(pos, pos, ast::ty_nil))
    }
}

fn parse_ty(p: parser, colons_before_params: bool) -> @ast::ty {
    let lo = p.span.lo;
    let t: ast::ty_;
    // FIXME: do something with this

    if eat_word(p, "bool") {
        t = ast::ty_bool;
    } else if eat_word(p, "int") {
        t = ast::ty_int(ast::ty_i);
    } else if eat_word(p, "uint") {
        t = ast::ty_uint(ast::ty_u);
    } else if eat_word(p, "float") {
        t = ast::ty_float(ast::ty_f);
    } else if eat_word(p, "str") {
        t = ast::ty_str;
    } else if eat_word(p, "char") {
        t = ast::ty_int(ast::ty_char);
    } else if eat_word(p, "i8") {
        t = ast::ty_int(ast::ty_i8);
    } else if eat_word(p, "i16") {
        t = ast::ty_int(ast::ty_i16);
    } else if eat_word(p, "i32") {
        t = ast::ty_int(ast::ty_i32);
    } else if eat_word(p, "i64") {
        t = ast::ty_int(ast::ty_i64);
    } else if eat_word(p, "u8") {
        t = ast::ty_uint(ast::ty_u8);
    } else if eat_word(p, "u16") {
        t = ast::ty_uint(ast::ty_u16);
    } else if eat_word(p, "u32") {
        t = ast::ty_uint(ast::ty_u32);
    } else if eat_word(p, "u64") {
        t = ast::ty_uint(ast::ty_u64);
    } else if eat_word(p, "f32") {
        t = ast::ty_float(ast::ty_f32);
    } else if eat_word(p, "f64") {
        t = ast::ty_float(ast::ty_f64);
    } else if p.token == token::LPAREN {
        p.bump();
        if p.token == token::RPAREN {
            p.bump();
            t = ast::ty_nil;
        } else {
            let ts = [parse_ty(p, false)];
            while p.token == token::COMMA {
                p.bump();
                ts += [parse_ty(p, false)];
            }
            if vec::len(ts) == 1u {
                t = ts[0].node;
            } else { t = ast::ty_tup(ts); }
            expect(p, token::RPAREN);
        }
    } else if p.token == token::AT {
        p.bump();
        t = ast::ty_box(parse_mt(p));
    } else if p.token == token::TILDE {
        p.bump();
        t = ast::ty_uniq(parse_mt(p));
    } else if p.token == token::BINOP(token::STAR) {
        p.bump();
        t = ast::ty_ptr(parse_mt(p));
    } else if p.token == token::LBRACE {
        let elems =
            parse_seq(token::LBRACE, token::RBRACE, seq_sep_opt(token::COMMA),
                      parse_ty_field, p);
        if vec::len(elems.node) == 0u { unexpected(p, token::RBRACE); }
        let hi = elems.span.hi;
        t = ast::ty_rec(elems.node);
        if p.token == token::COLON {
            p.bump();
            t = ast::ty_constr(@spanned(lo, hi, t),
                               parse_type_constraints(p));
        }
    } else if p.token == token::LBRACKET {
        expect(p, token::LBRACKET);
        t = ast::ty_vec(parse_mt(p));
        expect(p, token::RBRACKET);
    } else if eat_word(p, "fn") {
        let proto = parse_fn_ty_proto(p);
        alt proto {
          ast::proto_bare { p.warn("fn is deprecated, use native fn"); }
          _ { /* fallthrough */ }
        }
        t = ast::ty_fn(proto, parse_ty_fn(p));
    } else if eat_word(p, "native") {
        expect_word(p, "fn");
        t = ast::ty_fn(ast::proto_bare, parse_ty_fn(p));
    } else if p.token == token::MOD_SEP || is_ident(p.token) {
        let path = parse_path(p);
        t = ast::ty_path(path, p.get_id());
    } else { p.fatal("expecting type"); }
    ret parse_ty_postfix(t, p, colons_before_params, lo);
}

fn parse_arg_mode(p: parser) -> ast::mode {
    if eat(p, token::BINOP(token::AND)) { ast::by_mut_ref }
    else if eat(p, token::BINOP(token::MINUS)) { ast::by_move }
    else if eat(p, token::ANDAND) { ast::by_ref }
    else if eat(p, token::BINOP(token::PLUS)) {
        if eat(p, token::BINOP(token::PLUS)) { ast::by_val }
        else { ast::by_copy }
    }
    else { ast::mode_infer }
}

fn parse_arg(p: parser) -> ast::arg {
    let m = parse_arg_mode(p);
    let i = parse_value_ident(p);
    expect(p, token::COLON);
    let t = parse_ty(p, false);
    ret {mode: m, ty: t, ident: i, id: p.get_id()};
}

fn parse_fn_block_arg(p: parser) -> ast::arg {
    let m = parse_arg_mode(p);
    let i = parse_value_ident(p);
    let t = if eat(p, token::COLON) {
                parse_ty(p, false)
            } else {
                @spanned(p.span.lo, p.span.hi, ast::ty_infer)
            };
    ret {mode: m, ty: t, ident: i, id: p.get_id()};
}

fn parse_seq_to_before_gt<T: copy>(sep: option<token::token>,
                                  f: fn(parser) -> T,
                                  p: parser) -> [T] {
    let first = true;
    let v = [];
    while p.token != token::GT && p.token != token::BINOP(token::LSR) &&
              p.token != token::BINOP(token::ASR) {
        alt sep {
          some(t) { if first { first = false; } else { expect(p, t); } }
          _ { }
        }
        v += [f(p)];
    }

    ret v;
}

fn parse_seq_to_gt<T: copy>(sep: option<token::token>,
                           f: fn(parser) -> T, p: parser) -> [T] {
    let v = parse_seq_to_before_gt(sep, f, p);
    expect_gt(p);

    ret v;
}

fn parse_seq_lt_gt<T: copy>(sep: option<token::token>,
                           f: fn(parser) -> T,
                           p: parser) -> spanned<[T]> {
    let lo = p.span.lo;
    expect(p, token::LT);
    let result = parse_seq_to_before_gt::<T>(sep, f, p);
    let hi = p.span.hi;
    expect_gt(p);
    ret spanned(lo, hi, result);
}

fn parse_seq_to_end<T: copy>(ket: token::token, sep: seq_sep,
                            f: fn(parser) -> T, p: parser) -> [T] {
    let val = parse_seq_to_before_end(ket, sep, f, p);
    p.bump();
    ret val;
}

type seq_sep = {
    sep: option<token::token>,
    trailing_opt: bool   // is trailing separator optional?
};

fn seq_sep(t: token::token) -> seq_sep {
    ret {sep: option::some(t), trailing_opt: false};
}
fn seq_sep_opt(t: token::token) -> seq_sep {
    ret {sep: option::some(t), trailing_opt: true};
}
fn seq_sep_none() -> seq_sep {
    ret {sep: option::none, trailing_opt: false};
}

fn parse_seq_to_before_end<T: copy>(ket: token::token,
                                   sep: seq_sep,
                                   f: fn(parser) -> T, p: parser) -> [T] {
    let first: bool = true;
    let v: [T] = [];
    while p.token != ket {
        alt sep.sep {
          some(t) { if first { first = false; } else { expect(p, t); } }
          _ { }
        }
        if sep.trailing_opt && p.token == ket { break; }
        v += [f(p)];
    }
    ret v;
}


fn parse_seq<T: copy>(bra: token::token, ket: token::token,
                     sep: seq_sep, f: fn(parser) -> T,
                     p: parser) -> spanned<[T]> {
    let lo = p.span.lo;
    expect(p, bra);
    let result = parse_seq_to_before_end::<T>(ket, sep, f, p);
    let hi = p.span.hi;
    p.bump();
    ret spanned(lo, hi, result);
}

fn have_dollar(p: parser) -> option::t<ast::mac_> {
    alt p.token {
      token::DOLLAR_NUM(num) {
        p.bump();
        some(ast::mac_var(num))
      }
      token::DOLLAR_LPAREN {
        let lo = p.span.lo;
        p.bump();
        let e = parse_expr(p);
        expect(p, token::RPAREN);
        let hi = p.last_span.hi;
        some(ast::mac_aq(ast_util::mk_sp(lo,hi), e))
      }
      _ {none}
    }
}

fn lit_from_token(p: parser, tok: token::token) -> ast::lit_ {
    alt tok {
      token::LIT_INT(i, it) { ast::lit_int(i, it) }
      token::LIT_UINT(u, ut) { ast::lit_uint(u, ut) }
      token::LIT_FLOAT(s, ft) { ast::lit_float(p.get_str(s), ft) }
      token::LIT_STR(s) { ast::lit_str(p.get_str(s)) }
      token::LPAREN { expect(p, token::RPAREN); ast::lit_nil }
      _ { unexpected(p, tok); }
    }
}

fn parse_lit(p: parser) -> ast::lit {
    let sp = p.span;
    let lit = if eat_word(p, "true") {
        ast::lit_bool(true)
    } else if eat_word(p, "false") {
        ast::lit_bool(false)
    } else {
        let tok = p.token;
        p.bump();
        lit_from_token(p, tok)
    };
    ret {node: lit, span: sp};
}

fn is_ident(t: token::token) -> bool {
    alt t { token::IDENT(_, _) { ret true; } _ { } }
    ret false;
}

fn is_plain_ident(p: parser) -> bool {
    ret alt p.token { token::IDENT(_, false) { true } _ { false } };
}

fn parse_path(p: parser) -> @ast::path {
    let lo = p.span.lo;
    let global = eat(p, token::MOD_SEP), ids = [parse_ident(p)];
    while p.look_ahead(1u) != token::LT && eat(p, token::MOD_SEP) {
        ids += [parse_ident(p)];
    }
    ret @spanned(lo, p.last_span.hi,
                 {global: global, idents: ids, types: []});
}

fn parse_path_and_ty_param_substs(p: parser, colons: bool) -> @ast::path {
    let lo = p.span.lo;
    let path = parse_path(p);
    let b = if colons {
                eat(p, token::MOD_SEP)
            } else {
                p.token == token::LT
            };
    if b {
        let seq = parse_seq_lt_gt(some(token::COMMA),
                                  {|p| parse_ty(p, false)}, p);
        @spanned(lo, seq.span.hi, {types: seq.node with path.node})
    } else { path }
}

fn parse_mutability(p: parser) -> ast::mutability {
    if eat_word(p, "mutable") {
        ast::mut
    } else if eat_word(p, "const") {
        ast::maybe_mut
    } else {
        ast::imm
    }
}

fn parse_field(p: parser, sep: token::token) -> ast::field {
    let lo = p.span.lo;
    let m = parse_mutability(p);
    let i = parse_ident(p);
    expect(p, sep);
    let e = parse_expr(p);
    ret spanned(lo, e.span.hi, {mut: m, ident: i, expr: e});
}

fn mk_expr(p: parser, lo: uint, hi: uint, node: ast::expr_) -> @ast::expr {
    ret @{id: p.get_id(), node: node, span: ast_util::mk_sp(lo, hi)};
}

fn mk_mac_expr(p: parser, lo: uint, hi: uint, m: ast::mac_) -> @ast::expr {
    ret @{id: p.get_id(),
          node: ast::expr_mac({node: m, span: ast_util::mk_sp(lo, hi)}),
          span: ast_util::mk_sp(lo, hi)};
}

fn is_bar(t: token::token) -> bool {
    alt t { token::BINOP(token::OR) | token::OROR { true } _ { false } }
}

fn mk_lit_u32(p: parser, i: u32) -> @ast::expr {
    let span = p.span;
    let lv_lit = @{node: ast::lit_uint(i as u64, ast::ty_u32),
                   span: span};

    ret @{id: p.get_id(), node: ast::expr_lit(lv_lit), span: span};
}

// We don't allow single-entry tuples in the true AST; that indicates a
// parenthesized expression.  However, we preserve them temporarily while
// parsing because `(while{...})+3` parses differently from `while{...}+3`.
//
// To reflect the fact that the @ast::expr is not a true expr that should be
// part of the AST, we wrap such expressions in the pexpr enum.  They
// can then be converted to true expressions by a call to `to_expr()`.
enum pexpr {
    pexpr(@ast::expr),
}

fn mk_pexpr(p: parser, lo: uint, hi: uint, node: ast::expr_) -> pexpr {
    ret pexpr(mk_expr(p, lo, hi, node));
}

fn to_expr(e: pexpr) -> @ast::expr {
    alt e.node {
      ast::expr_tup(es) if vec::len(es) == 1u { es[0u] }
      _ { *e }
    }
}

fn parse_bottom_expr(p: parser) -> pexpr {
    let lo = p.span.lo;
    let hi = p.span.hi;

    let ex: ast::expr_;

    alt have_dollar(p) {
      some(x) {ret pexpr(mk_mac_expr(p, lo, p.span.hi, x));}
      _ {}
    }

    if p.token == token::LPAREN {
        p.bump();
        if p.token == token::RPAREN {
            hi = p.span.hi;
            p.bump();
            let lit = @spanned(lo, hi, ast::lit_nil);
            ret mk_pexpr(p, lo, hi, ast::expr_lit(lit));
        }
        let es = [parse_expr(p)];
        while p.token == token::COMMA { p.bump(); es += [parse_expr(p)]; }
        hi = p.span.hi;
        expect(p, token::RPAREN);

        // Note: we retain the expr_tup() even for simple
        // parenthesized expressions, but only for a "little while".
        // This is so that wrappers around parse_bottom_expr()
        // can tell whether the expression was parenthesized or not,
        // which affects expr_is_complete().
        ret mk_pexpr(p, lo, hi, ast::expr_tup(es));
    } else if p.token == token::LBRACE {
        p.bump();
        if is_word(p, "mutable") ||
               is_plain_ident(p) && p.look_ahead(1u) == token::COLON {
            let fields = [parse_field(p, token::COLON)];
            let base = none;
            while p.token != token::RBRACE {
                if eat_word(p, "with") { base = some(parse_expr(p)); break; }
                expect(p, token::COMMA);
                if p.token == token::RBRACE {
                    // record ends by an optional trailing comma
                    break;
                }
                fields += [parse_field(p, token::COLON)];
            }
            hi = p.span.hi;
            expect(p, token::RBRACE);
            ex = ast::expr_rec(fields, base);
        } else if is_bar(p.token) {
            ret pexpr(parse_fn_block_expr(p));
        } else {
            let blk = parse_block_tail(p, lo, ast::default_blk);
            ret mk_pexpr(p, blk.span.lo, blk.span.hi, ast::expr_block(blk));
        }
    } else if eat_word(p, "if") {
        ret pexpr(parse_if_expr(p));
    } else if eat_word(p, "for") {
        ret pexpr(parse_for_expr(p));
    } else if eat_word(p, "while") {
        ret pexpr(parse_while_expr(p));
    } else if eat_word(p, "do") {
        ret pexpr(parse_do_while_expr(p));
    } else if eat_word(p, "alt") {
        ret pexpr(parse_alt_expr(p));
    } else if eat_word(p, "fn") {
        let proto = parse_fn_ty_proto(p);
        alt proto {
          ast::proto_bare { p.fatal("fn expr are deprecated, use fn@"); }
          ast::proto_any { p.fatal("fn* cannot be used in an expression"); }
          _ { /* fallthrough */ }
        }
        ret pexpr(parse_fn_expr(p, proto));
    } else if eat_word(p, "unchecked") {
        ret pexpr(parse_block_expr(p, lo, ast::unchecked_blk));
    } else if eat_word(p, "unsafe") {
        ret pexpr(parse_block_expr(p, lo, ast::unsafe_blk));
    } else if p.token == token::LBRACKET {
        p.bump();
        let mut = parse_mutability(p);
        let es =
            parse_seq_to_end(token::RBRACKET, seq_sep(token::COMMA),
                             parse_expr, p);
        ex = ast::expr_vec(es, mut);
    } else if p.token == token::POUND_LT {
        p.bump();
        let ty = parse_ty(p, false);
        expect(p, token::GT);

        /* hack: early return to take advantage of specialized function */
        ret pexpr(mk_mac_expr(p, lo, p.span.hi,
                              ast::mac_embed_type(ty)));
    } else if p.token == token::POUND_LBRACE {
        p.bump();
        let blk = ast::mac_embed_block(
            parse_block_tail(p, lo, ast::default_blk));
        ret pexpr(mk_mac_expr(p, lo, p.span.hi, blk));
    } else if p.token == token::ELLIPSIS {
        p.bump();
        ret pexpr(mk_mac_expr(p, lo, p.span.hi, ast::mac_ellipsis));
    } else if p.token == token::POUND_LPAREN {
        p.bump();
        let e = parse_expr(p);
        expect(p, token::RPAREN);
        ret pexpr(mk_mac_expr(p, lo, p.span.hi,
                              ast::mac_qq(e.span, e)));
    } else if eat_word(p, "bind") {
        let e = parse_expr_res(p, RESTRICT_NO_CALL_EXPRS);
        fn parse_expr_opt(p: parser) -> option<@ast::expr> {
            alt p.token {
              token::UNDERSCORE { p.bump(); ret none; }
              _ { ret some(parse_expr(p)); }
            }
        }
        let es =
            parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                      parse_expr_opt, p);
        hi = es.span.hi;
        ex = ast::expr_bind(e, es.node);
    } else if p.token == token::POUND {
        let ex_ext = parse_syntax_ext(p);
        hi = ex_ext.span.hi;
        ex = ex_ext.node;
    } else if eat_word(p, "fail") {
        if can_begin_expr(p.token) {
            let e = parse_expr(p);
            hi = e.span.hi;
            ex = ast::expr_fail(some(e));
        } else { ex = ast::expr_fail(none); }
    } else if eat_word(p, "log") {
        expect(p, token::LPAREN);
        let lvl = parse_expr(p);
        expect(p, token::COMMA);
        let e = parse_expr(p);
        ex = ast::expr_log(2, lvl, e);
        hi = p.span.hi;
        expect(p, token::RPAREN);
    } else if eat_word(p, "assert") {
        let e = parse_expr(p);
        ex = ast::expr_assert(e);
        hi = e.span.hi;
    } else if eat_word(p, "check") {
        /* Should be a predicate (pure boolean function) applied to
           arguments that are all either slot variables or literals.
           but the typechecker enforces that. */

        let e = parse_expr(p);
        hi = e.span.hi;
        ex = ast::expr_check(ast::checked_expr, e);
    } else if eat_word(p, "claim") {
        /* Same rules as check, except that if check-claims
         is enabled (a command-line flag), then the parser turns
        claims into check */

        let e = parse_expr(p);
        hi = e.span.hi;
        ex = ast::expr_check(ast::claimed_expr, e);
    } else if eat_word(p, "ret") {
        if can_begin_expr(p.token) {
            let e = parse_expr(p);
            hi = e.span.hi;
            ex = ast::expr_ret(some(e));
        } else { ex = ast::expr_ret(none); }
    } else if eat_word(p, "break") {
        ex = ast::expr_break;
        hi = p.span.hi;
    } else if eat_word(p, "cont") {
        ex = ast::expr_cont;
        hi = p.span.hi;
    } else if eat_word(p, "be") {
        let e = parse_expr(p);

        // FIXME: Is this the right place for this check?
        if /*check*/ast_util::is_call_expr(e) {
            hi = e.span.hi;
            ex = ast::expr_be(e);
        } else { p.fatal("Non-call expression in tail call"); }
    } else if eat_word(p, "copy") {
        let e = parse_expr(p);
        ex = ast::expr_copy(e);
        hi = e.span.hi;
    } else if p.token == token::MOD_SEP ||
                  is_ident(p.token) && !is_word(p, "true") &&
                      !is_word(p, "false") {
        check_bad_word(p);
        let pth = parse_path_and_ty_param_substs(p, true);
        hi = pth.span.hi;
        ex = ast::expr_path(pth);
    } else {
        let lit = parse_lit(p);
        hi = lit.span.hi;
        ex = ast::expr_lit(@lit);
    }
    ret mk_pexpr(p, lo, hi, ex);
}

fn parse_block_expr(p: parser,
                    lo: uint,
                    blk_mode: ast::blk_check_mode) -> @ast::expr {
    expect(p, token::LBRACE);
    let blk = parse_block_tail(p, lo, blk_mode);
    ret mk_expr(p, blk.span.lo, blk.span.hi, ast::expr_block(blk));
}

fn parse_syntax_ext(p: parser) -> @ast::expr {
    let lo = p.span.lo;
    expect(p, token::POUND);
    ret parse_syntax_ext_naked(p, lo);
}

fn parse_syntax_ext_naked(p: parser, lo: uint) -> @ast::expr {
    alt p.token {
      token::IDENT(_, _) {}
      _ { p.fatal("expected a syntax expander name"); }
    }
    let pth = parse_path(p);
    //temporary for a backwards-compatible cycle:
    let sep = seq_sep(token::COMMA);
    let e = none;
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
                         ast::expr_vec(es.node, ast::imm)));
    }
    let b = none;
    if p.token == token::LBRACE {
        p.bump();
        let lo = p.span.lo;
        let depth = 1u;
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
    ret mk_mac_expr(p, lo, p.span.hi, ast::mac_invoc(pth, e, b));
}

fn parse_dot_or_call_expr(p: parser) -> pexpr {
    let b = parse_bottom_expr(p);
    parse_dot_or_call_expr_with(p, b)
}

fn permits_call(p: parser) -> bool {
    ret p.restriction != RESTRICT_NO_CALL_EXPRS;
}

fn parse_dot_or_call_expr_with(p: parser, e0: pexpr) -> pexpr {
    let e = e0;
    let lo = e.span.lo;
    let hi = e.span.hi;
    while !expr_is_complete(p, e) {
        alt p.token {
          // expr(...)
          token::LPAREN if permits_call(p) {
            let es = parse_seq(token::LPAREN, token::RPAREN,
                               seq_sep(token::COMMA), parse_expr, p);
            hi = es.span.hi;
            let nd = ast::expr_call(to_expr(e), es.node, false);
            e = mk_pexpr(p, lo, hi, nd);
          }

          // expr {|| ... }
          token::LBRACE if is_bar(p.look_ahead(1u)) && permits_call(p) {
            p.bump();
            let blk = parse_fn_block_expr(p);
            alt e.node {
              ast::expr_call(f, args, false) {
                e = pexpr(@{node: ast::expr_call(f, args + [blk], true)
                            with *to_expr(e)});
              }
              _ {
                e = mk_pexpr(p, lo, p.last_span.hi,
                            ast::expr_call(to_expr(e), [blk], true));
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
            e = mk_pexpr(p, lo, hi, ast::expr_index(to_expr(e), ix));
          }

          // expr.f
          token::DOT {
            p.bump();
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
                             ast::expr_field(to_expr(e),
                                             p.get_str(i),
                                             tys));
              }
              t { unexpected(p, t); }
            }
          }

          _ { ret e; }
        }
    }
    ret e;
}

fn parse_prefix_expr(p: parser) -> pexpr {
    let lo = p.span.lo;
    let hi = p.span.hi;

    let ex;
    alt p.token {
      token::NOT {
        p.bump();
        let e = to_expr(parse_prefix_expr(p));
        hi = e.span.hi;
        p.get_id(); // see ast_util::op_expr_callee_id
        ex = ast::expr_unary(ast::not, e);
      }
      token::BINOP(b) {
        alt b {
          token::MINUS {
            p.bump();
            let e = to_expr(parse_prefix_expr(p));
            hi = e.span.hi;
            p.get_id(); // see ast_util::op_expr_callee_id
            ex = ast::expr_unary(ast::neg, e);
          }
          token::STAR {
            p.bump();
            let e = to_expr(parse_prefix_expr(p));
            hi = e.span.hi;
            ex = ast::expr_unary(ast::deref, e);
          }
          _ { ret parse_dot_or_call_expr(p); }
        }
      }
      token::AT {
        p.bump();
        let m = parse_mutability(p);
        let e = to_expr(parse_prefix_expr(p));
        hi = e.span.hi;
        ex = ast::expr_unary(ast::box(m), e);
      }
      token::TILDE {
        p.bump();
        let m = parse_mutability(p);
        let e = to_expr(parse_prefix_expr(p));
        hi = e.span.hi;
        ex = ast::expr_unary(ast::uniq(m), e);
      }
      _ { ret parse_dot_or_call_expr(p); }
    }
    ret mk_pexpr(p, lo, hi, ex);
}

type op_spec = {tok: token::token, op: ast::binop, prec: int};


// FIXME make this a const, don't store it in parser state
fn prec_table() -> @[op_spec] {
    ret @[{tok: token::BINOP(token::STAR), op: ast::mul, prec: 11},
          {tok: token::BINOP(token::SLASH), op: ast::div, prec: 11},
          {tok: token::BINOP(token::PERCENT), op: ast::rem, prec: 11},
          {tok: token::BINOP(token::PLUS), op: ast::add, prec: 10},
          {tok: token::BINOP(token::MINUS), op: ast::subtract, prec: 10},
          {tok: token::BINOP(token::LSL), op: ast::lsl, prec: 9},
          {tok: token::BINOP(token::LSR), op: ast::lsr, prec: 9},
          {tok: token::BINOP(token::ASR), op: ast::asr, prec: 9},
          {tok: token::BINOP(token::AND), op: ast::bitand, prec: 8},
          {tok: token::BINOP(token::CARET), op: ast::bitxor, prec: 6},
          {tok: token::BINOP(token::OR), op: ast::bitor, prec: 6},
          // 'as' sits between here with 5
          {tok: token::LT, op: ast::lt, prec: 4},
          {tok: token::LE, op: ast::le, prec: 4},
          {tok: token::GE, op: ast::ge, prec: 4},
          {tok: token::GT, op: ast::gt, prec: 4},
          {tok: token::EQEQ, op: ast::eq, prec: 3},
          {tok: token::NE, op: ast::ne, prec: 3},
          {tok: token::ANDAND, op: ast::and, prec: 2},
          {tok: token::OROR, op: ast::or, prec: 1}];
}

fn parse_binops(p: parser) -> @ast::expr {
    ret parse_more_binops(p, parse_prefix_expr(p), 0);
}

const unop_prec: int = 100;

const as_prec: int = 5;

fn parse_more_binops(p: parser, plhs: pexpr, min_prec: int) ->
   @ast::expr {
    let lhs = to_expr(plhs);
    if expr_is_complete(p, plhs) { ret lhs; }
    let peeked = p.token;
    if peeked == token::BINOP(token::OR) &&
       p.restriction == RESTRICT_NO_BAR_OP { ret lhs; }
    for cur: op_spec in *p.precs {
        if cur.prec > min_prec && cur.tok == peeked {
            p.bump();
            let expr = parse_prefix_expr(p);
            let rhs = parse_more_binops(p, expr, cur.prec);
            p.get_id(); // see ast_util::op_expr_callee_id
            let bin = mk_pexpr(p, lhs.span.lo, rhs.span.hi,
                              ast::expr_binary(cur.op, lhs, rhs));
            ret parse_more_binops(p, bin, min_prec);
        }
    }
    if as_prec > min_prec && eat_word(p, "as") {
        let rhs = parse_ty(p, true);
        let _as =
            mk_pexpr(p, lhs.span.lo, rhs.span.hi, ast::expr_cast(lhs, rhs));
        ret parse_more_binops(p, _as, min_prec);
    }
    ret lhs;
}

fn parse_assign_expr(p: parser) -> @ast::expr {
    let lo = p.span.lo;
    let lhs = parse_binops(p);
    alt p.token {
      token::EQ {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_assign(lhs, rhs));
      }
      token::BINOPEQ(op) {
        p.bump();
        let rhs = parse_expr(p);
        let aop = ast::add;
        alt op {
          token::PLUS { aop = ast::add; }
          token::MINUS { aop = ast::subtract; }
          token::STAR { aop = ast::mul; }
          token::SLASH { aop = ast::div; }
          token::PERCENT { aop = ast::rem; }
          token::CARET { aop = ast::bitxor; }
          token::AND { aop = ast::bitand; }
          token::OR { aop = ast::bitor; }
          token::LSL { aop = ast::lsl; }
          token::LSR { aop = ast::lsr; }
          token::ASR { aop = ast::asr; }
        }
        p.get_id(); // see ast_util::op_expr_callee_id
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_assign_op(aop, lhs, rhs));
      }
      token::LARROW {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_move(lhs, rhs));
      }
      token::DARROW {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_swap(lhs, rhs));
      }
      _ {/* fall through */ }
    }
    ret lhs;
}

fn parse_if_expr_1(p: parser) ->
   {cond: @ast::expr,
    then: ast::blk,
    els: option<@ast::expr>,
    lo: uint,
    hi: uint} {
    let lo = p.last_span.lo;
    let cond = parse_expr(p);
    let thn = parse_block(p);
    let els: option<@ast::expr> = none;
    let hi = thn.span.hi;
    if eat_word(p, "else") {
        let elexpr = parse_else_expr(p);
        els = some(elexpr);
        hi = elexpr.span.hi;
    }
    ret {cond: cond, then: thn, els: els, lo: lo, hi: hi};
}

fn parse_if_expr(p: parser) -> @ast::expr {
    if eat_word(p, "check") {
        let q = parse_if_expr_1(p);
        ret mk_expr(p, q.lo, q.hi, ast::expr_if_check(q.cond, q.then, q.els));
    } else {
        let q = parse_if_expr_1(p);
        ret mk_expr(p, q.lo, q.hi, ast::expr_if(q.cond, q.then, q.els));
    }
}

// Parses:
//
//   CC := [copy ID*; move ID*]
//
// where any part is optional and trailing ; is permitted.
fn parse_capture_clause(p: parser) -> @ast::capture_clause {
    fn expect_opt_trailing_semi(p: parser) {
        if !eat(p, token::SEMI) {
            if p.token != token::RBRACKET {
                p.fatal("expecting ; or ]");
            }
        }
    }

    fn eat_ident_list(p: parser) -> [@ast::capture_item] {
        let res = [];
        while true {
            alt p.token {
              token::IDENT(_, _) {
                let id = p.get_id();
                let sp = ast_util::mk_sp(p.span.lo, p.span.hi);
                let ident = parse_ident(p);
                res += [@{id:id, name:ident, span:sp}];
                if !eat(p, token::COMMA) {
                    ret res;
                }
              }

              _ { ret res; }
            }
        }
        std::util::unreachable();
    }

    let copies = [];
    let moves = [];

    if eat(p, token::LBRACKET) {
        while !eat(p, token::RBRACKET) {
            if eat_word(p, "copy") {
                copies += eat_ident_list(p);
                expect_opt_trailing_semi(p);
            } else if eat_word(p, "move") {
                moves += eat_ident_list(p);
                expect_opt_trailing_semi(p);
            } else {
                let s: str = "expecting send, copy, or move clause";
                p.fatal(s);
            }
        }
    }

    ret @{copies: copies, moves: moves};
}

fn parse_fn_expr(p: parser, proto: ast::proto) -> @ast::expr {
    let lo = p.last_span.lo;
    let capture_clause = parse_capture_clause(p);
    let decl = parse_fn_decl(p, ast::impure_fn);
    let body = parse_block(p);
    ret mk_expr(p, lo, body.span.hi,
                ast::expr_fn(proto, decl, body, capture_clause));
}

fn parse_fn_block_expr(p: parser) -> @ast::expr {
    let lo = p.last_span.lo;
    let decl = parse_fn_block_decl(p);
    let body = parse_block_tail(p, lo, ast::default_blk);
    ret mk_expr(p, lo, body.span.hi, ast::expr_fn_block(decl, body));
}

fn parse_else_expr(p: parser) -> @ast::expr {
    if eat_word(p, "if") {
        ret parse_if_expr(p);
    } else {
        let blk = parse_block(p);
        ret mk_expr(p, blk.span.lo, blk.span.hi, ast::expr_block(blk));
    }
}

fn parse_for_expr(p: parser) -> @ast::expr {
    let lo = p.last_span.lo;
    let decl = parse_local(p, false);
    expect_word(p, "in");
    let seq = parse_expr(p);
    let body = parse_block_no_value(p);
    let hi = body.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_for(decl, seq, body));
}

fn parse_while_expr(p: parser) -> @ast::expr {
    let lo = p.last_span.lo;
    let cond = parse_expr(p);
    let body = parse_block_no_value(p);
    let hi = body.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_while(cond, body));
}

fn parse_do_while_expr(p: parser) -> @ast::expr {
    let lo = p.last_span.lo;
    let body = parse_block_no_value(p);
    expect_word(p, "while");
    let cond = parse_expr(p);
    let hi = cond.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_do_while(body, cond));
}

fn parse_alt_expr(p: parser) -> @ast::expr {
    let lo = p.last_span.lo;
    let discriminant = parse_expr(p);
    expect(p, token::LBRACE);
    let arms: [ast::arm] = [];
    while p.token != token::RBRACE {
        let pats = parse_pats(p);
        let guard = none;
        if eat_word(p, "if") { guard = some(parse_expr(p)); }
        let blk = parse_block(p);
        arms += [{pats: pats, guard: guard, body: blk}];
    }
    let hi = p.span.hi;
    p.bump();
    ret mk_expr(p, lo, hi, ast::expr_alt(discriminant, arms));
}

fn parse_expr(p: parser) -> @ast::expr {
    ret parse_expr_res(p, UNRESTRICTED);
}

fn parse_expr_res(p: parser, r: restriction) -> @ast::expr {
    let old = p.restriction;
    p.restriction = r;
    let e = parse_assign_expr(p);
    p.restriction = old;
    ret e;
}

fn parse_initializer(p: parser) -> option<ast::initializer> {
    alt p.token {
      token::EQ {
        p.bump();
        ret some({op: ast::init_assign, expr: parse_expr(p)});
      }
      token::LARROW {
        p.bump();
        ret some({op: ast::init_move, expr: parse_expr(p)});
      }
      // Now that the the channel is the first argument to receive,
      // combining it with an initializer doesn't really make sense.
      // case (token::RECV) {
      //     p.bump();
      //     ret some(rec(op = ast::init_recv,
      //                  expr = parse_expr(p)));
      // }
      _ {
        ret none;
      }
    }
}

fn parse_pats(p: parser) -> [@ast::pat] {
    let pats = [];
    while true {
        pats += [parse_pat(p)];
        if p.token == token::BINOP(token::OR) { p.bump(); } else { break; }
    }
    ret pats;
}

fn parse_pat(p: parser) -> @ast::pat {
    let lo = p.span.lo;
    let hi = p.span.hi;
    let pat;
    alt p.token {
      token::UNDERSCORE { p.bump(); pat = ast::pat_wild; }
      token::AT {
        p.bump();
        let sub = parse_pat(p);
        pat = ast::pat_box(sub);
        hi = sub.span.hi;
      }
      token::TILDE {
        p.bump();
        let sub = parse_pat(p);
        pat = ast::pat_uniq(sub);
        hi = sub.span.hi;
      }
      token::LBRACE {
        p.bump();
        let fields = [];
        let etc = false;
        let first = true;
        while p.token != token::RBRACE {
            if first { first = false; } else { expect(p, token::COMMA); }

            if p.token == token::UNDERSCORE {
                p.bump();
                if p.token != token::RBRACE {
                    p.fatal("expecting }, found " +
                                token::to_str(p.reader, p.token));
                }
                etc = true;
                break;
            }

            let lo1 = p.last_span.lo;
            let fieldname = parse_ident(p);
            let hi1 = p.last_span.lo;
            let fieldpath = ast_util::ident_to_path(ast_util::mk_sp(lo1, hi1),
                                          fieldname);
            let subpat;
            if p.token == token::COLON {
                p.bump();
                subpat = parse_pat(p);
            } else {
                if p.bad_expr_words.contains_key(fieldname) {
                    p.fatal("found " + fieldname + " in binding position");
                }
                subpat = @{id: p.get_id(),
                           node: ast::pat_ident(fieldpath, none),
                           span: ast_util::mk_sp(lo, hi)};
            }
            fields += [{ident: fieldname, pat: subpat}];
        }
        hi = p.span.hi;
        p.bump();
        pat = ast::pat_rec(fields, etc);
      }
      token::LPAREN {
        p.bump();
        if p.token == token::RPAREN {
            hi = p.span.hi;
            p.bump();
            let lit = @{node: ast::lit_nil, span: ast_util::mk_sp(lo, hi)};
            let expr = mk_expr(p, lo, hi, ast::expr_lit(lit));
            pat = ast::pat_lit(expr);
        } else {
            let fields = [parse_pat(p)];
            while p.token == token::COMMA {
                p.bump();
                fields += [parse_pat(p)];
            }
            if vec::len(fields) == 1u { expect(p, token::COMMA); }
            hi = p.span.hi;
            expect(p, token::RPAREN);
            pat = ast::pat_tup(fields);
        }
      }
      tok {
        if !is_ident(tok) || is_word(p, "true") || is_word(p, "false") {
            let val = parse_expr_res(p, RESTRICT_NO_BAR_OP);
            if eat_word(p, "to") {
                let end = parse_expr_res(p, RESTRICT_NO_BAR_OP);
                hi = end.span.hi;
                pat = ast::pat_range(val, end);
            } else {
                hi = val.span.hi;
                pat = ast::pat_lit(val);
            }
        } else if is_plain_ident(p) &&
                      alt p.look_ahead(1u) {
                        token::LPAREN | token::LBRACKET |
                            token::LT {
                          false
                        }
                        _ { true }
                      } {
            let name = parse_path(p);
            let sub = if eat(p, token::AT) {
                          some(parse_pat(p))
                      } else {
                          none
                      };
            pat = ast::pat_ident(name, sub);
        } else {
            let enum_path = parse_path_and_ty_param_substs(p, true);
            hi = enum_path.span.hi;
            let args: [@ast::pat];
            alt p.token {
              token::LPAREN {
                let a =
                    parse_seq(token::LPAREN, token::RPAREN,
                              seq_sep(token::COMMA), parse_pat, p);
                args = a.node;
                hi = a.span.hi;
              }
              _ { args = []; }
            }
            // at this point, we're not sure whether it's a enum or a bind
            if vec::len(args) == 0u &&
               vec::len(enum_path.node.idents) == 1u {
                pat = ast::pat_ident(enum_path, none);
            }
            else {
                pat = ast::pat_enum(enum_path, args);
            }
        }
      }
    }
    ret @{id: p.get_id(), node: pat, span: ast_util::mk_sp(lo, hi)};
}

fn parse_local(p: parser, allow_init: bool) -> @ast::local {
    let lo = p.span.lo;
    let pat = parse_pat(p);
    let ty = @spanned(lo, lo, ast::ty_infer);
    if eat(p, token::COLON) { ty = parse_ty(p, false); }
    let init = if allow_init { parse_initializer(p) } else { none };
    ret @spanned(lo, p.last_span.hi,
                 {ty: ty, pat: pat, init: init, id: p.get_id()});
}

fn parse_let(p: parser) -> @ast::decl {
    fn parse_let_style(p: parser) -> ast::let_style {
        if eat(p, token::BINOP(token::AND)) {
            ast::let_ref
        } else {
            ast::let_copy
        }
    }
    let lo = p.span.lo;
    let locals = [(parse_let_style(p), parse_local(p, true))];
    while eat(p, token::COMMA) {
        locals += [(parse_let_style(p), parse_local(p, true))];
    }
    ret @spanned(lo, p.last_span.hi, ast::decl_local(locals));
}

fn parse_stmt(p: parser, first_item_attrs: [ast::attribute]) -> @ast::stmt {
    fn check_expected_item(p: parser, current_attrs: [ast::attribute]) {
        // If we have attributes then we should have an item
        if vec::is_not_empty(current_attrs) {
            p.fatal("expected item");
        }
    }

    let lo = p.span.lo;
    if is_word(p, "let") {
        check_expected_item(p, first_item_attrs);
        expect_word(p, "let");
        let decl = parse_let(p);
        ret @spanned(lo, decl.span.hi, ast::stmt_decl(decl, p.get_id()));
    } else {
        let item_attrs;
        alt parse_outer_attrs_or_ext(p, first_item_attrs) {
          none { item_attrs = []; }
          some(left(attrs)) { item_attrs = attrs; }
          some(right(ext)) {
            ret @spanned(lo, ext.span.hi, ast::stmt_expr(ext, p.get_id()));
          }
        }

        let item_attrs = first_item_attrs + item_attrs;

        alt parse_item(p, item_attrs) {
          some(i) {
            let hi = i.span.hi;
            let decl = @spanned(lo, hi, ast::decl_item(i));
            ret @spanned(lo, hi, ast::stmt_decl(decl, p.get_id()));
          }
          none() { /* fallthrough */ }
        }

        check_expected_item(p, item_attrs);

        // Remainder are line-expr stmts.
        let e = parse_expr_res(p, RESTRICT_STMT_EXPR);
        ret @spanned(lo, e.span.hi, ast::stmt_expr(e, p.get_id()));
    }
}

fn expr_is_complete(p: parser, e: pexpr) -> bool {
    log(debug, ("expr_is_complete", p.restriction,
                print::pprust::expr_to_str(*e),
                expr_requires_semi_to_be_stmt(*e)));
    ret p.restriction == RESTRICT_STMT_EXPR &&
        !expr_requires_semi_to_be_stmt(*e);
}

fn expr_requires_semi_to_be_stmt(e: @ast::expr) -> bool {
    alt e.node {
      ast::expr_if(_, _, _) | ast::expr_if_check(_, _, _)
      | ast::expr_alt(_, _) | ast::expr_block(_)
      | ast::expr_do_while(_, _) | ast::expr_while(_, _)
      | ast::expr_for(_, _, _)
      | ast::expr_call(_, _, true) {
        false
      }
      _ { true }
    }
}

fn stmt_ends_with_semi(stmt: ast::stmt) -> bool {
    alt stmt.node {
      ast::stmt_decl(d, _) {
        ret alt d.node {
              ast::decl_local(_) { true }
              ast::decl_item(_) { false }
            }
      }
      ast::stmt_expr(e, _) {
        ret expr_requires_semi_to_be_stmt(e);
      }
      ast::stmt_semi(e, _) {
        ret false;
      }
    }
}

fn parse_block(p: parser) -> ast::blk {
    let (attrs, blk) = parse_inner_attrs_and_block(p, false);
    assert vec::is_empty(attrs);
    ret blk;
}

fn parse_inner_attrs_and_block(
    p: parser, parse_attrs: bool) -> ([ast::attribute], ast::blk) {

    fn maybe_parse_inner_attrs_and_next(
        p: parser, parse_attrs: bool) ->
        {inner: [ast::attribute], next: [ast::attribute]} {
        if parse_attrs {
            parse_inner_attrs_and_next(p)
        } else {
            {inner: [], next: []}
        }
    }

    let lo = p.span.lo;
    if eat_word(p, "unchecked") {
        expect(p, token::LBRACE);
        let {inner, next} = maybe_parse_inner_attrs_and_next(p, parse_attrs);
        ret (inner, parse_block_tail_(p, lo, ast::unchecked_blk, next));
    } else if eat_word(p, "unsafe") {
        expect(p, token::LBRACE);
        let {inner, next} = maybe_parse_inner_attrs_and_next(p, parse_attrs);
        ret (inner, parse_block_tail_(p, lo, ast::unsafe_blk, next));
    } else {
        expect(p, token::LBRACE);
        let {inner, next} = maybe_parse_inner_attrs_and_next(p, parse_attrs);
        ret (inner, parse_block_tail_(p, lo, ast::default_blk, next));
    }
}

fn parse_block_no_value(p: parser) -> ast::blk {
    // We parse blocks that cannot have a value the same as any other block;
    // the type checker will make sure that the tail expression (if any) has
    // unit type.
    ret parse_block(p);
}

// Precondition: already parsed the '{' or '#{'
// I guess that also means "already parsed the 'impure'" if
// necessary, and this should take a qualifier.
// some blocks start with "#{"...
fn parse_block_tail(p: parser, lo: uint, s: ast::blk_check_mode) -> ast::blk {
    parse_block_tail_(p, lo, s, [])
}

fn parse_block_tail_(p: parser, lo: uint, s: ast::blk_check_mode,
                     first_item_attrs: [ast::attribute]) -> ast::blk {
    let stmts = [];
    let expr = none;
    let view_items = maybe_parse_view_import_only(p, first_item_attrs);
    let initial_attrs = first_item_attrs;

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
              ast::stmt_expr(e, stmt_id) { // Expression without semicolon:
                alt p.token {
                  token::SEMI {
                    p.bump();
                    stmts += [@{node: ast::stmt_semi(e, stmt_id) with *stmt}];
                  }
                  token::RBRACE {
                    expr = some(e);
                  }
                  t {
                    if stmt_ends_with_semi(*stmt) {
                        p.fatal("expected ';' or '}' after expression but \
                                 found '" + token::to_str(p.reader, t) +
                                "'");
                    }
                    stmts += [stmt];
                  }
                }
              }

              _ { // All other kinds of statements:
                stmts += [stmt];

                if stmt_ends_with_semi(*stmt) {
                    expect(p, token::SEMI);
                }
              }
            }
          }
        }
    }
    let hi = p.span.hi;
    p.bump();
    let bloc = {view_items: view_items, stmts: stmts, expr: expr,
                id: p.get_id(), rules: s};
    ret spanned(lo, hi, bloc);
}

fn parse_ty_param(p: parser) -> ast::ty_param {
    let bounds = [];
    let ident = parse_ident(p);
    if eat(p, token::COLON) {
        while p.token != token::COMMA && p.token != token::GT {
            if eat_word(p, "send") { bounds += [ast::bound_send]; }
            else if eat_word(p, "copy") { bounds += [ast::bound_copy]; }
            else { bounds += [ast::bound_iface(parse_ty(p, false))]; }
        }
    }
    ret {ident: ident, id: p.get_id(), bounds: @bounds};
}

fn parse_ty_params(p: parser) -> [ast::ty_param] {
    if eat(p, token::LT) {
        parse_seq_to_gt(some(token::COMMA), parse_ty_param, p)
    } else { [] }
}

fn parse_fn_decl(p: parser, purity: ast::purity)
    -> ast::fn_decl {
    let inputs: ast::spanned<[ast::arg]> =
        parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_arg, p);
    // Use the args list to translate each bound variable
    // mentioned in a constraint to an arg index.
    // Seems weird to do this in the parser, but I'm not sure how else to.
    let constrs = [];
    if p.token == token::COLON {
        p.bump();
        constrs = parse_constrs({|x| parse_ty_constr(inputs.node, x) }, p);
    }
    let (ret_style, ret_ty) = parse_ret_ty(p);
    ret {inputs: inputs.node,
         output: ret_ty,
         purity: purity,
         cf: ret_style,
         constraints: constrs};
}

fn parse_fn_block_decl(p: parser) -> ast::fn_decl {
    let inputs = if eat(p, token::OROR) {
                     []
                 } else {
                     parse_seq(token::BINOP(token::OR),
                               token::BINOP(token::OR),
                               seq_sep(token::COMMA),
                               parse_fn_block_arg, p).node
                 };
    let output = if eat(p, token::RARROW) {
                     parse_ty(p, false)
                 } else {
                     @spanned(p.span.lo, p.span.hi, ast::ty_infer)
                 };
    ret {inputs: inputs,
         output: output,
         purity: ast::impure_fn,
         cf: ast::return_val,
         constraints: []};
}

fn parse_fn_header(p: parser) -> {ident: ast::ident, tps: [ast::ty_param]} {
    let id = parse_value_ident(p);
    let ty_params = parse_ty_params(p);
    ret {ident: id, tps: ty_params};
}

fn mk_item(p: parser, lo: uint, hi: uint, ident: ast::ident, node: ast::item_,
           attrs: [ast::attribute]) -> @ast::item {
    ret @{ident: ident,
          attrs: attrs,
          id: p.get_id(),
          node: node,
          span: ast_util::mk_sp(lo, hi)};
}

fn parse_item_fn(p: parser, purity: ast::purity,
                 attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo;
    let t = parse_fn_header(p);
    let decl = parse_fn_decl(p, purity);
    let (inner_attrs, body) = parse_inner_attrs_and_block(p, true);
    let attrs = attrs + inner_attrs;
    ret mk_item(p, lo, body.span.hi, t.ident,
                ast::item_fn(decl, t.tps, body), attrs);
}

fn parse_method_name(p: parser) -> ast::ident {
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

fn parse_method(p: parser) -> @ast::method {
    let attrs = parse_outer_attributes(p);
    let lo = p.span.lo;
    expect_word(p, "fn");
    let ident = parse_method_name(p);
    let tps = parse_ty_params(p);
    let decl = parse_fn_decl(p, ast::impure_fn);
    let (inner_attrs, body) = parse_inner_attrs_and_block(p, true);
    let attrs = attrs + inner_attrs;
    @{ident: ident, attrs: attrs, tps: tps, decl: decl, body: body,
      id: p.get_id(), span: ast_util::mk_sp(lo, body.span.hi)}
}

fn parse_item_iface(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo, ident = parse_ident(p),
        tps = parse_ty_params(p), meths = parse_ty_methods(p);
    ret mk_item(p, lo, p.last_span.hi, ident,
                ast::item_iface(tps, meths), attrs);
}

// Parses three variants (with the initial params always optional):
//    impl <T: copy> of to_str for [T] { ... }
//    impl name<T> of to_str for [T] { ... }
//    impl name<T> for [T] { ... }
fn parse_item_impl(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo;
    fn wrap_path(p: parser, pt: @ast::path) -> @ast::ty {
        @{node: ast::ty_path(pt, p.get_id()), span: pt.span}
    }
    let (ident, tps) = if !is_word(p, "of") {
        if p.token == token::LT { (none, parse_ty_params(p)) }
        else { (some(parse_ident(p)), parse_ty_params(p)) }
    } else { (none, []) };
    let ifce = if eat_word(p, "of") {
        let path = parse_path_and_ty_param_substs(p, false);
        if option::is_none(ident) {
            ident = some(path.node.idents[vec::len(path.node.idents) - 1u]);
        }
        some(wrap_path(p, path))
    } else { none };
    let ident = alt ident {
        some(name) { name }
        none { expect_word(p, "of"); fail; }
    };
    expect_word(p, "for");
    let ty = parse_ty(p, false), meths = [];
    expect(p, token::LBRACE);
    while !eat(p, token::RBRACE) { meths += [parse_method(p)]; }
    ret mk_item(p, lo, p.last_span.hi, ident,
                ast::item_impl(tps, ifce, ty, meths), attrs);
}

fn parse_item_res(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo;
    let ident = parse_value_ident(p);
    let ty_params = parse_ty_params(p);
    expect(p, token::LPAREN);
    let arg_ident = parse_value_ident(p);
    expect(p, token::COLON);
    let t = parse_ty(p, false);
    expect(p, token::RPAREN);
    let dtor = parse_block_no_value(p);
    let decl =
        {inputs:
             [{mode: ast::by_ref, ty: t, ident: arg_ident,
               id: p.get_id()}],
         output: @spanned(lo, lo, ast::ty_nil),
         purity: ast::impure_fn,
         cf: ast::return_val,
         constraints: []};
    ret mk_item(p, lo, dtor.span.hi, ident,
                ast::item_res(decl, ty_params, dtor, p.get_id(), p.get_id()),
                attrs);
}

fn parse_mod_items(p: parser, term: token::token,
                   first_item_attrs: [ast::attribute]) -> ast::_mod {
    // Shouldn't be any view items since we've already parsed an item attr
    let view_items = maybe_parse_view(p, first_item_attrs);
    let items: [@ast::item] = [];
    let initial_attrs = first_item_attrs;
    while p.token != term {
        let attrs = initial_attrs + parse_outer_attributes(p);
        initial_attrs = [];
        #debug["parse_mod_items: parse_item(attrs=%?)", attrs];
        alt parse_item(p, attrs) {
          some(i) { items += [i]; }
          _ {
            p.fatal("expected item but found '" +
                    token::to_str(p.reader, p.token) + "'");
          }
        }
        #debug["parse_mod_items: attrs=%?", attrs];
    }
    ret {view_items: view_items, items: items};
}

fn parse_item_const(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo;
    let id = parse_value_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    expect(p, token::EQ);
    let e = parse_expr(p);
    let hi = p.span.hi;
    expect(p, token::SEMI);
    ret mk_item(p, lo, hi, id, ast::item_const(ty, e), attrs);
}

fn parse_item_mod(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo;
    let id = parse_ident(p);
    expect(p, token::LBRACE);
    let inner_attrs = parse_inner_attrs_and_next(p);
    let first_item_outer_attrs = inner_attrs.next;
    let m = parse_mod_items(p, token::RBRACE, first_item_outer_attrs);
    let hi = p.span.hi;
    expect(p, token::RBRACE);
    ret mk_item(p, lo, hi, id, ast::item_mod(m), attrs + inner_attrs.inner);
}

fn parse_item_native_fn(p: parser, attrs: [ast::attribute],
                        purity: ast::purity) -> @ast::native_item {
    let lo = p.last_span.lo;
    let t = parse_fn_header(p);
    let decl = parse_fn_decl(p, purity);
    let hi = p.span.hi;
    expect(p, token::SEMI);
    ret @{ident: t.ident,
          attrs: attrs,
          node: ast::native_item_fn(decl, t.tps),
          id: p.get_id(),
          span: ast_util::mk_sp(lo, hi)};
}

fn parse_native_item(p: parser, attrs: [ast::attribute]) ->
   @ast::native_item {
    if eat_word(p, "fn") {
        ret parse_item_native_fn(p, attrs, ast::impure_fn);
    } else if eat_word(p, "pure") {
        expect_word(p, "fn");
        ret parse_item_native_fn(p, attrs, ast::pure_fn);
    } else if eat_word(p, "unsafe") {
        expect_word(p, "fn");
        ret parse_item_native_fn(p, attrs, ast::unsafe_fn);
    } else { unexpected(p, p.token); }
}

fn parse_native_mod_items(p: parser, first_item_attrs: [ast::attribute]) ->
   ast::native_mod {
    // Shouldn't be any view items since we've already parsed an item attr
    let view_items =
        if vec::len(first_item_attrs) == 0u {
            parse_native_view(p)
        } else { [] };
    let items: [@ast::native_item] = [];
    let initial_attrs = first_item_attrs;
    while p.token != token::RBRACE {
        let attrs = initial_attrs + parse_outer_attributes(p);
        initial_attrs = [];
        items += [parse_native_item(p, attrs)];
    }
    ret {view_items: view_items,
         items: items};
}

fn parse_item_native_mod(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo;
    expect_word(p, "mod");
    let id = parse_ident(p);
    expect(p, token::LBRACE);
    let more_attrs = parse_inner_attrs_and_next(p);
    let inner_attrs = more_attrs.inner;
    let first_item_outer_attrs = more_attrs.next;
    let m = parse_native_mod_items(p, first_item_outer_attrs);
    let hi = p.span.hi;
    expect(p, token::RBRACE);
    ret mk_item(p, lo, hi, id, ast::item_native_mod(m), attrs + inner_attrs);
}

fn parse_type_decl(p: parser) -> {lo: uint, ident: ast::ident} {
    let lo = p.last_span.lo;
    let id = parse_ident(p);
    ret {lo: lo, ident: id};
}

fn parse_item_type(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let t = parse_type_decl(p);
    let tps = parse_ty_params(p);
    expect(p, token::EQ);
    let ty = parse_ty(p, false);
    let hi = p.span.hi;
    expect(p, token::SEMI);
    ret mk_item(p, t.lo, hi, t.ident, ast::item_ty(ty, tps), attrs);
}

fn parse_item_enum(p: parser, attrs: [ast::attribute]) -> @ast::item {
    let lo = p.last_span.lo;
    let id = parse_ident(p);
    let ty_params = parse_ty_params(p);
    let variants: [ast::variant] = [];
    // Newtype syntax
    if p.token == token::EQ {
        if p.bad_expr_words.contains_key(id) {
            p.fatal("found " + id + " in enum constructor position");
        }
        p.bump();
        let ty = parse_ty(p, false);
        expect(p, token::SEMI);
        let variant =
            spanned(ty.span.lo, ty.span.hi,
                    {name: id,
                     attrs: [],
                     args: [{ty: ty, id: p.get_id()}],
                     id: p.get_id(),
                     disr_expr: none});
        ret mk_item(p, lo, ty.span.hi, id,
                    ast::item_enum([variant], ty_params), attrs);
    }
    expect(p, token::LBRACE);

    let all_nullary = true, have_disr = false;

    while p.token != token::RBRACE {
        let variant_attrs = parse_outer_attributes(p);
        let vlo = p.span.lo;
        let ident = parse_value_ident(p);
        let args = [], disr_expr = none;
        if p.token == token::LPAREN {
            all_nullary = false;
            let arg_tys = parse_seq(token::LPAREN, token::RPAREN,
                                    seq_sep(token::COMMA),
                                    {|p| parse_ty(p, false)}, p);
            for ty in arg_tys.node {
                args += [{ty: ty, id: p.get_id()}];
            }
        } else if eat(p, token::EQ) {
            have_disr = true;
            disr_expr = some(parse_expr(p));
        }

        let vr = {name: ident, attrs: variant_attrs,
                  args: args, id: p.get_id(),
                  disr_expr: disr_expr};
        variants += [spanned(vlo, p.last_span.hi, vr)];

        if !eat(p, token::COMMA) { break; }
    }
    expect(p, token::RBRACE);
    if (have_disr && !all_nullary) {
        p.fatal("discriminator values can only be used with a c-like enum");
    }
    ret mk_item(p, lo, p.last_span.hi, id,
                ast::item_enum(variants, ty_params), attrs);
}

fn parse_fn_ty_proto(p: parser) -> ast::proto {
    alt p.token {
      token::AT {
        p.bump();
        ast::proto_box
      }
      token::TILDE {
        p.bump();
        ast::proto_uniq
      }
      token::BINOP(token::AND) {
        p.bump();
        ast::proto_block
      }
      _ {
        ast::proto_any
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

fn parse_item(p: parser, attrs: [ast::attribute]) -> option<@ast::item> {
    if eat_word(p, "const") {
        ret some(parse_item_const(p, attrs));
    } else if eat_word(p, "inline") {
        expect_word(p, "fn");
        ret some(parse_item_fn(p, ast::impure_fn, attrs));
    } else if is_word(p, "fn") && !fn_expr_lookahead(p.look_ahead(1u)) {
        p.bump();
        ret some(parse_item_fn(p, ast::impure_fn, attrs));
    } else if eat_word(p, "pure") {
        expect_word(p, "fn");
        ret some(parse_item_fn(p, ast::pure_fn, attrs));
    } else if is_word(p, "unsafe") && p.look_ahead(1u) != token::LBRACE {
        p.bump();
        expect_word(p, "fn");
        ret some(parse_item_fn(p, ast::unsafe_fn, attrs));
    } else if eat_word(p, "mod") {
        ret some(parse_item_mod(p, attrs));
    } else if eat_word(p, "native") {
        ret some(parse_item_native_mod(p, attrs));
    } if eat_word(p, "type") {
        ret some(parse_item_type(p, attrs));
    } else if eat_word(p, "enum") {
        ret some(parse_item_enum(p, attrs));
    } else if eat_word(p, "iface") {
        ret some(parse_item_iface(p, attrs));
    } else if eat_word(p, "impl") {
        ret some(parse_item_impl(p, attrs));
    } else if eat_word(p, "resource") {
        ret some(parse_item_res(p, attrs));
    } else { ret none; }
}

// A type to distingush between the parsing of item attributes or syntax
// extensions, which both begin with token.POUND
type attr_or_ext = option<either::t<[ast::attribute], @ast::expr>>;

fn parse_outer_attrs_or_ext(
    p: parser,
    first_item_attrs: [ast::attribute]) -> attr_or_ext {
    let expect_item_next = vec::is_not_empty(first_item_attrs);
    if p.token == token::POUND {
        let lo = p.span.lo;
        if p.look_ahead(1u) == token::LBRACKET {
            p.bump();
            let first_attr = parse_attribute_naked(p, ast::attr_outer, lo);
            ret some(left([first_attr] + parse_outer_attributes(p)));
        } else if !(p.look_ahead(1u) == token::LT
                    || p.look_ahead(1u) == token::LBRACKET
                    || expect_item_next) {
            p.bump();
            ret some(right(parse_syntax_ext_naked(p, lo)));
        } else { ret none; }
    } else { ret none; }
}

// Parse attributes that appear before an item
fn parse_outer_attributes(p: parser) -> [ast::attribute] {
    let attrs: [ast::attribute] = [];
    while p.token == token::POUND {
        attrs += [parse_attribute(p, ast::attr_outer)];
    }
    ret attrs;
}

fn parse_attribute(p: parser, style: ast::attr_style) -> ast::attribute {
    let lo = p.span.lo;
    expect(p, token::POUND);
    ret parse_attribute_naked(p, style, lo);
}

fn parse_attribute_naked(p: parser, style: ast::attr_style, lo: uint) ->
   ast::attribute {
    expect(p, token::LBRACKET);
    let meta_item = parse_meta_item(p);
    expect(p, token::RBRACKET);
    let hi = p.span.hi;
    ret spanned(lo, hi, {style: style, value: *meta_item});
}

// Parse attributes that appear after the opening of an item, each terminated
// by a semicolon. In addition to a vector of inner attributes, this function
// also returns a vector that may contain the first outer attribute of the
// next item (since we can't know whether the attribute is an inner attribute
// of the containing item or an outer attribute of the first contained item
// until we see the semi).
fn parse_inner_attrs_and_next(p: parser) ->
   {inner: [ast::attribute], next: [ast::attribute]} {
    let inner_attrs: [ast::attribute] = [];
    let next_outer_attrs: [ast::attribute] = [];
    while p.token == token::POUND {
        if p.look_ahead(1u) != token::LBRACKET {
            // This is an extension
            break;
        }
        let attr = parse_attribute(p, ast::attr_inner);
        if p.token == token::SEMI {
            p.bump();
            inner_attrs += [attr];
        } else {
            // It's not really an inner attribute
            let outer_attr =
                spanned(attr.span.lo, attr.span.hi,
                        {style: ast::attr_outer, value: attr.node.value});
            next_outer_attrs += [outer_attr];
            break;
        }
    }
    ret {inner: inner_attrs, next: next_outer_attrs};
}

fn parse_meta_item(p: parser) -> @ast::meta_item {
    let lo = p.span.lo;
    let ident = parse_ident(p);
    alt p.token {
      token::EQ {
        p.bump();
        let lit = parse_lit(p);
        let hi = p.span.hi;
        ret @spanned(lo, hi, ast::meta_name_value(ident, lit));
      }
      token::LPAREN {
        let inner_items = parse_meta_seq(p);
        let hi = p.span.hi;
        ret @spanned(lo, hi, ast::meta_list(ident, inner_items));
      }
      _ {
        let hi = p.span.hi;
        ret @spanned(lo, hi, ast::meta_word(ident));
      }
    }
}

fn parse_meta_seq(p: parser) -> [@ast::meta_item] {
    ret parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_meta_item, p).node;
}

fn parse_optional_meta(p: parser) -> [@ast::meta_item] {
    alt p.token { token::LPAREN { ret parse_meta_seq(p); } _ { ret []; } }
}

fn parse_use(p: parser) -> ast::view_item_ {
    let ident = parse_ident(p);
    let metadata = parse_optional_meta(p);
    ret ast::view_item_use(ident, metadata, p.get_id());
}

fn parse_rest_import_name(p: parser, first: ast::ident,
                          def_ident: option<ast::ident>) ->
   ast::view_item_ {
    let identifiers: [ast::ident] = [first];
    let glob: bool = false;
    let from_idents = option::none::<[ast::import_ident]>;
    while true {
        alt p.token {
          token::SEMI { break; }
          token::MOD_SEP {
            if glob { p.fatal("cannot path into a glob"); }
            if option::is_some(from_idents) {
                p.fatal("cannot path into import list");
            }
            p.bump();
          }
          _ { p.fatal("expecting '::' or ';'"); }
        }
        alt p.token {
          token::IDENT(_, _) { identifiers += [parse_ident(p)]; }





          //the lexer can't tell the different kinds of stars apart ) :
          token::BINOP(token::STAR) {
            glob = true;
            p.bump();
          }





          token::LBRACE {
            let from_idents_ =
                parse_seq(token::LBRACE, token::RBRACE, seq_sep(token::COMMA),
                          parse_import_ident, p).node;
            if vec::is_empty(from_idents_) {
                p.fatal("at least one import is required");
            }
            from_idents = some(from_idents_);
          }





          _ {
            p.fatal("expecting an identifier, or '*'");
          }
        }
    }
    alt def_ident {
      some(i) {
        if glob { p.fatal("globbed imports can't be renamed"); }
        if option::is_some(from_idents) {
            p.fatal("can't rename import list");
        }
        ret ast::view_item_import(i, @identifiers, p.get_id());
      }
      _ {
        if glob {
            ret ast::view_item_import_glob(@identifiers, p.get_id());
        } else if option::is_some(from_idents) {
            ret ast::view_item_import_from(@identifiers,
                                           option::get(from_idents),
                                           p.get_id());
        } else {
            let len = vec::len(identifiers);
            ret ast::view_item_import(identifiers[len - 1u], @identifiers,
                                      p.get_id());
        }
      }
    }
}

fn parse_full_import_name(p: parser, def_ident: ast::ident) ->
   ast::view_item_ {
    alt p.token {
      token::IDENT(i, _) {
        p.bump();
        ret parse_rest_import_name(p, p.get_str(i), some(def_ident));
      }
      _ { p.fatal("expecting an identifier"); }
    }
}

fn parse_import(p: parser) -> ast::view_item_ {
    alt p.token {
      token::IDENT(i, _) {
        p.bump();
        alt p.token {
          token::EQ {
            p.bump();
            ret parse_full_import_name(p, p.get_str(i));
          }
          _ { ret parse_rest_import_name(p, p.get_str(i), none); }
        }
      }
      _ { p.fatal("expecting an identifier"); }
    }
}

fn parse_enum_export(p:parser, tyname:ast::ident) -> ast::view_item_ {
    let enumnames:[ast::import_ident] =
        parse_seq(token::LBRACE, token::RBRACE,
             seq_sep(token::COMMA), {|p| parse_import_ident(p) }, p).node;
    let id = p.get_id();
    if vec::is_empty(enumnames) {
       ret ast::view_item_export_enum_none(tyname, id);
    }
    else {
       ret ast::view_item_export_enum_some(tyname, enumnames, id);
    }
}

fn parse_export(p: parser) -> ast::view_item_ {
    let first = parse_ident(p);
    alt p.token {
       token::MOD_SEP {
           p.bump();
           ret parse_enum_export(p, first);
       }
       t {
           if t == token::COMMA { p.bump(); }
           let ids =
               parse_seq_to_before_end(token::SEMI, seq_sep(token::COMMA),
                                       parse_ident, p);
           ret ast::view_item_export(vec::concat([[first], ids]), p.get_id());
       }
    }
}

fn parse_view_item(p: parser) -> @ast::view_item {
    let lo = p.span.lo;
    let the_item =
        if eat_word(p, "use") {
            parse_use(p)
        } else if eat_word(p, "import") {
            parse_import(p)
        } else if eat_word(p, "export") { parse_export(p) } else { fail };
    let hi = p.span.lo;
    expect(p, token::SEMI);
    ret @spanned(lo, hi, the_item);
}

fn is_view_item(p: parser) -> bool {
    alt p.token {
      token::IDENT(sid, false) {
        let st = p.get_str(sid);
        ret str::eq(st, "use") || str::eq(st, "import") ||
                str::eq(st, "export");
      }
      _ { ret false; }
    }
}

fn maybe_parse_view(
    p: parser,
    first_item_attrs: [ast::attribute]) -> [@ast::view_item] {

    maybe_parse_view_while(p, first_item_attrs, is_view_item)
}

fn maybe_parse_view_import_only(
    p: parser,
    first_item_attrs: [ast::attribute]) -> [@ast::view_item] {

    maybe_parse_view_while(p, first_item_attrs, bind is_word(_, "import"))
}

fn maybe_parse_view_while(
    p: parser,
    first_item_attrs: [ast::attribute],
    f: fn@(parser) -> bool) -> [@ast::view_item] {

    if vec::len(first_item_attrs) == 0u {
        let items = [];
        while f(p) { items += [parse_view_item(p)]; }
        ret items;
    } else {
        // Shouldn't be any view items since we've already parsed an item attr
        ret [];
    }
}

fn parse_native_view(p: parser) -> [@ast::view_item] {
    maybe_parse_view_while(p, [], is_view_item)
}

fn parse_crate_from_source_file(input: str, cfg: ast::crate_cfg,
                                sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, SOURCE_FILE);
    let r = parse_crate_mod(p, cfg);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}


fn parse_expr_from_source_str(name: str, source: @str, cfg: ast::crate_cfg,
                              sess: parse_sess) -> @ast::expr {
    let p = new_parser_from_source_str(sess, cfg, name, none, source);
    let r = parse_expr(p);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn parse_from_source_str<T>(f: fn (p: parser) -> T,
                            name: str, ss: codemap::file_substr,
                            source: @str, cfg: ast::crate_cfg,
                            sess: parse_sess)
    -> T
{
    let p = new_parser_from_source_str(sess, cfg, name, ss, source);
    let r = f(p);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn parse_crate_from_source_str(name: str, source: @str, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_source_str(sess, cfg, name, none, source);
    let r = parse_crate_mod(p, cfg);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

// Parses a source module as a crate
fn parse_crate_mod(p: parser, _cfg: ast::crate_cfg) -> @ast::crate {
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
fn parse_crate_directive(p: parser, first_outer_attr: [ast::attribute]) ->
   ast::crate_directive {

    // Collect the next attributes
    let outer_attrs = first_outer_attr + parse_outer_attributes(p);
    // In a crate file outer attributes are only going to apply to mods
    let expect_mod = vec::len(outer_attrs) > 0u;

    let lo = p.span.lo;
    if expect_mod || is_word(p, "mod") {
        expect_word(p, "mod");
        let id = parse_ident(p);
        alt p.token {
          // mod x = "foo.rs";
          token::SEMI {
            let hi = p.span.hi;
            p.bump();
            ret spanned(lo, hi, ast::cdir_src_mod(id, outer_attrs));
          }
          // mod x = "foo_dir" { ...directives... }
          token::LBRACE {
            p.bump();
            let inner_attrs = parse_inner_attrs_and_next(p);
            let mod_attrs = outer_attrs + inner_attrs.inner;
            let next_outer_attr = inner_attrs.next;
            let cdirs =
                parse_crate_directives(p, token::RBRACE, next_outer_attr);
            let hi = p.span.hi;
            expect(p, token::RBRACE);
            ret spanned(lo, hi,
                        ast::cdir_dir_mod(id, cdirs, mod_attrs));
          }
          t { unexpected(p, t); }
        }
    } else if is_view_item(p) {
        let vi = parse_view_item(p);
        ret spanned(lo, vi.span.hi, ast::cdir_view_item(vi));
    } else { ret p.fatal("expected crate directive"); }
}

fn parse_crate_directives(p: parser, term: token::token,
                          first_outer_attr: [ast::attribute]) ->
   [@ast::crate_directive] {

    // This is pretty ugly. If we have an outer attribute then we can't accept
    // seeing the terminator next, so if we do see it then fail the same way
    // parse_crate_directive would
    if vec::len(first_outer_attr) > 0u && p.token == term {
        expect_word(p, "mod");
    }

    let cdirs: [@ast::crate_directive] = [];
    let first_outer_attr = first_outer_attr;
    while p.token != term {
        let cdir = @parse_crate_directive(p, first_outer_attr);
        cdirs += [cdir];
        first_outer_attr = [];
    }
    ret cdirs;
}

fn parse_crate_from_crate_file(input: str, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, CRATE_FILE);
    let lo = p.span.lo;
    let prefix = std::fs::dirname(p.reader.filemap.name);
    let leading_attrs = parse_inner_attrs_and_next(p);
    let crate_attrs = leading_attrs.inner;
    let first_cdir_attr = leading_attrs.next;
    let cdirs = parse_crate_directives(p, token::EOF, first_cdir_attr);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    let cx =
        @{p: p,
          sess: sess,
          cfg: p.cfg};
    let (companionmod, _) = fs::splitext(fs::basename(input));
    let (m, attrs) = eval::eval_crate_directives_to_mod(
        cx, cdirs, prefix, option::some(companionmod));
    let hi = p.span.hi;
    expect(p, token::EOF);
    ret @spanned(lo, hi,
                 {directives: cdirs,
                  module: m,
                  attrs: crate_attrs + attrs,
                  config: p.cfg});
}

fn parse_crate_from_file(input: str, cfg: ast::crate_cfg, sess: parse_sess) ->
   @ast::crate {
    if str::ends_with(input, ".rc") {
        parse_crate_from_crate_file(input, cfg, sess)
    } else if str::ends_with(input, ".rs") {
        parse_crate_from_source_file(input, cfg, sess)
    } else {
        sess.span_diagnostic.handler().fatal("unknown input file type: " +
                                             input)
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

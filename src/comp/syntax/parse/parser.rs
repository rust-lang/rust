
import std::io;
import std::vec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::either;
import std::either::left;
import std::either::right;
import std::map::hashmap;
import token::can_begin_expr;
import ex = ext::base;
import codemap::span;
import std::map::new_str_hash;
import util::interner;
import ast::node_id;
import ast::spanned;

tag restriction { UNRESTRICTED; RESTRICT_NO_CALL_EXPRS; }

tag file_type { CRATE_FILE; SOURCE_FILE; }

tag ty_or_bang { a_ty(@ast::ty); a_bang; }

type parse_sess = @{cm: codemap::codemap, mutable next_id: node_id};

fn next_node_id(sess: &parse_sess) -> node_id {
    let rv = sess.next_id;
    sess.next_id += 1;
    ret rv;
}

type parser =
    obj {
        fn peek() -> token::token;
        fn bump();
        fn swap(token::token, uint, uint);
        fn look_ahead(uint) -> token::token;
        fn fatal(str) -> ! ;
        fn warn(str);
        fn restrict(restriction);
        fn get_restriction() -> restriction;
        fn get_file_type() -> file_type;
        fn get_cfg() -> ast::crate_cfg;
        fn get_span() -> span;
        fn get_lo_pos() -> uint;
        fn get_hi_pos() -> uint;
        fn get_last_lo_pos() -> uint;
        fn get_last_hi_pos() -> uint;
        fn get_prec_table() -> @[op_spec];
        fn get_str(token::str_num) -> str;
        fn get_reader() -> lexer::reader;
        fn get_filemap() -> codemap::filemap;
        fn get_bad_expr_words() -> hashmap<str, ()>;
        fn get_chpos() -> uint;
        fn get_byte_pos() -> uint;
        fn get_id() -> node_id;
        fn get_sess() -> parse_sess;
    };

fn new_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg, path: str,
                        chpos: uint, byte_pos: uint, ftype: file_type) ->
   parser {
    let src = io::read_whole_file_str(path);
    let filemap = codemap::new_filemap(path, chpos, byte_pos);
    sess.cm.files += [filemap];
    let itr = @interner::mk(str::hash, str::eq);
    let rdr = lexer::new_reader(sess.cm, src, filemap, itr);

    ret new_parser(sess, cfg, rdr, ftype);
}

fn new_parser(sess: parse_sess, cfg: ast::crate_cfg, rdr: lexer::reader,
              ftype: file_type) -> parser {
    obj stdio_parser(sess: parse_sess,
                     cfg: ast::crate_cfg,
                     ftype: file_type,
                     mutable tok: token::token,
                     mutable tok_span: span,
                     mutable last_tok_span: span,
                     mutable buffer: [{tok: token::token, span: span}],
                     mutable restr: restriction,
                     rdr: lexer::reader,
                     precs: @[op_spec],
                     bad_words: hashmap<str, ()>) {
        fn peek() -> token::token { ret tok; }
        fn bump() {
            last_tok_span = tok_span;
            if vec::len(buffer) == 0u {
                let next = lexer::next_token(rdr);
                tok = next.tok;
                tok_span = ast::mk_sp(next.chpos, rdr.get_chpos());
            } else {
                let next = vec::pop(buffer);
                tok = next.tok;
                tok_span = next.span;
            }
        }
        fn swap(next: token::token, lo: uint, hi: uint) {
            tok = next;
            tok_span = ast::mk_sp(lo, hi);
        }
        fn look_ahead(distance: uint) -> token::token {
            while vec::len(buffer) < distance {
                let next = lexer::next_token(rdr);
                let sp = ast::mk_sp(next.chpos, rdr.get_chpos());
                buffer = [{tok: next.tok, span: sp}] + buffer;
            }
            ret buffer[distance - 1u].tok;
        }
        fn fatal(m: str) -> ! {
            codemap::emit_error(some(self.get_span()), m, sess.cm);
            fail;
        }
        fn warn(m: str) {
            codemap::emit_warning(some(self.get_span()), m, sess.cm);
        }
        fn restrict(r: restriction) { restr = r; }
        fn get_restriction() -> restriction { ret restr; }
        fn get_span() -> span { ret tok_span; }
        fn get_lo_pos() -> uint { ret tok_span.lo; }
        fn get_hi_pos() -> uint { ret tok_span.hi; }
        fn get_last_lo_pos() -> uint { ret last_tok_span.lo; }
        fn get_last_hi_pos() -> uint { ret last_tok_span.hi; }
        fn get_file_type() -> file_type { ret ftype; }
        fn get_cfg() -> ast::crate_cfg { ret cfg; }
        fn get_prec_table() -> @[op_spec] { ret precs; }
        fn get_str(i: token::str_num) -> str {
            ret interner::get(*rdr.get_interner(), i);
        }
        fn get_reader() -> lexer::reader { ret rdr; }
        fn get_filemap() -> codemap::filemap { ret rdr.get_filemap(); }
        fn get_bad_expr_words() -> hashmap<str, ()> { ret bad_words; }
        fn get_chpos() -> uint { ret rdr.get_chpos(); }
        fn get_byte_pos() -> uint { ret rdr.get_byte_pos(); }
        fn get_id() -> node_id { ret next_node_id(sess); }
        fn get_sess() -> parse_sess { ret sess; }
    }

    let tok0 = lexer::next_token(rdr);
    let span0 = ast::mk_sp(tok0.chpos, rdr.get_chpos());
    ret stdio_parser(sess, cfg, ftype, tok0.tok, span0, span0, [],
                     UNRESTRICTED, rdr, prec_table(), bad_expr_word_table());
}

// These are the words that shouldn't be allowed as value identifiers,
// because, if used at the start of a line, they will cause the line to be
// interpreted as a specific kind of statement, which would be confusing.
fn bad_expr_word_table() -> hashmap<str, ()> {
    let words = new_str_hash();
    words.insert("mod", ());
    words.insert("if", ());
    words.insert("else", ());
    words.insert("while", ());
    words.insert("do", ());
    words.insert("alt", ());
    words.insert("for", ());
    words.insert("each", ());
    words.insert("break", ());
    words.insert("cont", ());
    words.insert("put", ());
    words.insert("ret", ());
    words.insert("be", ());
    words.insert("fail", ());
    words.insert("type", ());
    words.insert("resource", ());
    words.insert("check", ());
    words.insert("assert", ());
    words.insert("claim", ());
    words.insert("prove", ());
    words.insert("native", ());
    words.insert("fn", ());
    words.insert("block", ());
    words.insert("lambda", ());
    words.insert("pred", ());
    words.insert("iter", ());
    words.insert("block", ());
    words.insert("import", ());
    words.insert("export", ());
    words.insert("let", ());
    words.insert("const", ());
    words.insert("log", ());
    words.insert("log_err", ());
    words.insert("tag", ());
    words.insert("obj", ());
    words.insert("copy", ());
    ret words;
}

fn unexpected(p: &parser, t: token::token) -> ! {
    let s: str = "unexpected token: ";
    s += token::to_str(p.get_reader(), t);
    p.fatal(s);
}

fn expect(p: &parser, t: token::token) {
    if p.peek() == t {
        p.bump();
    } else {
        let s: str = "expecting ";
        s += token::to_str(p.get_reader(), t);
        s += ", found ";
        s += token::to_str(p.get_reader(), p.peek());
        p.fatal(s);
    }
}

fn expect_gt(p: &parser) {
    if p.peek() == token::GT {
        p.bump();
    } else if p.peek() == token::BINOP(token::LSR) {
        p.swap(token::GT, p.get_lo_pos() + 1u, p.get_hi_pos());
    } else if p.peek() == token::BINOP(token::ASR) {
        p.swap(token::BINOP(token::LSR), p.get_lo_pos() + 1u, p.get_hi_pos());
    } else {
        let s: str = "expecting ";
        s += token::to_str(p.get_reader(), token::GT);
        s += ", found ";
        s += token::to_str(p.get_reader(), p.peek());
        p.fatal(s);
    }
}

fn spanned<T>(lo: uint, hi: uint, node: &T) -> spanned<T> {
    ret {node: node, span: ast::mk_sp(lo, hi)};
}

fn parse_ident(p: &parser) -> ast::ident {
    alt p.peek() {
      token::IDENT(i, _) { p.bump(); ret p.get_str(i); }
      _ { p.fatal("expecting ident"); }
    }
}

fn parse_value_ident(p: &parser) -> ast::ident {
    check_bad_word(p);
    ret parse_ident(p);
}

fn eat(p: &parser, tok: &token::token) -> bool {
    ret if p.peek() == tok { p.bump(); true } else { false };
}

fn is_word(p: &parser, word: &str) -> bool {
    ret alt p.peek() {
          token::IDENT(sid, false) { str::eq(word, p.get_str(sid)) }
          _ { false }
        };
}

fn eat_word(p: &parser, word: &str) -> bool {
    alt p.peek() {
      token::IDENT(sid, false) {
        if str::eq(word, p.get_str(sid)) {
            p.bump();
            ret true;
        } else { ret false; }
      }
      _ { ret false; }
    }
}

fn expect_word(p: &parser, word: &str) {
    if !eat_word(p, word) {
        p.fatal("expecting " + word + ", found " +
                    token::to_str(p.get_reader(), p.peek()));
    }
}

fn check_bad_word(p: &parser) {
    alt p.peek() {
      token::IDENT(sid, false) {
        let w = p.get_str(sid);
        if p.get_bad_expr_words().contains_key(w) {
            p.fatal("found " + w + " in expression position");
        }
      }
      _ { }
    }
}

fn parse_ty_fn(proto: ast::proto, p: &parser) -> ast::ty_ {
    fn parse_fn_input_ty(p: &parser) -> ast::ty_arg {
        let lo = p.get_lo_pos();
        // Ignore arg name, if present
        if is_plain_ident(p) && p.look_ahead(1u) == token::COLON {
            p.bump();
            p.bump();
        }
        let mode = ast::val;
        if p.peek() == token::BINOP(token::AND) {
            p.bump();
            mode = ast::alias(eat_word(p, "mutable"));
        }
        let t = parse_ty(p, false);
        ret spanned(lo, t.span.hi, {mode: mode, ty: t});
    }
    let lo = p.get_lo_pos();
    let inputs =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                  parse_fn_input_ty, p);
    // FIXME: there's no syntax for this right now anyway
    //  auto constrs = parse_constrs(~[], p);
    let constrs: [@ast::constr] = [];
    let output: @ast::ty;
    let cf = ast::return;
    if p.peek() == token::RARROW {
        p.bump();
        let tmp = parse_ty_or_bang(p);
        alt tmp {
          a_ty(t) { output = t; }
          a_bang. {
            output = @spanned(lo, inputs.span.hi, ast::ty_bot);
            cf = ast::noreturn;
          }
        }
    } else { output = @spanned(lo, inputs.span.hi, ast::ty_nil); }
    ret ast::ty_fn(proto, inputs.node, output, cf, constrs);
}

fn parse_proto(p: &parser) -> ast::proto {
    if eat_word(p, "iter") {
        ret ast::proto_iter;
    } else if eat_word(p, "fn") {
        ret ast::proto_fn;
    } else if eat_word(p, "block") {
        ret ast::proto_block;
    } else if eat_word(p, "pred") {
        ret ast::proto_fn;
    } else { unexpected(p, p.peek()); }
}

fn parse_ty_obj(p: &parser, hi: &mutable uint) -> ast::ty_ {
    fn parse_method_sig(p: &parser) -> ast::ty_method {
        let flo = p.get_lo_pos();
        let proto: ast::proto = parse_proto(p);
        let ident = parse_value_ident(p);
        let f = parse_ty_fn(proto, p);
        expect(p, token::SEMI);
        alt f {
          ast::ty_fn(proto, inputs, output, cf, constrs) {
            ret spanned(flo, output.span.hi,
                        {proto: proto,
                         ident: ident,
                         inputs: inputs,
                         output: output,
                         cf: cf,
                         constrs: constrs});
          }
        }
    }
    let meths =
        parse_seq(token::LBRACE, token::RBRACE, none, parse_method_sig, p);
    hi = meths.span.hi;
    ret ast::ty_obj(meths.node);
}

fn parse_mt(p: &parser) -> ast::mt {
    let mut = parse_mutability(p);
    let t = parse_ty(p, false);
    ret {ty: t, mut: mut};
}

fn parse_ty_field(p: &parser) -> ast::ty_field {
    let lo = p.get_lo_pos();
    let mut = parse_mutability(p);
    let id = parse_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    ret spanned(lo, ty.span.hi, {ident: id, mt: {ty: ty, mut: mut}});
}

// if i is the jth ident in args, return j
// otherwise, fail
fn ident_index(p: &parser, args: &[ast::arg], i: &ast::ident) -> uint {
    let j = 0u;
    for a: ast::arg in args { if a.ident == i { ret j; } j += 1u; }
    p.fatal("Unbound variable " + i + " in constraint arg");
}

fn parse_type_constr_arg(p: &parser) -> @ast::ty_constr_arg {
    let sp = p.get_span();
    let carg = ast::carg_base;
    expect(p, token::BINOP(token::STAR));
    if p.peek() == token::DOT {
        // "*..." notation for record fields
        p.bump();
        let pth: ast::path = parse_path(p);
        carg = ast::carg_ident(pth);
    }
    // No literals yet, I guess?
    ret @{node: carg, span: sp};
}

fn parse_constr_arg(args: &[ast::arg], p: &parser) -> @ast::constr_arg {
    let sp = p.get_span();
    let carg = ast::carg_base;
    if p.peek() == token::BINOP(token::STAR) {
        p.bump();
    } else {
        let i: ast::ident = parse_value_ident(p);
        carg = ast::carg_ident(ident_index(p, args, i));
    }
    ret @{node: carg, span: sp};
}

fn parse_ty_constr(fn_args: &[ast::arg], p: &parser) -> @ast::constr {
    let lo = p.get_lo_pos();
    let path = parse_path(p);
    let pf = bind parse_constr_arg(fn_args, _);
    let args: {node: [@ast::constr_arg], span: span} =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA), pf, p);
    ret @spanned(lo, args.span.hi,
                 {path: path, args: args.node, id: p.get_id()});
}

fn parse_constr_in_type(p: &parser) -> @ast::ty_constr {
    let lo = p.get_lo_pos();
    let path = parse_path(p);
    let args: [@ast::ty_constr_arg] =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                  parse_type_constr_arg, p).node;
    let hi = p.get_lo_pos();
    let tc: ast::ty_constr_ = {path: path, args: args, id: p.get_id()};
    ret @spanned(lo, hi, tc);
}


fn parse_constrs<T>(pser: fn(&parser) -> @ast::constr_general<T>, p: &parser)
   -> [@ast::constr_general<T>] {
    let constrs: [@ast::constr_general<T>] = [];
    while true {
        let constr = pser(p);
        constrs += [constr];
        if p.peek() == token::COMMA { p.bump(); } else { break; }
    }
    constrs
}

fn parse_type_constraints(p: &parser) -> [@ast::ty_constr] {
    ret parse_constrs(parse_constr_in_type, p);
}

fn parse_ty_postfix(orig_t: ast::ty_, p: &parser, colons_before_params: bool)
   -> @ast::ty {
    let lo = p.get_lo_pos();

    if colons_before_params && p.peek() == token::MOD_SEP {
        p.bump();
        expect(p, token::LT);
    } else if !colons_before_params && p.peek() == token::LT {
        p.bump();
    } else { ret @spanned(lo, p.get_lo_pos(), orig_t); }

    // If we're here, we have explicit type parameter instantiation.
    let seq = parse_seq_to_gt(some(token::COMMA), bind parse_ty(_, false), p);

    alt orig_t {
      ast::ty_path(pth, ann) {
        let hi = p.get_hi_pos();
        ret @spanned(lo, hi,
                     ast::ty_path(spanned(lo, hi,
                                          {global: pth.node.global,
                                           idents: pth.node.idents,
                                           types: seq}), ann));
      }
      _ { p.fatal("type parameter instantiation only allowed for paths"); }
    }
}

fn parse_ty_or_bang(p: &parser) -> ty_or_bang {
    alt p.peek() {
      token::NOT. { p.bump(); ret a_bang; }
      _ { ret a_ty(parse_ty(p, false)); }
    }
}

fn parse_ty(p: &parser, colons_before_params: bool) -> @ast::ty {
    let lo = p.get_lo_pos();
    let hi = lo;
    let t: ast::ty_;
    // FIXME: do something with this

    if eat_word(p, "bool") {
        t = ast::ty_bool;
    } else if eat_word(p, "int") {
        t = ast::ty_int;
    } else if eat_word(p, "uint") {
        t = ast::ty_uint;
    } else if eat_word(p, "float") {
        t = ast::ty_float;
    } else if eat_word(p, "str") {
        t = ast::ty_str;
    } else if eat_word(p, "istr") {
        t = ast::ty_istr;
    } else if eat_word(p, "char") {
        t = ast::ty_char;
        /*
            } else if (eat_word(p, "task")) {
                t = ast::ty_task;
        */
    } else if eat_word(p, "i8") {
        t = ast::ty_machine(ast::ty_i8);
    } else if eat_word(p, "i16") {
        t = ast::ty_machine(ast::ty_i16);
    } else if eat_word(p, "i32") {
        t = ast::ty_machine(ast::ty_i32);
    } else if eat_word(p, "i64") {
        t = ast::ty_machine(ast::ty_i64);
    } else if eat_word(p, "u8") {
        t = ast::ty_machine(ast::ty_u8);
    } else if eat_word(p, "u16") {
        t = ast::ty_machine(ast::ty_u16);
    } else if eat_word(p, "u32") {
        t = ast::ty_machine(ast::ty_u32);
    } else if eat_word(p, "u64") {
        t = ast::ty_machine(ast::ty_u64);
    } else if eat_word(p, "f32") {
        t = ast::ty_machine(ast::ty_f32);
    } else if eat_word(p, "f64") {
        t = ast::ty_machine(ast::ty_f64);
    } else if p.peek() == token::LPAREN {
        p.bump();
        if p.peek() == token::RPAREN {
            hi = p.get_hi_pos();
            p.bump();
            t = ast::ty_nil;
        } else {
            let ts = [parse_ty(p, false)];
            while p.peek() == token::COMMA {
                p.bump();
                ts += [parse_ty(p, false)];
            }
            if vec::len(ts) == 1u {
                t = ts[0].node;
            } else { t = ast::ty_tup(ts); }
            hi = p.get_hi_pos();
            expect(p, token::RPAREN);
        }
    } else if p.peek() == token::AT {
        p.bump();
        let mt = parse_mt(p);
        hi = mt.ty.span.hi;
        t = ast::ty_box(mt);
    } else if p.peek() == token::BINOP(token::STAR) {
        p.bump();
        let mt = parse_mt(p);
        hi = mt.ty.span.hi;
        t = ast::ty_ptr(mt);
    } else if p.peek() == token::LBRACE {
        let elems =
            parse_seq(token::LBRACE, token::RBRACE, some(token::COMMA),
                      parse_ty_field, p);
        hi = elems.span.hi;
        t = ast::ty_rec(elems.node);
        if p.peek() == token::COLON {
            p.bump();
            t =
                ast::ty_constr(@spanned(lo, hi, t),
                               parse_type_constraints(p));
        }
    } else if p.peek() == token::LBRACKET {
        expect(p, token::LBRACKET);
        t = ast::ty_vec(parse_mt(p));
        hi = p.get_hi_pos();
        expect(p, token::RBRACKET);
    } else if eat_word(p, "fn") {
        t = parse_ty_fn(ast::proto_fn, p);
        alt t { ast::ty_fn(_, _, out, _, _) { hi = out.span.hi; } }
    } else if eat_word(p, "block") {
        t = parse_ty_fn(ast::proto_block, p);
        alt t { ast::ty_fn(_, _, out, _, _) { hi = out.span.hi; } }
    } else if eat_word(p, "iter") {
        t = parse_ty_fn(ast::proto_iter, p);
        alt t { ast::ty_fn(_, _, out, _, _) { hi = out.span.hi; } }
    } else if eat_word(p, "obj") {
        t = parse_ty_obj(p, hi);
    } else if eat_word(p, "mutable") {
        p.warn("ignoring deprecated 'mutable' type constructor");
        let typ = parse_ty(p, false);
        t = typ.node;
        hi = typ.span.hi;
    } else if p.peek() == token::MOD_SEP || is_ident(p.peek()) {
        let path = parse_path(p);
        t = ast::ty_path(path, p.get_id());
        hi = path.span.hi;
    } else { p.fatal("expecting type"); }
    ret parse_ty_postfix(t, p, colons_before_params);
}

fn parse_arg_mode(p: &parser) -> ast::mode {
    if eat(p, token::BINOP(token::AND)) {
        ast::alias(eat_word(p, "mutable"))
    } else if eat(p, token::BINOP(token::MINUS)) {
        ast::move
    } else { ast::val }
}

fn parse_arg(p: &parser) -> ast::arg {
    let i = parse_value_ident(p);
    expect(p, token::COLON);
    let m = parse_arg_mode(p);
    let t = parse_ty(p, false);
    ret {mode: m, ty: t, ident: i, id: p.get_id()};
}

fn parse_fn_block_arg(p: &parser) -> ast::arg {
    let m = parse_arg_mode(p);
    let i = parse_value_ident(p);
    let t = @spanned(p.get_lo_pos(), p.get_hi_pos(), ast::ty_infer);
    ret {mode: m, ty: t, ident: i, id: p.get_id()};
}

fn parse_seq_to_before_gt<T>(sep: option::t<token::token>,
                             f: fn(&parser) -> T, p: &parser) -> [T] {
    let first = true;
    let v = [];
    while p.peek() != token::GT && p.peek() != token::BINOP(token::LSR) &&
              p.peek() != token::BINOP(token::ASR) {
        alt sep {
          some(t) { if first { first = false; } else { expect(p, t); } }
          _ { }
        }
        v += [f(p)];
    }

    ret v;
}

fn parse_seq_to_gt<T>(sep: option::t<token::token>, f: fn(&parser) -> T,
                      p: &parser) -> [T] {
    let v = parse_seq_to_before_gt(sep, f, p);
    expect_gt(p);

    ret v;
}

fn parse_seq_lt_gt<T>(sep: option::t<token::token>, f: fn(&parser) -> T,
                      p: &parser) -> spanned<[T]> {
    let lo = p.get_lo_pos();
    expect(p, token::LT);
    let result = parse_seq_to_before_gt::<T>(sep, f, p);
    let hi = p.get_hi_pos();
    expect_gt(p);
    ret spanned(lo, hi, result);
}

fn parse_seq_to_end<T>(ket: token::token, sep: option::t<token::token>,
                       f: fn(&parser) -> T, p: &parser) -> [T] {
    let val = parse_seq_to_before_end(ket, sep, f, p);
    p.bump();
    ret val;
}

fn parse_seq_to_before_end<T>(ket: token::token, sep: option::t<token::token>,
                              f: fn(&parser) -> T, p: &parser) -> [T] {
    let first: bool = true;
    let v: [T] = [];
    while p.peek() != ket {
        alt sep {
          some(t) { if first { first = false; } else { expect(p, t); } }
          _ { }
        }
        v += [f(p)];
    }
    ret v;
}


fn parse_seq<T>(bra: token::token, ket: token::token,
                sep: option::t<token::token>, f: fn(&parser) -> T, p: &parser)
   -> spanned<[T]> {
    let lo = p.get_lo_pos();
    expect(p, bra);
    let result = parse_seq_to_before_end::<T>(ket, sep, f, p);
    let hi = p.get_hi_pos();
    p.bump();
    ret spanned(lo, hi, result);
}


fn parse_lit(p: &parser) -> ast::lit {
    let sp = p.get_span();
    let lit: ast::lit_ = ast::lit_nil;
    if eat_word(p, "true") {
        lit = ast::lit_bool(true);
    } else if eat_word(p, "false") {
        lit = ast::lit_bool(false);
    } else {
        alt p.peek() {
          token::LIT_INT(i) { p.bump(); lit = ast::lit_int(i); }
          token::LIT_UINT(u) { p.bump(); lit = ast::lit_uint(u); }
          token::LIT_FLOAT(s) {
            p.bump();
            lit = ast::lit_float(p.get_str(s));
          }
          token::LIT_MACH_INT(tm, i) {
            p.bump();
            lit = ast::lit_mach_int(tm, i);
          }
          token::LIT_MACH_FLOAT(tm, s) {
            p.bump();
            lit = ast::lit_mach_float(tm, p.get_str(s));
          }
          token::LIT_CHAR(c) { p.bump(); lit = ast::lit_char(c); }
          token::LIT_STR(s) {
            p.bump();
            lit = ast::lit_str(p.get_str(s), ast::sk_rc);
          }
          token::LPAREN. {
            p.bump();
            expect(p, token::RPAREN);
            lit = ast::lit_nil;
          }
          t { unexpected(p, t); }
        }
    }
    ret {node: lit, span: sp};
}

fn is_ident(t: token::token) -> bool {
    alt t { token::IDENT(_, _) { ret true; } _ { } }
    ret false;
}

fn is_plain_ident(p: &parser) -> bool {
    ret alt p.peek() { token::IDENT(_, false) { true } _ { false } };
}

fn parse_path(p: &parser) -> ast::path {
    let lo = p.get_lo_pos();
    let hi = lo;

    let global;
    if p.peek() == token::MOD_SEP {
        global = true;
        p.bump();
    } else { global = false; }

    let ids: [ast::ident] = [];
    while true {
        alt p.peek() {
          token::IDENT(i, _) {
            hi = p.get_hi_pos();
            ids += [p.get_str(i)];
            hi = p.get_hi_pos();
            p.bump();
            if p.peek() == token::MOD_SEP && p.look_ahead(1u) != token::LT {
                p.bump();
            } else { break; }
          }
          _ { break; }
        }
    }
    ret spanned(lo, hi, {global: global, idents: ids, types: []});
}

fn parse_path_and_ty_param_substs(p: &parser) -> ast::path {
    let lo = p.get_lo_pos();
    let path = parse_path(p);
    if p.peek() == token::MOD_SEP {
        p.bump();

        let seq =
            parse_seq_lt_gt(some(token::COMMA), bind parse_ty(_, false), p);
        let hi = seq.span.hi;
        path =
            spanned(lo, hi,
                    {global: path.node.global,
                     idents: path.node.idents,
                     types: seq.node});
    }
    ret path;
}

fn parse_mutability(p: &parser) -> ast::mutability {
    if eat_word(p, "mutable") {
        if p.peek() == token::QUES { p.bump(); ret ast::maybe_mut; }
        ret ast::mut;
    }
    ret ast::imm;
}

fn parse_field(p: &parser, sep: &token::token) -> ast::field {
    let lo = p.get_lo_pos();
    let m = parse_mutability(p);
    let i = parse_ident(p);
    expect(p, sep);
    let e = parse_expr(p);
    ret spanned(lo, e.span.hi, {mut: m, ident: i, expr: e});
}

fn mk_expr(p: &parser, lo: uint, hi: uint, node: &ast::expr_) -> @ast::expr {
    ret @{id: p.get_id(), node: node, span: ast::mk_sp(lo, hi)};
}

fn mk_mac_expr(p: &parser, lo: uint, hi: uint, m: &ast::mac_) -> @ast::expr {
    ret @{id: p.get_id(),
          node: ast::expr_mac({node: m, span: ast::mk_sp(lo, hi)}),
          span: ast::mk_sp(lo, hi)};
}

fn parse_bottom_expr(p: &parser) -> @ast::expr {
    let lo = p.get_lo_pos();
    let hi = p.get_hi_pos();

    let ex: ast::expr_;
    if p.peek() == token::LPAREN {
        p.bump();
        if p.peek() == token::RPAREN {
            hi = p.get_hi_pos();
            p.bump();
            let lit = @spanned(lo, hi, ast::lit_nil);
            ret mk_expr(p, lo, hi, ast::expr_lit(lit));
        }
        let es = [parse_expr(p)];
        while p.peek() == token::COMMA { p.bump(); es += [parse_expr(p)]; }
        hi = p.get_hi_pos();
        expect(p, token::RPAREN);
        if vec::len(es) == 1u {
            ret mk_expr(p, lo, hi, es[0].node);
        } else { ret mk_expr(p, lo, hi, ast::expr_tup(es)); }
    } else if p.peek() == token::LBRACE {
        p.bump();
        if is_word(p, "mutable") ||
               is_plain_ident(p) && p.look_ahead(1u) == token::COLON {
            let fields = [parse_field(p, token::COLON)];
            let base = none;
            while p.peek() != token::RBRACE {
                if eat_word(p, "with") { base = some(parse_expr(p)); break; }
                expect(p, token::COMMA);
                fields += [parse_field(p, token::COLON)];
            }
            hi = p.get_hi_pos();
            expect(p, token::RBRACE);
            ex = ast::expr_rec(fields, base);
        } else if p.peek() == token::BINOP(token::OR) {
            ret parse_fn_block_expr(p);
        } else {
            let blk = parse_block_tail(p, lo);
            ret mk_expr(p, blk.span.lo, blk.span.hi, ast::expr_block(blk));
        }
    } else if eat_word(p, "if") {
        ret parse_if_expr(p);
    } else if eat_word(p, "for") {
        ret parse_for_expr(p);
    } else if eat_word(p, "while") {
        ret parse_while_expr(p);
    } else if eat_word(p, "do") {
        ret parse_do_while_expr(p);
    } else if eat_word(p, "alt") {
        ret parse_alt_expr(p);
        /*
            } else if (eat_word(p, "spawn")) {
                ret parse_spawn_expr(p);
        */
    } else if eat_word(p, "fn") {
        ret parse_fn_expr(p, ast::proto_fn);
    } else if eat_word(p, "block") {
        ret parse_fn_expr(p, ast::proto_block);
    } else if eat_word(p, "lambda") {
        ret parse_fn_expr(p, ast::proto_closure);
    } else if p.peek() == token::LBRACKET {
        p.bump();
        let mut = parse_mutability(p);
        let es =
            parse_seq_to_end(token::RBRACKET, some(token::COMMA), parse_expr,
                             p);
        ex = ast::expr_vec(es, mut);
    } else if p.peek() == token::POUND_LT {
        p.bump();
        let ty = parse_ty(p, false);
        expect(p, token::GT);

        /* hack: early return to take advantage of specialized function */
        ret mk_mac_expr(p, lo, p.get_hi_pos(), ast::mac_embed_type(ty))
    } else if p.peek() == token::POUND_LBRACE {
        p.bump();
        let blk = ast::mac_embed_block(parse_block_tail(p, lo));
        ret mk_mac_expr(p, lo, p.get_hi_pos(), blk);
    } else if p.peek() == token::ELLIPSIS {
        p.bump();
        ret mk_mac_expr(p, lo, p.get_hi_pos(), ast::mac_ellipsis)
    } else if p.peek() == token::TILDE {
        p.bump();
        alt p.peek() {
          token::LIT_STR(s) {
            p.bump();
            let lit =
                @{node: ast::lit_str(p.get_str(s), ast::sk_unique),
                  span: p.get_span()};
            ex = ast::expr_lit(lit);
          }
          _ { ex = ast::expr_uniq(parse_expr(p)); }
        }
    } else if eat_word(p, "obj") {
        // Anonymous object

        // Only make people type () if they're actually adding new fields
        let fields: option::t<[ast::anon_obj_field]> = none;
        if p.peek() == token::LPAREN {
            p.bump();
            fields =
                some(parse_seq_to_end(token::RPAREN, some(token::COMMA),
                                      parse_anon_obj_field, p));
        }
        let meths: [@ast::method] = [];
        let inner_obj: option::t<@ast::expr> = none;
        expect(p, token::LBRACE);
        while p.peek() != token::RBRACE {
            if eat_word(p, "with") {
                inner_obj = some(parse_expr(p));
            } else { meths += [parse_method(p)]; }
        }
        hi = p.get_hi_pos();
        expect(p, token::RBRACE);
        // fields and methods may be *additional* or *overriding* fields
        // and methods if there's a inner_obj, or they may be the *only*
        // fields and methods if there's no inner_obj.

        // We don't need to pull ".node" out of fields because it's not a
        // "spanned".
        let ob = {fields: fields, methods: meths, inner_obj: inner_obj};
        ex = ast::expr_anon_obj(ob);
    } else if eat_word(p, "bind") {
        let e = parse_expr_res(p, RESTRICT_NO_CALL_EXPRS);
        fn parse_expr_opt(p: &parser) -> option::t<@ast::expr> {
            alt p.peek() {
              token::UNDERSCORE. { p.bump(); ret none; }
              _ { ret some(parse_expr(p)); }
            }
        }
        let es =
            parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                      parse_expr_opt, p);
        hi = es.span.hi;
        ex = ast::expr_bind(e, es.node);
    } else if p.peek() == token::POUND {
        let ex_ext = parse_syntax_ext(p);
        hi = ex_ext.span.hi;
        ex = ex_ext.node;
    } else if eat_word(p, "fail") {
        if can_begin_expr(p.peek()) {
            let e = parse_expr(p);
            hi = e.span.hi;
            ex = ast::expr_fail(some(e));
        } else { ex = ast::expr_fail(none); }
    } else if eat_word(p, "log") {
        let e = parse_expr(p);
        ex = ast::expr_log(1, e);
        hi = e.span.hi;
    } else if eat_word(p, "log_err") {
        let e = parse_expr(p);
        ex = ast::expr_log(0, e);
        hi = e.span.hi;
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
        ex = ast::expr_check(ast::checked, e);
    } else if eat_word(p, "claim") {
        /* Same rules as check, except that if check-claims
         is enabled (a command-line flag), then the parser turns
        claims into check */

        let e = parse_expr(p);
        hi = e.span.hi;
        ex = ast::expr_check(ast::unchecked, e);
    } else if eat_word(p, "ret") {
        if can_begin_expr(p.peek()) {
            let e = parse_expr(p);
            hi = e.span.hi;
            ex = ast::expr_ret(some(e));
        } else { ex = ast::expr_ret(none); }
    } else if eat_word(p, "break") {
        ex = ast::expr_break;
        hi = p.get_hi_pos();
    } else if eat_word(p, "cont") {
        ex = ast::expr_cont;
        hi = p.get_hi_pos();
    } else if eat_word(p, "put") {
        alt p.peek() {
          token::SEMI. { ex = ast::expr_put(none); }
          _ {
            let e = parse_expr(p);
            hi = e.span.hi;
            ex = ast::expr_put(some(e));
          }
        }
    } else if eat_word(p, "be") {
        let e = parse_expr(p);

        // FIXME: Is this the right place for this check?
        if /*check*/ast::is_call_expr(e) {
            hi = e.span.hi;
            ex = ast::expr_be(e);
        } else { p.fatal("Non-call expression in tail call"); }
    } else if eat_word(p, "copy") {
        let e = parse_expr(p);
        ex = ast::expr_copy(e);
        hi = e.span.hi;
    } else if eat_word(p, "self") {
        expect(p, token::DOT);
        // The rest is a call expression.
        let f: @ast::expr = parse_self_method(p);
        let es =
            parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                      parse_expr, p);
        hi = es.span.hi;
        ex = ast::expr_call(f, es.node);
    } else if p.peek() == token::MOD_SEP ||
                  is_ident(p.peek()) && !is_word(p, "true") &&
                      !is_word(p, "false") {
        check_bad_word(p);
        let pth = parse_path_and_ty_param_substs(p);
        hi = pth.span.hi;
        ex = ast::expr_path(pth);
    } else {
        let lit = parse_lit(p);
        hi = lit.span.hi;
        ex = ast::expr_lit(@lit);
    }
    ret mk_expr(p, lo, hi, ex);
}

fn parse_syntax_ext(p: &parser) -> @ast::expr {
    let lo = p.get_lo_pos();
    expect(p, token::POUND);
    ret parse_syntax_ext_naked(p, lo);
}

fn parse_syntax_ext_naked(p: &parser, lo: uint) -> @ast::expr {
    let pth = parse_path(p);
    if vec::len(pth.node.idents) == 0u {
        p.fatal("expected a syntax expander name");
    }
    //temporary for a backwards-compatible cycle:
    let es =
        if p.peek() == token::LPAREN {
            parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                      parse_expr, p)
        } else {
            parse_seq(token::LBRACKET, token::RBRACKET, some(token::COMMA),
                      parse_expr, p)
        };
    let hi = es.span.hi;
    let e = mk_expr(p, es.span.lo, hi, ast::expr_vec(es.node, ast::imm));
    ret mk_mac_expr(p, lo, hi, ast::mac_invoc(pth, e, none));
}

fn parse_self_method(p: &parser) -> @ast::expr {
    let sp = p.get_span();
    let f_name: ast::ident = parse_ident(p);
    ret mk_expr(p, sp.lo, sp.hi, ast::expr_self_method(f_name));
}

fn parse_dot_or_call_expr(p: &parser) -> @ast::expr {
    ret parse_dot_or_call_expr_with(p, parse_bottom_expr(p));
}

fn parse_dot_or_call_expr_with(p: &parser, e: @ast::expr) -> @ast::expr {
    let lo = e.span.lo;
    let hi = e.span.hi;
    while true {
        alt p.peek() {
          token::LPAREN. {
            if p.get_restriction() == RESTRICT_NO_CALL_EXPRS {
                ret e;
            } else {
                // Call expr.

                let es =
                    parse_seq(token::LPAREN, token::RPAREN,
                              some(token::COMMA), parse_expr, p);
                hi = es.span.hi;
                let nd = ast::expr_call(e, es.node);
                e = mk_expr(p, lo, hi, nd);
            }
          }
          token::LBRACKET. {
            p.bump();
            let ix = parse_expr(p);
            hi = ix.span.hi;
            expect(p, token::RBRACKET);
            e = mk_expr(p, lo, hi, ast::expr_index(e, ix));
          }
          token::DOT. {
            p.bump();
            alt p.peek() {
              token::IDENT(i, _) {
                hi = p.get_hi_pos();
                p.bump();
                e = mk_expr(p, lo, hi, ast::expr_field(e, p.get_str(i)));
              }
              t { unexpected(p, t); }
            }
          }
          _ { ret e; }
        }
    }
    ret e;
}

fn parse_prefix_expr(p: &parser) -> @ast::expr {
    if eat_word(p, "mutable") {
        p.warn("ignoring deprecated 'mutable' prefix operator");
    }
    let lo = p.get_lo_pos();
    let hi = p.get_hi_pos();
    // FIXME: can only remove this sort of thing when both typestate and
    // alt-exhaustive-match checking are co-operating.

    let lit = @spanned(lo, lo, ast::lit_nil);
    let ex: ast::expr_ = ast::expr_lit(lit);
    alt p.peek() {
      token::NOT. {
        p.bump();
        let e = parse_prefix_expr(p);
        hi = e.span.hi;
        ex = ast::expr_unary(ast::not, e);
      }
      token::BINOP(b) {
        alt b {
          token::MINUS. {
            p.bump();
            let e = parse_prefix_expr(p);
            hi = e.span.hi;
            ex = ast::expr_unary(ast::neg, e);
          }
          token::STAR. {
            p.bump();
            let e = parse_prefix_expr(p);
            hi = e.span.hi;
            ex = ast::expr_unary(ast::deref, e);
          }
          _ { ret parse_dot_or_call_expr(p); }
        }
      }
      token::AT. {
        p.bump();
        let m = parse_mutability(p);
        let e = parse_prefix_expr(p);
        hi = e.span.hi;
        ex = ast::expr_unary(ast::box(m), e);
      }
      _ { ret parse_dot_or_call_expr(p); }
    }
    ret mk_expr(p, lo, hi, ex);
}

fn parse_ternary(p: &parser) -> @ast::expr {
    let cond_expr = parse_binops(p);
    if p.peek() == token::QUES {
        p.bump();
        let then_expr = parse_expr(p);
        expect(p, token::COLON);
        let else_expr = parse_expr(p);
        ret mk_expr(p, cond_expr.span.lo, else_expr.span.hi,
                    ast::expr_ternary(cond_expr, then_expr, else_expr));
    } else { ret cond_expr; }
}

type op_spec = {tok: token::token, op: ast::binop, prec: int};


// FIXME make this a const, don't store it in parser state
fn prec_table() -> @[op_spec] {
    ret @[{tok: token::BINOP(token::STAR), op: ast::mul, prec: 11},
          {tok: token::BINOP(token::SLASH), op: ast::div, prec: 11},
          {tok: token::BINOP(token::PERCENT), op: ast::rem, prec: 11},
          {tok: token::BINOP(token::PLUS), op: ast::add, prec: 10},
          {tok: token::BINOP(token::MINUS), op: ast::sub, prec: 10},
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

fn parse_binops(p: &parser) -> @ast::expr {
    ret parse_more_binops(p, parse_prefix_expr(p), 0);
}

const unop_prec: int = 100;

const as_prec: int = 5;
const ternary_prec: int = 0;

fn parse_more_binops(p: &parser, lhs: @ast::expr, min_prec: int) ->
   @ast::expr {
    let peeked = p.peek();
    for cur: op_spec in *p.get_prec_table() {
        if cur.prec > min_prec && cur.tok == peeked {
            p.bump();
            let rhs = parse_more_binops(p, parse_prefix_expr(p), cur.prec);
            let bin =
                mk_expr(p, lhs.span.lo, rhs.span.hi,
                        ast::expr_binary(cur.op, lhs, rhs));
            ret parse_more_binops(p, bin, min_prec);
        }
    }
    if as_prec > min_prec && eat_word(p, "as") {
        let rhs = parse_ty(p, true);
        let _as =
            mk_expr(p, lhs.span.lo, rhs.span.hi, ast::expr_cast(lhs, rhs));
        ret parse_more_binops(p, _as, min_prec);
    }
    ret lhs;
}

fn parse_assign_expr(p: &parser) -> @ast::expr {
    let lo = p.get_lo_pos();
    let lhs = parse_ternary(p);
    alt p.peek() {
      token::EQ. {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_assign(lhs, rhs));
      }
      token::BINOPEQ(op) {
        p.bump();
        let rhs = parse_expr(p);
        let aop = ast::add;
        alt op {
          token::PLUS. { aop = ast::add; }
          token::MINUS. { aop = ast::sub; }
          token::STAR. { aop = ast::mul; }
          token::SLASH. { aop = ast::div; }
          token::PERCENT. { aop = ast::rem; }
          token::CARET. { aop = ast::bitxor; }
          token::AND. { aop = ast::bitand; }
          token::OR. { aop = ast::bitor; }
          token::LSL. { aop = ast::lsl; }
          token::LSR. { aop = ast::lsr; }
          token::ASR. { aop = ast::asr; }
        }
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_assign_op(aop, lhs, rhs));
      }
      token::LARROW. {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_move(lhs, rhs));
      }
      token::DARROW. {
        p.bump();
        let rhs = parse_expr(p);
        ret mk_expr(p, lo, rhs.span.hi, ast::expr_swap(lhs, rhs));
      }
      _ {/* fall through */ }
    }
    ret lhs;
}

fn parse_if_expr_1(p: &parser) ->
   {cond: @ast::expr,
    then: ast::blk,
    els: option::t<@ast::expr>,
    lo: uint,
    hi: uint} {
    let lo = p.get_last_lo_pos();
    let cond = parse_expr(p);
    let thn = parse_block(p);
    let els: option::t<@ast::expr> = none;
    let hi = thn.span.hi;
    if eat_word(p, "else") {
        let elexpr = parse_else_expr(p);
        els = some(elexpr);
        hi = elexpr.span.hi;
    }
    ret {cond: cond, then: thn, els: els, lo: lo, hi: hi};
}

fn parse_if_expr(p: &parser) -> @ast::expr {
    if eat_word(p, "check") {
        let q = parse_if_expr_1(p);
        ret mk_expr(p, q.lo, q.hi, ast::expr_if_check(q.cond, q.then, q.els));
    } else {
        let q = parse_if_expr_1(p);
        ret mk_expr(p, q.lo, q.hi, ast::expr_if(q.cond, q.then, q.els));
    }
}

fn parse_fn_expr(p: &parser, proto: ast::proto) -> @ast::expr {
    let lo = p.get_last_lo_pos();
    let decl = parse_fn_decl(p, ast::impure_fn, ast::il_normal);
    let body = parse_block(p);
    let _fn = {decl: decl, proto: proto, body: body};
    ret mk_expr(p, lo, body.span.hi, ast::expr_fn(_fn));
}

fn parse_fn_block_expr(p: &parser) -> @ast::expr {
    let lo = p.get_last_lo_pos();
    let decl = parse_fn_block_decl(p);
    let body = parse_block_tail(p, lo);
    let _fn = {decl: decl, proto: ast::proto_block, body: body};
    ret mk_expr(p, lo, body.span.hi, ast::expr_fn(_fn));
}

fn parse_else_expr(p: &parser) -> @ast::expr {
    if eat_word(p, "if") {
        ret parse_if_expr(p);
    } else {
        let blk = parse_block(p);
        ret mk_expr(p, blk.span.lo, blk.span.hi, ast::expr_block(blk));
    }
}

fn parse_for_expr(p: &parser) -> @ast::expr {
    let lo = p.get_last_lo_pos();
    let is_each = eat_word(p, "each");
    let decl = parse_local(p, false);
    expect_word(p, "in");
    let seq = parse_expr(p);
    let body = parse_block(p);
    let hi = body.span.hi;
    if is_each {
        ret mk_expr(p, lo, hi, ast::expr_for_each(decl, seq, body));
    } else { ret mk_expr(p, lo, hi, ast::expr_for(decl, seq, body)); }
}

fn parse_while_expr(p: &parser) -> @ast::expr {
    let lo = p.get_last_lo_pos();
    let cond = parse_expr(p);
    let body = parse_block(p);
    let hi = body.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_while(cond, body));
}

fn parse_do_while_expr(p: &parser) -> @ast::expr {
    let lo = p.get_last_lo_pos();
    let body = parse_block(p);
    expect_word(p, "while");
    let cond = parse_expr(p);
    let hi = cond.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_do_while(body, cond));
}

fn parse_alt_expr(p: &parser) -> @ast::expr {
    let lo = p.get_last_lo_pos();
    let discriminant = parse_expr(p);
    expect(p, token::LBRACE);
    let arms: [ast::arm] = [];
    while p.peek() != token::RBRACE {
        let pats = parse_pats(p);
        let blk = parse_block(p);
        arms += [{pats: pats, body: blk}];
    }
    let hi = p.get_hi_pos();
    p.bump();
    ret mk_expr(p, lo, hi, ast::expr_alt(discriminant, arms));
}

fn parse_expr(p: &parser) -> @ast::expr {
    ret parse_expr_res(p, UNRESTRICTED);
}

fn parse_expr_res(p: &parser, r: restriction) -> @ast::expr {
    let old = p.get_restriction();
    p.restrict(r);
    let e = parse_assign_expr(p);
    p.restrict(old);
    ret e;
}

fn parse_initializer(p: &parser) -> option::t<ast::initializer> {
    alt p.peek() {
      token::EQ. {
        p.bump();
        ret some({op: ast::init_assign, expr: parse_expr(p)});
      }
      token::LARROW. {
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

fn parse_pats(p: &parser) -> [@ast::pat] {
    let pats = [];
    while true {
        pats += [parse_pat(p)];
        if p.peek() == token::BINOP(token::OR) { p.bump(); } else { break; }
    }
    ret pats;
}

fn parse_pat(p: &parser) -> @ast::pat {
    let lo = p.get_lo_pos();
    let hi = p.get_hi_pos();
    let pat;
    alt p.peek() {
      token::UNDERSCORE. { p.bump(); pat = ast::pat_wild; }
      token::AT. {
        p.bump();
        let sub = parse_pat(p);
        pat = ast::pat_box(sub);
        hi = sub.span.hi;
      }
      token::LBRACE. {
        p.bump();
        let fields = [];
        let etc = false;
        let first = true;
        while p.peek() != token::RBRACE {
            if first { first = false; } else { expect(p, token::COMMA); }

            if p.peek() == token::UNDERSCORE {
                p.bump();
                if p.peek() != token::RBRACE {
                    p.fatal("expecting }, found " +
                                token::to_str(p.get_reader(), p.peek()));
                }
                etc = true;
                break;
            }

            let fieldname = parse_ident(p);
            let subpat;
            if p.peek() == token::COLON {
                p.bump();
                subpat = parse_pat(p);
            } else {
                if p.get_bad_expr_words().contains_key(fieldname) {
                    p.fatal("found " + fieldname + " in binding position");
                }
                subpat =
                    @{id: p.get_id(),
                      node: ast::pat_bind(fieldname),
                      span: ast::mk_sp(lo, hi)};
            }
            fields += [{ident: fieldname, pat: subpat}];
        }
        hi = p.get_hi_pos();
        p.bump();
        pat = ast::pat_rec(fields, etc);
      }
      token::LPAREN. {
        p.bump();
        if p.peek() == token::RPAREN {
            hi = p.get_hi_pos();
            p.bump();
            pat =
                ast::pat_lit(@{node: ast::lit_nil, span: ast::mk_sp(lo, hi)});
        } else {
            let fields = [parse_pat(p)];
            while p.peek() == token::COMMA {
                p.bump();
                fields += [parse_pat(p)];
            }
            if vec::len(fields) == 1u { expect(p, token::COMMA); }
            hi = p.get_hi_pos();
            expect(p, token::RPAREN);
            pat = ast::pat_tup(fields);
        }
      }
      tok {
        if !is_ident(tok) || is_word(p, "true") || is_word(p, "false") {
            let lit = parse_lit(p);
            hi = lit.span.hi;
            pat = ast::pat_lit(@lit);
        } else if is_plain_ident(p) &&
                      alt p.look_ahead(1u) {
                        token::DOT. | token::LPAREN. | token::LBRACKET. {
                          false
                        }
                        _ { true }
                      } {
            hi = p.get_hi_pos();
            pat = ast::pat_bind(parse_value_ident(p));
        } else {
            let tag_path = parse_path_and_ty_param_substs(p);
            hi = tag_path.span.hi;
            let args: [@ast::pat];
            alt p.peek() {
              token::LPAREN. {
                let a =
                    parse_seq(token::LPAREN, token::RPAREN,
                              some(token::COMMA), parse_pat, p);
                args = a.node;
                hi = a.span.hi;
              }
              token::DOT. { args = []; p.bump(); }
              _ { expect(p, token::LPAREN); fail; }
            }
            pat = ast::pat_tag(tag_path, args);
        }
      }
    }
    ret @{id: p.get_id(), node: pat, span: ast::mk_sp(lo, hi)};
}

fn parse_local(p: &parser, allow_init: bool) -> @ast::local {
    let lo = p.get_lo_pos();
    let pat = parse_pat(p);
    let ty = @spanned(lo, lo, ast::ty_infer);
    if eat(p, token::COLON) { ty = parse_ty(p, false); }
    let init = if allow_init { parse_initializer(p) } else { none };
    ret @spanned(lo, p.get_last_hi_pos(),
                 {ty: ty, pat: pat, init: init, id: p.get_id()});
}

fn parse_let(p: &parser) -> @ast::decl {
    let lo = p.get_lo_pos();
    let locals = [parse_local(p, true)];
    while p.peek() == token::COMMA {
        p.bump();
        locals += [parse_local(p, true)];
    }
    ret @spanned(lo, p.get_last_hi_pos(), ast::decl_local(locals));
}

fn parse_stmt(p: &parser) -> @ast::stmt {
    if p.get_file_type() == SOURCE_FILE {
        ret parse_source_stmt(p);
    } else { ret parse_crate_stmt(p); }
}

fn parse_crate_stmt(p: &parser) -> @ast::stmt {
    let cdir = parse_crate_directive(p, []);
    ret @spanned(cdir.span.lo, cdir.span.hi,
                 ast::stmt_crate_directive(@cdir));
}

fn parse_source_stmt(p: &parser) -> @ast::stmt {
    let lo = p.get_lo_pos();
    if eat_word(p, "let") {
        let decl = parse_let(p);
        ret @spanned(lo, decl.span.hi, ast::stmt_decl(decl, p.get_id()));
    } else {

        let item_attrs;
        alt parse_outer_attrs_or_ext(p) {
          none. { item_attrs = []; }
          some(left(attrs)) { item_attrs = attrs; }
          some(right(ext)) {
            ret @spanned(lo, ext.span.hi, ast::stmt_expr(ext, p.get_id()));
          }
        }

        let maybe_item = parse_item(p, item_attrs);

        // If we have attributes then we should have an item
        if vec::len(item_attrs) > 0u {
            alt maybe_item {
              some(_) {/* fallthrough */ }
              _ { ret p.fatal("expected item"); }
            }
        }


        alt maybe_item {
          some(i) {
            let hi = i.span.hi;
            let decl = @spanned(lo, hi, ast::decl_item(i));
            ret @spanned(lo, hi, ast::stmt_decl(decl, p.get_id()));
          }
          none. {
            // Remainder are line-expr stmts.
            let e = parse_expr(p);
            ret @spanned(lo, e.span.hi, ast::stmt_expr(e, p.get_id()));
          }
          _ { p.fatal("expected statement"); }
        }
    }
}

fn stmt_to_expr(stmt: @ast::stmt) -> option::t<@ast::expr> {
    ret alt stmt.node { ast::stmt_expr(e, _) { some(e) } _ { none } };
}

fn stmt_ends_with_semi(stmt: &ast::stmt) -> bool {
    alt stmt.node {
      ast::stmt_decl(d, _) {
        ret alt d.node {
              ast::decl_local(_) { true }
              ast::decl_item(_) { false }
            }
      }
      ast::stmt_expr(e, _) {
        ret alt e.node {
              ast::expr_vec(_, _) { true }
              ast::expr_rec(_, _) { true }
              ast::expr_tup(_) { true }
              ast::expr_call(_, _) { true }
              ast::expr_self_method(_) { false }
              ast::expr_bind(_, _) { true }
              ast::expr_binary(_, _, _) { true }
              ast::expr_unary(_, _) { true }
              ast::expr_lit(_) { true }
              ast::expr_cast(_, _) { true }
              ast::expr_if(_, _, _) { false }
              ast::expr_ternary(_, _, _) { true }
              ast::expr_for(_, _, _) { false }
              ast::expr_for_each(_, _, _) { false }
              ast::expr_while(_, _) { false }
              ast::expr_do_while(_, _) { false }
              ast::expr_alt(_, _) { false }
              ast::expr_fn(_) { false }
              ast::expr_block(_) { false }
              ast::expr_move(_, _) { true }
              ast::expr_assign(_, _) { true }
              ast::expr_swap(_, _) { true }
              ast::expr_assign_op(_, _, _) { true }
              ast::expr_field(_, _) { true }
              ast::expr_index(_, _) { true }
              ast::expr_path(_) { true }
              ast::expr_mac(_) { true }
              ast::expr_fail(_) { true }
              ast::expr_break. { true }
              ast::expr_cont. { true }
              ast::expr_ret(_) { true }
              ast::expr_put(_) { true }
              ast::expr_be(_) { true }
              ast::expr_log(_, _) { true }
              ast::expr_check(_, _) { true }
              ast::expr_if_check(_, _, _) { false }
              ast::expr_anon_obj(_) { false }
              ast::expr_assert(_) { true }
            }
      }


      // We should not be calling this on a cdir.
      ast::stmt_crate_directive(cdir) {
        fail;
      }
    }
}

fn parse_block(p: &parser) -> ast::blk {
    let lo = p.get_lo_pos();
    expect(p, token::LBRACE);
    be parse_block_tail(p, lo);
}

// some blocks start with "#{"...
fn parse_block_tail(p: &parser, lo: uint) -> ast::blk {
    let stmts: [@ast::stmt] = [];
    let expr: option::t<@ast::expr> = none;
    while p.peek() != token::RBRACE {
        alt p.peek() {
          token::SEMI. {
            p.bump(); // empty
          }
          _ {
            let stmt = parse_stmt(p);
            alt stmt_to_expr(stmt) {
              some(e) {
                alt p.peek() {
                  token::SEMI. { p.bump(); stmts += [stmt]; }
                  token::RBRACE. { expr = some(e); }
                  t {
                    if stmt_ends_with_semi(*stmt) {
                        p.fatal("expected ';' or '}' after " +
                                    "expression but found " +
                                    token::to_str(p.get_reader(), t));
                    }
                    stmts += [stmt];
                  }
                }
              }
              none. {
                // Not an expression statement.
                stmts += [stmt];


                if p.get_file_type() == SOURCE_FILE &&
                       stmt_ends_with_semi(*stmt) {
                    expect(p, token::SEMI);
                }
              }
            }
          }
        }
    }
    let hi = p.get_hi_pos();
    p.bump();
    let bloc = {stmts: stmts, expr: expr, id: p.get_id()};
    ret spanned(lo, hi, bloc);
}

fn parse_ty_param(p: &parser) -> ast::ty_param {
    let k =
        alt p.peek() {
          token::TILDE. { p.bump(); ast::kind_unique }
          token::AT. { p.bump(); ast::kind_shared }
          _ { ast::kind_pinned }
        };
    ret {ident: parse_ident(p), kind: k};
}

fn parse_ty_params(p: &parser) -> [ast::ty_param] {
    let ty_params: [ast::ty_param] = [];
    if p.peek() == token::LT {
        p.bump();
        ty_params = parse_seq_to_gt(some(token::COMMA), parse_ty_param, p);
    }
    if p.peek() == token::LT {
        ty_params =
            parse_seq(token::LT, token::GT, some(token::COMMA),
                      parse_ty_param, p).node;
    }
    ret ty_params;
}

fn parse_fn_decl(p: &parser, purity: ast::purity, il: ast::inlineness) ->
   ast::fn_decl {
    let inputs: ast::spanned<[ast::arg]> =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA), parse_arg,
                  p);
    let rslt: ty_or_bang;
    // Use the args list to translate each bound variable
    // mentioned in a constraint to an arg index.
    // Seems weird to do this in the parser, but I'm not sure how else to.
    let constrs = [];
    if p.peek() == token::COLON {
        p.bump();
        constrs = parse_constrs(bind parse_ty_constr(inputs.node, _), p);
    }
    if p.peek() == token::RARROW {
        p.bump();
        rslt = parse_ty_or_bang(p);
    } else {
        rslt = a_ty(@spanned(inputs.span.lo, inputs.span.hi, ast::ty_nil));
    }
    alt rslt {
      a_ty(t) {
        ret {inputs: inputs.node,
             output: t,
             purity: purity,
             il: il,
             cf: ast::return,
             constraints: constrs};
      }
      a_bang. {
        ret {inputs: inputs.node,
             output: @spanned(p.get_lo_pos(), p.get_hi_pos(), ast::ty_bot),
             purity: purity,
             il: il,
             cf: ast::noreturn,
             constraints: constrs};
      }
    }
}

fn parse_fn_block_decl(p: &parser) -> ast::fn_decl {
    let inputs: ast::spanned<[ast::arg]> =
        parse_seq(token::BINOP(token::OR), token::BINOP(token::OR),
                  some(token::COMMA), parse_fn_block_arg, p);
    ret {inputs: inputs.node,
         output: @spanned(p.get_lo_pos(), p.get_hi_pos(), ast::ty_infer),
         purity: ast::impure_fn,
         il: ast::il_normal,
         cf: ast::return,
         constraints: []};
}

fn parse_fn(p: &parser, proto: ast::proto, purity: ast::purity,
            il: ast::inlineness) -> ast::_fn {
    let decl = parse_fn_decl(p, purity, il);
    let body = parse_block(p);
    ret {decl: decl, proto: proto, body: body};
}

fn parse_fn_header(p: &parser) -> {ident: ast::ident, tps: [ast::ty_param]} {
    let id = parse_value_ident(p);
    let ty_params = parse_ty_params(p);
    ret {ident: id, tps: ty_params};
}

fn mk_item(p: &parser, lo: uint, hi: uint, ident: &ast::ident,
           node: &ast::item_, attrs: &[ast::attribute]) -> @ast::item {
    ret @{ident: ident,
          attrs: attrs,
          id: p.get_id(),
          node: node,
          span: ast::mk_sp(lo, hi)};
}

fn parse_item_fn_or_iter(p: &parser, purity: ast::purity, proto: ast::proto,
                         attrs: &[ast::attribute], il: ast::inlineness) ->
   @ast::item {
    let lo = p.get_last_lo_pos();
    let t = parse_fn_header(p);
    let f = parse_fn(p, proto, purity, il);
    ret mk_item(p, lo, f.body.span.hi, t.ident, ast::item_fn(f, t.tps),
                attrs);
}

fn parse_obj_field(p: &parser) -> ast::obj_field {
    let mut = parse_mutability(p);
    let ident = parse_value_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    ret {mut: mut, ty: ty, ident: ident, id: p.get_id()};
}

fn parse_anon_obj_field(p: &parser) -> ast::anon_obj_field {
    let mut = parse_mutability(p);
    let ident = parse_value_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    expect(p, token::EQ);
    let expr = parse_expr(p);
    ret {mut: mut, ty: ty, expr: expr, ident: ident, id: p.get_id()};
}

fn parse_method(p: &parser) -> @ast::method {
    let lo = p.get_lo_pos();
    let proto = parse_proto(p);
    let ident = parse_value_ident(p);
    let f = parse_fn(p, proto, ast::impure_fn, ast::il_normal);
    let meth = {ident: ident, meth: f, id: p.get_id()};
    ret @spanned(lo, f.body.span.hi, meth);
}

fn parse_item_obj(p: &parser, attrs: &[ast::attribute]) -> @ast::item {
    let lo = p.get_last_lo_pos();
    let ident = parse_value_ident(p);
    let ty_params = parse_ty_params(p);
    let fields: ast::spanned<[ast::obj_field]> =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                  parse_obj_field, p);
    let meths: [@ast::method] = [];
    expect(p, token::LBRACE);
    while p.peek() != token::RBRACE { meths += [parse_method(p)]; }
    let hi = p.get_hi_pos();
    expect(p, token::RBRACE);
    let ob: ast::_obj = {fields: fields.node, methods: meths};
    ret mk_item(p, lo, hi, ident, ast::item_obj(ob, ty_params, p.get_id()),
                attrs);
}

fn parse_item_res(p: &parser, attrs: &[ast::attribute]) -> @ast::item {
    let lo = p.get_last_lo_pos();
    let ident = parse_value_ident(p);
    let ty_params = parse_ty_params(p);
    expect(p, token::LPAREN);
    let arg_ident = parse_value_ident(p);
    expect(p, token::COLON);
    let t = parse_ty(p, false);
    expect(p, token::RPAREN);
    let dtor = parse_block(p);
    let decl =
        {inputs:
             [{mode: ast::alias(false),
               ty: t,
               ident: arg_ident,
               id: p.get_id()}],
         output: @spanned(lo, lo, ast::ty_nil),
         purity: ast::impure_fn,
         il: ast::il_normal,
         cf: ast::return,
         constraints: []};
    let f = {decl: decl, proto: ast::proto_fn, body: dtor};
    ret mk_item(p, lo, dtor.span.hi, ident,
                ast::item_res(f, p.get_id(), ty_params, p.get_id()), attrs);
}

fn parse_mod_items(p: &parser, term: token::token,
                   first_item_attrs: &[ast::attribute]) -> ast::_mod {
    // Shouldn't be any view items since we've already parsed an item attr
    let view_items =
        if vec::len(first_item_attrs) == 0u { parse_view(p) } else { [] };
    let items: [@ast::item] = [];
    let initial_attrs = first_item_attrs;
    while p.peek() != term {
        let attrs = initial_attrs + parse_outer_attributes(p);
        initial_attrs = [];
        alt parse_item(p, attrs) {
          some(i) { items += [i]; }
          _ {
            p.fatal("expected item but found " +
                        token::to_str(p.get_reader(), p.peek()));
          }
        }
    }
    ret {view_items: view_items, items: items};
}

fn parse_item_const(p: &parser, attrs: &[ast::attribute]) -> @ast::item {
    let lo = p.get_last_lo_pos();
    let id = parse_value_ident(p);
    expect(p, token::COLON);
    let ty = parse_ty(p, false);
    expect(p, token::EQ);
    let e = parse_expr(p);
    let hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret mk_item(p, lo, hi, id, ast::item_const(ty, e), attrs);
}

fn parse_item_mod(p: &parser, attrs: &[ast::attribute]) -> @ast::item {
    let lo = p.get_last_lo_pos();
    let id = parse_ident(p);
    expect(p, token::LBRACE);
    let inner_attrs = parse_inner_attrs_and_next(p);
    let first_item_outer_attrs = inner_attrs.next;
    let m = parse_mod_items(p, token::RBRACE, first_item_outer_attrs);
    let hi = p.get_hi_pos();
    expect(p, token::RBRACE);
    ret mk_item(p, lo, hi, id, ast::item_mod(m), attrs + inner_attrs.inner);
}

fn parse_item_native_type(p: &parser, attrs: &[ast::attribute]) ->
   @ast::native_item {
    let t = parse_type_decl(p);
    let hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret @{ident: t.ident,
          attrs: attrs,
          node: ast::native_item_ty,
          id: p.get_id(),
          span: ast::mk_sp(t.lo, hi)};
}

fn parse_item_native_fn(p: &parser, attrs: &[ast::attribute]) ->
   @ast::native_item {
    let lo = p.get_last_lo_pos();
    let t = parse_fn_header(p);
    let decl = parse_fn_decl(p, ast::impure_fn, ast::il_normal);
    let link_name = none;
    if p.peek() == token::EQ { p.bump(); link_name = some(parse_str(p)); }
    let hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret @{ident: t.ident,
          attrs: attrs,
          node: ast::native_item_fn(link_name, decl, t.tps),
          id: p.get_id(),
          span: ast::mk_sp(lo, hi)};
}

fn parse_native_item(p: &parser, attrs: &[ast::attribute]) ->
   @ast::native_item {
    if eat_word(p, "type") {
        ret parse_item_native_type(p, attrs);
    } else if eat_word(p, "fn") {
        ret parse_item_native_fn(p, attrs);
    } else { unexpected(p, p.peek()); }
}

fn parse_native_mod_items(p: &parser, native_name: &str, abi: ast::native_abi,
                          first_item_attrs: &[ast::attribute]) ->
   ast::native_mod {
    // Shouldn't be any view items since we've already parsed an item attr
    let view_items =
        if vec::len(first_item_attrs) == 0u {
            parse_native_view(p)
        } else { [] };
    let items: [@ast::native_item] = [];
    let initial_attrs = first_item_attrs;
    while p.peek() != token::RBRACE {
        let attrs = initial_attrs + parse_outer_attributes(p);
        initial_attrs = [];
        items += [parse_native_item(p, attrs)];
    }
    ret {native_name: native_name,
         abi: abi,
         view_items: view_items,
         items: items};
}

fn parse_item_native_mod(p: &parser, attrs: &[ast::attribute]) -> @ast::item {
    let lo = p.get_last_lo_pos();
    let abi = ast::native_abi_cdecl;
    if !is_word(p, "mod") {
        let t = parse_str(p);
        if str::eq(t, "cdecl") {
        } else if str::eq(t, "rust") {
            abi = ast::native_abi_rust;
        } else if str::eq(t, "llvm") {
            abi = ast::native_abi_llvm;
        } else if str::eq(t, "rust-intrinsic") {
            abi = ast::native_abi_rust_intrinsic;
        } else if str::eq(t, "x86stdcall") {
            abi = ast::native_abi_x86stdcall;
        } else { p.fatal("unsupported abi: " + t); }
    }
    expect_word(p, "mod");
    let id = parse_ident(p);
    let native_name;
    if p.peek() == token::EQ {
        expect(p, token::EQ);
        native_name = parse_str(p);
    } else { native_name = id; }
    expect(p, token::LBRACE);
    let more_attrs = parse_inner_attrs_and_next(p);
    let inner_attrs = more_attrs.inner;
    let first_item_outer_attrs = more_attrs.next;
    let m =
        parse_native_mod_items(p, native_name, abi, first_item_outer_attrs);
    let hi = p.get_hi_pos();
    expect(p, token::RBRACE);
    ret mk_item(p, lo, hi, id, ast::item_native_mod(m), attrs + inner_attrs);
}

fn parse_type_decl(p: &parser) -> {lo: uint, ident: ast::ident} {
    let lo = p.get_last_lo_pos();
    let id = parse_ident(p);
    ret {lo: lo, ident: id};
}

fn parse_item_type(p: &parser, attrs: &[ast::attribute]) -> @ast::item {
    let t = parse_type_decl(p);
    let tps = parse_ty_params(p);
    expect(p, token::EQ);
    let ty = parse_ty(p, false);
    let hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret mk_item(p, t.lo, hi, t.ident, ast::item_ty(ty, tps), attrs);
}

fn parse_item_tag(p: &parser, attrs: &[ast::attribute]) -> @ast::item {
    let lo = p.get_last_lo_pos();
    let id = parse_ident(p);
    let ty_params = parse_ty_params(p);
    let variants: [ast::variant] = [];
    // Newtype syntax
    if p.peek() == token::EQ {
        if p.get_bad_expr_words().contains_key(id) {
            p.fatal("found " + id + " in tag constructor position");
        }
        p.bump();
        let ty = parse_ty(p, false);
        expect(p, token::SEMI);
        let variant =
            spanned(ty.span.lo, ty.span.hi,
                    {name: id,
                     args: [{ty: ty, id: p.get_id()}],
                     id: p.get_id()});
        ret mk_item(p, lo, ty.span.hi, id,
                    ast::item_tag([variant], ty_params), attrs);
    }
    expect(p, token::LBRACE);
    while p.peek() != token::RBRACE {
        let tok = p.peek();
        alt tok {
          token::IDENT(name, _) {
            check_bad_word(p);
            let vlo = p.get_lo_pos();
            p.bump();
            let args: [ast::variant_arg] = [];
            let vhi = p.get_hi_pos();
            alt p.peek() {
              token::LPAREN. {
                let arg_tys =
                    parse_seq(token::LPAREN, token::RPAREN,
                              some(token::COMMA), bind parse_ty(_, false), p);
                for ty: @ast::ty in arg_tys.node {
                    args += [{ty: ty, id: p.get_id()}];
                }
                vhi = arg_tys.span.hi;
              }
              _ {/* empty */ }
            }
            expect(p, token::SEMI);
            p.get_id();
            let vr = {name: p.get_str(name), args: args, id: p.get_id()};
            variants += [spanned(vlo, vhi, vr)];
          }
          token::RBRACE. {/* empty */ }
          _ {
            p.fatal("expected name of variant or '}' but found " +
                        token::to_str(p.get_reader(), tok));
          }
        }
    }
    let hi = p.get_hi_pos();
    p.bump();
    ret mk_item(p, lo, hi, id, ast::item_tag(variants, ty_params), attrs);
}

fn parse_auth(p: &parser) -> ast::_auth {
    if eat_word(p, "unsafe") {
        ret ast::auth_unsafe;
    } else { unexpected(p, p.peek()); }
}

fn parse_item(p: &parser, attrs: &[ast::attribute]) -> option::t<@ast::item> {
    if eat_word(p, "const") {
        ret some(parse_item_const(p, attrs));
    } else if eat_word(p, "inline") {
        expect_word(p, "fn");
        ret some(parse_item_fn_or_iter(p, ast::impure_fn, ast::proto_fn,
                                       attrs, ast::il_inline));
    } else if is_word(p, "fn") && p.look_ahead(1u) != token::LPAREN {
        p.bump();
        ret some(parse_item_fn_or_iter(p, ast::impure_fn, ast::proto_fn,
                                       attrs, ast::il_normal));
    } else if eat_word(p, "pred") {
        ret some(parse_item_fn_or_iter(p, ast::pure_fn, ast::proto_fn, attrs,
                                       ast::il_normal));
    } else if eat_word(p, "iter") {
        ret some(parse_item_fn_or_iter(p, ast::impure_fn, ast::proto_iter,
                                       attrs, ast::il_normal));
    } else if eat_word(p, "mod") {
        ret some(parse_item_mod(p, attrs));
    } else if eat_word(p, "native") {
        ret some(parse_item_native_mod(p, attrs));
    }
    if eat_word(p, "type") {
        ret some(parse_item_type(p, attrs));
    } else if eat_word(p, "tag") {
        ret some(parse_item_tag(p, attrs));
    } else if is_word(p, "obj") && p.look_ahead(1u) != token::LPAREN {
        p.bump();
        ret some(parse_item_obj(p, attrs));
    } else if eat_word(p, "resource") {
        ret some(parse_item_res(p, attrs));
    } else { ret none; }
}

// A type to distingush between the parsing of item attributes or syntax
// extensions, which both begin with token.POUND
type attr_or_ext = option::t<either::t<[ast::attribute], @ast::expr>>;

fn parse_outer_attrs_or_ext(p: &parser) -> attr_or_ext {
    if p.peek() == token::POUND {
        let lo = p.get_lo_pos();
        p.bump();
        if p.peek() == token::LBRACKET {
            let first_attr = parse_attribute_naked(p, ast::attr_outer, lo);
            ret some(left([first_attr] + parse_outer_attributes(p)));
        } else if !(p.peek() == token::LT || p.peek() == token::LBRACKET) {
            ret some(right(parse_syntax_ext_naked(p, lo)));
        } else { ret none; }
    } else { ret none; }
}

// Parse attributes that appear before an item
fn parse_outer_attributes(p: &parser) -> [ast::attribute] {
    let attrs: [ast::attribute] = [];
    while p.peek() == token::POUND {
        attrs += [parse_attribute(p, ast::attr_outer)];
    }
    ret attrs;
}

fn parse_attribute(p: &parser, style: ast::attr_style) -> ast::attribute {
    let lo = p.get_lo_pos();
    expect(p, token::POUND);
    ret parse_attribute_naked(p, style, lo);
}

fn parse_attribute_naked(p: &parser, style: ast::attr_style, lo: uint) ->
   ast::attribute {
    expect(p, token::LBRACKET);
    let meta_item = parse_meta_item(p);
    expect(p, token::RBRACKET);
    let hi = p.get_hi_pos();
    ret spanned(lo, hi, {style: style, value: *meta_item});
}

// Parse attributes that appear after the opening of an item, each terminated
// by a semicolon. In addition to a vector of inner attributes, this function
// also returns a vector that may contain the first outer attribute of the
// next item (since we can't know whether the attribute is an inner attribute
// of the containing item or an outer attribute of the first contained item
// until we see the semi).
fn parse_inner_attrs_and_next(p: &parser) ->
   {inner: [ast::attribute], next: [ast::attribute]} {
    let inner_attrs: [ast::attribute] = [];
    let next_outer_attrs: [ast::attribute] = [];
    while p.peek() == token::POUND {
        let attr = parse_attribute(p, ast::attr_inner);
        if p.peek() == token::SEMI {
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

fn parse_meta_item(p: &parser) -> @ast::meta_item {
    let lo = p.get_lo_pos();
    let ident = parse_ident(p);
    alt p.peek() {
      token::EQ. {
        p.bump();
        let lit = parse_lit(p);
        let hi = p.get_hi_pos();
        ret @spanned(lo, hi, ast::meta_name_value(ident, lit));
      }
      token::LPAREN. {
        let inner_items = parse_meta_seq(p);
        let hi = p.get_hi_pos();
        ret @spanned(lo, hi, ast::meta_list(ident, inner_items));
      }
      _ {
        let hi = p.get_hi_pos();
        ret @spanned(lo, hi, ast::meta_word(ident));
      }
    }
}

fn parse_meta_seq(p: &parser) -> [@ast::meta_item] {
    ret parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                  parse_meta_item, p).node;
}

fn parse_optional_meta(p: &parser) -> [@ast::meta_item] {
    alt p.peek() { token::LPAREN. { ret parse_meta_seq(p); } _ { ret []; } }
}

fn parse_use(p: &parser) -> ast::view_item_ {
    let ident = parse_ident(p);
    let metadata = parse_optional_meta(p);
    ret ast::view_item_use(ident, metadata, p.get_id());
}

fn parse_rest_import_name(p: &parser, first: ast::ident,
                          def_ident: option::t<ast::ident>) ->
   ast::view_item_ {
    let identifiers: [ast::ident] = [first];
    let glob: bool = false;
    let from_idents = option::none::<[ast::import_ident]>;
    while true {
        alt p.peek() {
          token::SEMI. { break; }
          token::MOD_SEP. {
            if glob { p.fatal("cannot path into a glob"); }
            if option::is_some(from_idents) {
                p.fatal("cannot path into import list");
            }
            p.bump();
          }
          _ { p.fatal("expecting '::' or ';'"); }
        }
        alt p.peek() {
          token::IDENT(_, _) { identifiers += [parse_ident(p)]; }


          //the lexer can't tell the different kinds of stars apart ) :
          token::BINOP(token::STAR.) {
            glob = true;
            p.bump();
          }


          token::LBRACE. {
            fn parse_import_ident(p: &parser) -> ast::import_ident {
                let lo = p.get_lo_pos();
                let ident = parse_ident(p);
                let hi = p.get_hi_pos();
                ret spanned(lo, hi, {name: ident, id: p.get_id()});
            }
            let from_idents_ =
                parse_seq(token::LBRACE, token::RBRACE, some(token::COMMA),
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
        ret ast::view_item_import(i, identifiers, p.get_id());
      }
      _ {
        if glob {
            ret ast::view_item_import_glob(identifiers, p.get_id());
        } else if option::is_some(from_idents) {
            ret ast::view_item_import_from(identifiers,
                                           option::get(from_idents),
                                           p.get_id());
        } else {
            let len = vec::len(identifiers);
            ret ast::view_item_import(identifiers[len - 1u], identifiers,
                                      p.get_id());
        }
      }
    }
}

fn parse_full_import_name(p: &parser, def_ident: ast::ident) ->
   ast::view_item_ {
    alt p.peek() {
      token::IDENT(i, _) {
        p.bump();
        ret parse_rest_import_name(p, p.get_str(i), some(def_ident));
      }
      _ { p.fatal("expecting an identifier"); }
    }
}

fn parse_import(p: &parser) -> ast::view_item_ {
    alt p.peek() {
      token::IDENT(i, _) {
        p.bump();
        alt p.peek() {
          token::EQ. {
            p.bump();
            ret parse_full_import_name(p, p.get_str(i));
          }
          _ { ret parse_rest_import_name(p, p.get_str(i), none); }
        }
      }
      _ { p.fatal("expecting an identifier"); }
    }
}

fn parse_export(p: &parser) -> ast::view_item_ {
    let ids =
        parse_seq_to_before_end(token::SEMI, option::some(token::COMMA),
                                parse_ident, p);
    ret ast::view_item_export(ids, p.get_id());
}

fn parse_view_item(p: &parser) -> @ast::view_item {
    let lo = p.get_lo_pos();
    let the_item =
        if eat_word(p, "use") {
            parse_use(p)
        } else if eat_word(p, "import") {
            parse_import(p)
        } else if eat_word(p, "export") { parse_export(p) } else { fail };
    let hi = p.get_lo_pos();
    expect(p, token::SEMI);
    ret @spanned(lo, hi, the_item);
}

fn is_view_item(p: &parser) -> bool {
    alt p.peek() {
      token::IDENT(sid, false) {
        let st = p.get_str(sid);
        ret str::eq(st, "use") || str::eq(st, "import") ||
                str::eq(st, "export");
      }
      _ { ret false; }
    }
}

fn parse_view(p: &parser) -> [@ast::view_item] {
    let items: [@ast::view_item] = [];
    while is_view_item(p) { items += [parse_view_item(p)]; }
    ret items;
}

fn parse_native_view(p: &parser) -> [@ast::view_item] {
    let items: [@ast::view_item] = [];
    while is_view_item(p) { items += [parse_view_item(p)]; }
    ret items;
}

fn parse_crate_from_source_file(input: &str, cfg: &ast::crate_cfg,
                                sess: &parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, 0u, 0u, SOURCE_FILE);
    ret parse_crate_mod(p, cfg);
}

fn parse_crate_from_source_str(name: &str, source: &str, cfg: &ast::crate_cfg,
                               sess: &parse_sess) -> @ast::crate {
    let ftype = SOURCE_FILE;
    let filemap = codemap::new_filemap(name, 0u, 0u);
    sess.cm.files += [filemap];
    let itr = @interner::mk(str::hash, str::eq);
    let rdr = lexer::new_reader(sess.cm, source, filemap, itr);
    let p = new_parser(sess, cfg, rdr, ftype);
    ret parse_crate_mod(p, cfg);
}

// Parses a source module as a crate
fn parse_crate_mod(p: &parser, _cfg: &ast::crate_cfg) -> @ast::crate {
    let lo = p.get_lo_pos();
    let crate_attrs = parse_inner_attrs_and_next(p);
    let first_item_outer_attrs = crate_attrs.next;
    let m = parse_mod_items(p, token::EOF, first_item_outer_attrs);
    ret @spanned(lo, p.get_lo_pos(),
                 {directives: [],
                  module: m,
                  attrs: crate_attrs.inner,
                  config: p.get_cfg()});
}

fn parse_str(p: &parser) -> ast::ident {
    alt p.peek() {
      token::LIT_STR(s) { p.bump(); ret p.get_str(s); }
      _ { fail; }
    }
}

// Logic for parsing crate files (.rc)
//
// Each crate file is a sequence of directives.
//
// Each directive imperatively extends its environment with 0 or more items.
fn parse_crate_directive(p: &parser, first_outer_attr: &[ast::attribute]) ->
   ast::crate_directive {

    // Collect the next attributes
    let outer_attrs = first_outer_attr + parse_outer_attributes(p);
    // In a crate file outer attributes are only going to apply to mods
    let expect_mod = vec::len(outer_attrs) > 0u;

    let lo = p.get_lo_pos();
    if expect_mod || is_word(p, "mod") {
        expect_word(p, "mod");
        let id = parse_ident(p);
        let file_opt =
            alt p.peek() {
              token::EQ. { p.bump(); some(parse_str(p)) }
              _ { none }
            };
        alt p.peek() {


          // mod x = "foo.rs";
          token::SEMI. {
            let hi = p.get_hi_pos();
            p.bump();
            ret spanned(lo, hi, ast::cdir_src_mod(id, file_opt, outer_attrs));
          }


          // mod x = "foo_dir" { ...directives... }
          token::LBRACE. {
            p.bump();
            let inner_attrs = parse_inner_attrs_and_next(p);
            let mod_attrs = outer_attrs + inner_attrs.inner;
            let next_outer_attr = inner_attrs.next;
            let cdirs =
                parse_crate_directives(p, token::RBRACE, next_outer_attr);
            let hi = p.get_hi_pos();
            expect(p, token::RBRACE);
            ret spanned(lo, hi,
                        ast::cdir_dir_mod(id, file_opt, cdirs, mod_attrs));
          }
          t { unexpected(p, t); }
        }
    } else if eat_word(p, "auth") {
        let n = parse_path(p);
        expect(p, token::EQ);
        let a = parse_auth(p);
        let hi = p.get_hi_pos();
        expect(p, token::SEMI);
        ret spanned(lo, hi, ast::cdir_auth(n, a));
    } else if is_view_item(p) {
        let vi = parse_view_item(p);
        ret spanned(lo, vi.span.hi, ast::cdir_view_item(vi));
    } else { ret p.fatal("expected crate directive"); }
}

fn parse_crate_directives(p: &parser, term: token::token,
                          first_outer_attr: &[ast::attribute]) ->
   [@ast::crate_directive] {

    // This is pretty ugly. If we have an outer attribute then we can't accept
    // seeing the terminator next, so if we do see it then fail the same way
    // parse_crate_directive would
    if vec::len(first_outer_attr) > 0u && p.peek() == term {
        expect_word(p, "mod");
    }

    let cdirs: [@ast::crate_directive] = [];
    while p.peek() != term {
        let cdir = @parse_crate_directive(p, first_outer_attr);
        cdirs += [cdir];
    }
    ret cdirs;
}

fn parse_crate_from_crate_file(input: &str, cfg: &ast::crate_cfg,
                               sess: &parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, 0u, 0u, CRATE_FILE);
    let lo = p.get_lo_pos();
    let prefix = std::fs::dirname(p.get_filemap().name);
    let leading_attrs = parse_inner_attrs_and_next(p);
    let crate_attrs = leading_attrs.inner;
    let first_cdir_attr = leading_attrs.next;
    let cdirs = parse_crate_directives(p, token::EOF, first_cdir_attr);
    let deps: [str] = [];
    let cx =
        @{p: p,
          mode: eval::mode_parse,
          mutable deps: deps,
          sess: sess,
          mutable chpos: p.get_chpos(),
          mutable byte_pos: p.get_byte_pos(),
          cfg: p.get_cfg()};
    let m = eval::eval_crate_directives_to_mod(cx, cdirs, prefix);
    let hi = p.get_hi_pos();
    expect(p, token::EOF);
    ret @spanned(lo, hi,
                 {directives: cdirs,
                  module: m,
                  attrs: crate_attrs,
                  config: p.get_cfg()});
}

fn parse_crate_from_file(input: &str, cfg: &ast::crate_cfg, sess: &parse_sess)
   -> @ast::crate {
    if str::ends_with(input, ".rc") {
        parse_crate_from_crate_file(input, cfg, sess)
    } else if str::ends_with(input, ".rs") {
        parse_crate_from_source_file(input, cfg, sess)
    } else {
        codemap::emit_error(none, "unknown input file type: " + input,
                            sess.cm);
        fail
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

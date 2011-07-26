
import std::ioivec;
import std::ivec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::either;
import std::either::left;
import std::either::right;
import std::map::hashmap;
import token::can_begin_expr;
import ex=ext::base;
import codemap::span;
import std::map::new_str_hash;
import util::interner;
import ast::node_id;
import ast::spanned;

tag restriction { UNRESTRICTED; RESTRICT_NO_CALL_EXPRS; }

tag file_type { CRATE_FILE; SOURCE_FILE; }

tag ty_or_bang { a_ty(@ast::ty); a_bang; }

type parse_sess = @rec(codemap::codemap cm,
                       mutable node_id next_id);

fn next_node_id(&parse_sess sess) -> node_id {
    auto rv = sess.next_id;
    sess.next_id += 1;
    ret rv;
}

type parser =
    obj {
        fn peek() -> token::token;
        fn bump();
        fn look_ahead(uint) -> token::token;
        fn fatal(str) -> !;
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
        fn get_prec_table() -> @op_spec[];
        fn get_str(token::str_num) -> str;
        fn get_reader() -> lexer::reader;
        fn get_filemap() -> codemap::filemap;
        fn get_bad_expr_words() -> hashmap[str, ()];
        fn get_chpos() -> uint;
        fn get_byte_pos() -> uint;
        fn get_id() -> node_id;
        fn get_sess() -> parse_sess;
    };

fn new_parser_from_file(parse_sess sess, ast::crate_cfg cfg,
                        str path, uint chpos, uint byte_pos) -> parser {
    auto ftype = SOURCE_FILE;
    if (str::ends_with(path, ".rc")) { ftype = CRATE_FILE; }
    auto srdr = ioivec::file_reader(path);
    auto src = str::unsafe_from_bytes_ivec(srdr.read_whole_stream());
    auto filemap = codemap::new_filemap(path, chpos, byte_pos);
    sess.cm.files += ~[filemap];
    auto itr = @interner::mk(str::hash, str::eq);
    auto rdr = lexer::new_reader(sess.cm, src, filemap, itr);

    ret new_parser(sess, cfg, rdr, ftype);
}

fn new_parser(parse_sess sess, ast::crate_cfg cfg, lexer::reader rdr,
              file_type ftype) -> parser {
    obj stdio_parser(parse_sess sess,
                     ast::crate_cfg cfg,
                     file_type ftype,
                     mutable token::token tok,
                     mutable span tok_span,
                     mutable span last_tok_span,
                     mutable tup(token::token, span)[] buffer,
                     mutable restriction restr,
                     lexer::reader rdr,
                     @op_spec[] precs,
                     hashmap[str, ()] bad_words) {
        fn peek() -> token::token { ret tok; }
        fn bump() {
            last_tok_span = tok_span;
            if ivec::len(buffer) == 0u {
                auto next = lexer::next_token(rdr);
                tok = next._0;
                tok_span = rec(lo=next._1, hi=rdr.get_chpos());
            } else {
                auto next = ivec::pop(buffer);
                tok = next._0;
                tok_span = next._1;
            }
        }
        fn look_ahead(uint distance) -> token::token {
            while ivec::len(buffer) < distance {
                auto next = lexer::next_token(rdr);
                auto sp = rec(lo=next._1, hi=rdr.get_chpos());
                buffer = ~[tup(next._0, sp)] + buffer;
            }
            ret buffer.(distance-1u)._0;
        }
        fn fatal(str m) -> ! {
            codemap::emit_error(some(self.get_span()), m, sess.cm);
            fail;
        }
        fn warn(str m) {
            codemap::emit_warning(some(self.get_span()), m, sess.cm);
        }
        fn restrict(restriction r) { restr = r; }
        fn get_restriction() -> restriction { ret restr; }
        fn get_span() -> span { ret tok_span; }
        fn get_lo_pos() -> uint { ret tok_span.lo; }
        fn get_hi_pos() -> uint { ret tok_span.hi; }
        fn get_last_lo_pos() -> uint { ret last_tok_span.lo; }
        fn get_last_hi_pos() -> uint { ret last_tok_span.hi; }
        fn get_file_type() -> file_type { ret ftype; }
        fn get_cfg() -> ast::crate_cfg { ret cfg; }
        fn get_prec_table() -> @op_spec[] { ret precs; }
        fn get_str(token::str_num i) -> str {
            ret interner::get(*rdr.get_interner(), i);
        }
        fn get_reader() -> lexer::reader { ret rdr; }
        fn get_filemap() -> codemap::filemap { ret rdr.get_filemap(); }
        fn get_bad_expr_words() -> hashmap[str, ()] { ret bad_words; }
        fn get_chpos() -> uint { ret rdr.get_chpos(); }
        fn get_byte_pos() -> uint { ret rdr.get_byte_pos(); }
        fn get_id() -> node_id { ret next_node_id(sess); }
        fn get_sess() -> parse_sess { ret sess; }
    }

    auto tok0 = lexer::next_token(rdr);
    auto span0 = rec(lo=tok0._1, hi=rdr.get_chpos());
    ret stdio_parser(sess, cfg, ftype, tok0._0,
                     span0, span0, ~[], UNRESTRICTED, rdr,
                     prec_table(), bad_expr_word_table());
}

// These are the words that shouldn't be allowed as value identifiers,
// because, if used at the start of a line, they will cause the line to be
// interpreted as a specific kind of statement, which would be confusing.
fn bad_expr_word_table() -> hashmap[str, ()] {
    auto words = new_str_hash();
    words.insert("mod", ());
    words.insert("if", ());
    words.insert("else", ());
    words.insert("while", ());
    words.insert("do", ());
    words.insert("alt", ());
    words.insert("for", ());
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
    words.insert("state", ());
    words.insert("gc", ());
    words.insert("native", ());
    words.insert("auto", ());
    words.insert("fn", ());
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
    ret words;
}

fn unexpected(&parser p, token::token t) -> ! {
    let str s = "unexpected token: ";
    s += token::to_str(p.get_reader(), t);
    p.fatal(s);
}

fn expect(&parser p, token::token t) {
    if (p.peek() == t) {
        p.bump();
    } else {
        let str s = "expecting ";
        s += token::to_str(p.get_reader(), t);
        s += ", found ";
        s += token::to_str(p.get_reader(), p.peek());
        p.fatal(s);
    }
}

fn spanned[T](uint lo, uint hi, &T node) -> spanned[T] {
    ret rec(node=node, span=rec(lo=lo, hi=hi));
}

fn parse_ident(&parser p) -> ast::ident {
    alt (p.peek()) {
        case (token::IDENT(?i, _)) { p.bump(); ret p.get_str(i); }
        case (_) { p.fatal("expecting ident"); fail; }
    }
}

fn parse_value_ident(&parser p) -> ast::ident {
    check_bad_word(p);
    ret parse_ident(p);
}

fn is_word(&parser p, &str word) -> bool {
    ret alt (p.peek()) {
            case (token::IDENT(?sid, false)) { str::eq(word, p.get_str(sid)) }
            case (_) { false }
        };
}

fn eat_word(&parser p, &str word) -> bool {
    alt (p.peek()) {
        case (token::IDENT(?sid, false)) {
            if (str::eq(word, p.get_str(sid))) {
                p.bump();
                ret true;
            } else { ret false; }
        }
        case (_) { ret false; }
    }
}

fn expect_word(&parser p, &str word) {
    if (!eat_word(p, word)) {
        p.fatal("expecting " + word + ", found " +
                  token::to_str(p.get_reader(), p.peek()));
    }
}

fn check_bad_word(&parser p) {
    alt (p.peek()) {
        case (token::IDENT(?sid, false)) {
            auto w = p.get_str(sid);
            if (p.get_bad_expr_words().contains_key(w)) {
                p.fatal("found " + w + " in expression position");
            }
        }
        case (_) { }
    }
}

fn parse_ty_fn(ast::proto proto, &parser p, uint lo) -> ast::ty_ {
    fn parse_fn_input_ty(&parser p) -> ast::ty_arg {
        auto lo = p.get_lo_pos();
        auto mode = ast::val;
        if (p.peek() == token::BINOP(token::AND)) {
            p.bump();
            mode = ast::alias(eat_word(p, "mutable"));
        }
        auto t = parse_ty(p);
        alt (p.peek()) {
            case (token::IDENT(_, _)) { p.bump();/* ignore param name */ }
            case (_) {/* no param name present */ }
        }
        ret spanned(lo, t.span.hi, rec(mode=mode, ty=t));
    }
    auto lo = p.get_lo_pos();
    auto inputs =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                       parse_fn_input_ty, p);
    // FIXME: there's no syntax for this right now anyway
    //  auto constrs = parse_constrs(~[], p);
    let (@ast::constr)[] constrs = ~[];
    let @ast::ty output;
    auto cf = ast::return;
    if (p.peek() == token::RARROW) {
        p.bump();
        auto tmp = parse_ty_or_bang(p);
        alt (tmp) {
            case (a_ty(?t)) { output = t; }
            case (a_bang) {
                output = @spanned(lo, inputs.span.hi, ast::ty_bot);
                cf = ast::noreturn;
            }
        }
    } else { output = @spanned(lo, inputs.span.hi, ast::ty_nil); }
    ret ast::ty_fn(proto, inputs.node, output, cf, constrs);
}

fn parse_proto(&parser p) -> ast::proto {
    if (eat_word(p, "iter")) {
        ret ast::proto_iter;
    } else if (eat_word(p, "fn")) {
        ret ast::proto_fn;
    } else if (eat_word(p, "pred")) {
        ret ast::proto_fn;
    } else { unexpected(p, p.peek()); }
}

fn parse_ty_obj(&parser p, &mutable uint hi) -> ast::ty_ {
    fn parse_method_sig(&parser p) -> ast::ty_method {
        auto flo = p.get_lo_pos();
        let ast::proto proto = parse_proto(p);
        auto ident = parse_value_ident(p);
        auto f = parse_ty_fn(proto, p, flo);
        expect(p, token::SEMI);
        alt (f) {
            case (ast::ty_fn(?proto, ?inputs, ?output, ?cf, ?constrs)) {
                ret spanned(flo, output.span.hi,
                            rec(proto=proto,
                                ident=ident,
                                inputs=inputs,
                                output=output,
                                cf=cf,
                                constrs=constrs));
            }
        }
        fail;
    }
    auto meths = parse_seq(token::LBRACE, token::RBRACE, none,
                           parse_method_sig, p);
    hi = meths.span.hi;
    ret ast::ty_obj(meths.node);
}

fn parse_mt(&parser p) -> ast::mt {
    auto mut = parse_mutability(p);
    auto t = parse_ty(p);
    ret rec(ty=t, mut=mut);
}

fn parse_ty_field(&parser p) -> ast::ty_field {
    auto lo = p.get_lo_pos();
    auto mt = parse_mt(p);
    auto id = parse_ident(p);
    ret spanned(lo, mt.ty.span.hi, rec(ident=id, mt=mt));
}

// FIXME rename to parse_ty_field once the other one is dropped
fn parse_ty_field_modern(&parser p) -> ast::ty_field {
    auto lo = p.get_lo_pos();
    auto mut = parse_mutability(p);
    auto id = parse_ident(p);
    expect(p, token::COLON);
    auto ty = parse_ty(p);
    ret spanned(lo, ty.span.hi, rec(ident=id, mt=rec(ty=ty, mut=mut)));
}

// if i is the jth ident in args, return j
// otherwise, fail
fn ident_index(&parser p, &ast::arg[] args, &ast::ident i) -> uint {
    auto j = 0u;
    for (ast::arg a in args) { if (a.ident == i) { ret j; } j += 1u; }
    p.fatal("Unbound variable " + i + " in constraint arg");
}

fn parse_type_constr_arg(&parser p) -> @ast::ty_constr_arg {
    auto sp = p.get_span();
    auto carg = ast::carg_base;
    expect(p, token::BINOP(token::STAR));
    if (p.peek() == token::DOT) {
        // "*..." notation for record fields
        p.bump();
        let ast::path pth = parse_path(p);
        carg = ast::carg_ident(pth);
    }
    // No literals yet, I guess?
    ret @rec(node=carg, span=sp);
}

fn parse_constr_arg(&ast::arg[] args, &parser p) -> @ast::constr_arg {
    auto sp = p.get_span();
    auto carg = ast::carg_base;
    if (p.peek() == token::BINOP(token::STAR)) {
        p.bump();
    } else {
        let ast::ident i = parse_value_ident(p);
        carg = ast::carg_ident(ident_index(p, args, i));
    }
    ret @rec(node=carg, span=sp);
}

fn parse_ty_constr(&ast::arg[] fn_args, &parser p) -> @ast::constr {
    auto lo = p.get_lo_pos();
    auto path = parse_path(p);
    auto pf = bind parse_constr_arg(fn_args, _);
    let rec((@ast::constr_arg)[] node, span span) args =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA), pf,
                       p);
    ret @spanned(lo, args.span.hi,
                 rec(path=path, args=args.node, id=p.get_id()));
}

fn parse_constr_in_type(&parser p) -> @ast::ty_constr {
    auto lo = p.get_lo_pos();
    auto path = parse_path(p);
    let (@ast::ty_constr_arg)[] args =
        (parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                        parse_type_constr_arg, p)).node;
    auto hi = p.get_lo_pos();
    let ast::ty_constr_ tc = rec(path=path, args=args, id=p.get_id());
    ret @spanned(lo, hi, tc);
}


fn parse_constrs[T](fn(&parser p) ->
                    (@ast::constr_general[T]) pser, &parser p)
    ->  (@ast::constr_general[T])[] {
    let (@ast::constr_general[T])[] constrs = ~[];
    while (true) {
        auto constr = pser(p);
        constrs += ~[constr];
        if (p.peek() == token::COMMA) { p.bump(); } else { break; }
    }
    constrs
}

fn parse_type_constraints(&parser p) -> (@ast::ty_constr)[] {
    ret parse_constrs(parse_constr_in_type, p);
}

fn parse_ty_postfix(ast::ty_ orig_t, &parser p) -> @ast::ty {
    auto lo = p.get_lo_pos();
    if (p.peek() == token::LBRACKET) {
        p.bump();

        auto mut;
        if (eat_word(p, "mutable")) {
            if (p.peek() == token::QUES) {
                p.bump();
                mut = ast::maybe_mut;
            } else {
                mut = ast::mut;
            }
        } else {
            mut = ast::imm;
        }

        if (mut == ast::imm && p.peek() != token::RBRACKET) {
            // This is explicit type parameter instantiation.
            auto seq = parse_seq_to_end(token::RBRACKET,
                                        some(token::COMMA), parse_ty, p);

            alt (orig_t) {
                case (ast::ty_path(?pth, ?ann)) {
                    auto hi = p.get_hi_pos();
                    ret @spanned(lo, hi,
                                 ast::ty_path(spanned(lo, hi,
                                              rec(global=pth.node.global,
                                                  idents=pth.node.idents,
                                                  types=seq)),
                                              ann));
                }
                case (_) {
                    p.fatal("type parameter instantiation only allowed for " +
                          "paths");
                }
            }
        }

        expect(p, token::RBRACKET);
        auto hi = p.get_hi_pos();
        // FIXME: spans are probably wrong
        auto t = ast::ty_ivec(rec(ty=@spanned(lo, hi, orig_t), mut=mut));
        ret parse_ty_postfix(t, p);
    }
    ret @spanned(lo, p.get_lo_pos(), orig_t);
}

fn parse_ty_or_bang(&parser p) -> ty_or_bang {
    alt (p.peek()) {
        case (token::NOT) { p.bump(); ret a_bang; }
        case (_) { ret a_ty(parse_ty(p)); }
    }
}

fn parse_ty(&parser p) -> @ast::ty {
    auto lo = p.get_lo_pos();
    auto hi = lo;
    let ast::ty_ t;
    // FIXME: do something with this

    parse_layer(p);
    if (eat_word(p, "bool")) {
        t = ast::ty_bool;
    } else if (eat_word(p, "int")) {
        t = ast::ty_int;
    } else if (eat_word(p, "uint")) {
        t = ast::ty_uint;
    } else if (eat_word(p, "float")) {
        t = ast::ty_float;
    } else if (eat_word(p, "str")) {
        t = ast::ty_str;
    } else if (eat_word(p, "istr")) {
        t = ast::ty_istr;
    } else if (eat_word(p, "char")) {
        t = ast::ty_char;
    } else if (eat_word(p, "task")) {
        t = ast::ty_task;
    } else if (eat_word(p, "i8")) {
        t = ast::ty_machine(ast::ty_i8);
    } else if (eat_word(p, "i16")) {
        t = ast::ty_machine(ast::ty_i16);
    } else if (eat_word(p, "i32")) {
        t = ast::ty_machine(ast::ty_i32);
    } else if (eat_word(p, "i64")) {
        t = ast::ty_machine(ast::ty_i64);
    } else if (eat_word(p, "u8")) {
        t = ast::ty_machine(ast::ty_u8);
    } else if (eat_word(p, "u16")) {
        t = ast::ty_machine(ast::ty_u16);
    } else if (eat_word(p, "u32")) {
        t = ast::ty_machine(ast::ty_u32);
    } else if (eat_word(p, "u64")) {
        t = ast::ty_machine(ast::ty_u64);
    } else if (eat_word(p, "f32")) {
        t = ast::ty_machine(ast::ty_f32);
    } else if (eat_word(p, "f64")) {
        t = ast::ty_machine(ast::ty_f64);
    } else if (p.peek() == token::LPAREN) {
        p.bump();
        alt (p.peek()) {
            case (token::RPAREN) {
                hi = p.get_hi_pos();
                p.bump();
                t = ast::ty_nil;
            }
            case (_) {
                t = parse_ty(p).node;
                hi = p.get_hi_pos();
                expect(p, token::RPAREN);
            }
        }
    } else if (p.peek() == token::AT) {
        p.bump();
        auto mt = parse_mt(p);
        hi = mt.ty.span.hi;
        t = ast::ty_box(mt);
    } else if (p.peek() == token::BINOP(token::STAR)) {
        p.bump();
        auto mt = parse_mt(p);
        hi = mt.ty.span.hi;
        t = ast::ty_ptr(mt);
    } else if (p.peek() == token::LBRACE) {
        auto elems = parse_seq(token::LBRACE, token::RBRACE,
                               some(token::COMMA), parse_ty_field_modern, p);
        hi = elems.span.hi;
        t = ast::ty_rec(elems.node);
        if (p.peek() == token::COLON) {
            p.bump();
            t = ast::ty_constr(@spanned(lo, hi, t),
                               parse_type_constraints(p));
        }
    } else if (eat_word(p, "vec")) {
        expect(p, token::LBRACKET);
        t = ast::ty_vec(parse_mt(p));
        hi = p.get_hi_pos();
        expect(p, token::RBRACKET);
    } else if (eat_word(p, "tup")) {
        auto elems =
            parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                           parse_mt, p);
        hi = elems.span.hi;
        t = ast::ty_tup(elems.node);
    } else if (eat_word(p, "rec")) {
        auto elems =
            parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                           parse_ty_field, p);
        hi = elems.span.hi;
        // possible constrs
        // FIXME: something seems dodgy or at least repetitive
        // about how constrained types get parsed
        t = ast::ty_rec(elems.node);
        if (p.peek() == token::COLON) {
            p.bump();
            t = ast::ty_constr(@spanned(lo, hi, t),
                               parse_type_constraints(p));
        }
    } else if (eat_word(p, "fn")) {
        auto flo = p.get_last_lo_pos();
        t = parse_ty_fn(ast::proto_fn, p, flo);
        alt (t) { case (ast::ty_fn(_, _, ?out, _, _)) { hi = out.span.hi; } }
    } else if (eat_word(p, "iter")) {
        auto flo = p.get_last_lo_pos();
        t = parse_ty_fn(ast::proto_iter, p, flo);
        alt (t) { case (ast::ty_fn(_, _, ?out, _, _)) { hi = out.span.hi; } }
    } else if (eat_word(p, "obj")) {
        t = parse_ty_obj(p, hi);
    } else if (eat_word(p, "port")) {
        expect(p, token::LBRACKET);
        t = ast::ty_port(parse_ty(p));
        hi = p.get_hi_pos();
        expect(p, token::RBRACKET);
    } else if (eat_word(p, "chan")) {
        expect(p, token::LBRACKET);
        t = ast::ty_chan(parse_ty(p));
        hi = p.get_hi_pos();
        expect(p, token::RBRACKET);
    } else if (eat_word(p, "mutable")) {
        p.warn("ignoring deprecated 'mutable' type constructor");
        auto typ = parse_ty(p);
        t = typ.node;
        hi = typ.span.hi;
    } else if (p.peek() == token::MOD_SEP || is_ident(p.peek())) {
        auto path = parse_path(p);
        t = ast::ty_path(path, p.get_id());
        hi = path.span.hi;
    } else { p.fatal("expecting type"); t = ast::ty_nil; fail; }
    ret parse_ty_postfix(t, p);
}

fn parse_arg(&parser p) -> ast::arg {
    let ast::mode m = ast::val;
    if (p.peek() == token::BINOP(token::AND)) {
        p.bump();
        m = ast::alias(eat_word(p, "mutable"));
    }
    let @ast::ty t = parse_ty(p);
    let ast::ident i = parse_value_ident(p);
    ret rec(mode=m, ty=t, ident=i, id=p.get_id());
}

fn parse_seq_to_end[T](token::token ket, option::t[token::token] sep,
                       fn(&parser)->T  f, &parser p) -> T[] {
    auto val = parse_seq_to_before_end(ket, sep, f, p);
    p.bump();
    ret val;
}

fn parse_seq_to_before_end[T](token::token ket, option::t[token::token] sep,
                              fn(&parser)->T  f, &parser p) -> T[] {
    let bool first = true;
    let T[] v = ~[];
    while (p.peek() != ket) {
        alt (sep) {
            case (some(?t)) {
                if (first) { first = false; } else { expect(p, t); }
            }
            case (_) { }
        }
        v += ~[f(p)];
    }
    ret v;
}


fn parse_seq[T](token::token bra, token::token ket,
                     option::t[token::token] sep,
                     fn(&parser)->T  f, &parser p) -> spanned[T[]] {
    auto lo = p.get_lo_pos();
    expect(p, bra);
    auto result = parse_seq_to_before_end[T](ket, sep, f, p);
    auto hi = p.get_hi_pos();
    p.bump();
    ret spanned(lo, hi, result);
}


fn parse_lit(&parser p) -> ast::lit {
    auto sp = p.get_span();
    let ast::lit_ lit = ast::lit_nil;
    if (eat_word(p, "true")) {
        lit = ast::lit_bool(true);
    } else if (eat_word(p, "false")) {
        lit = ast::lit_bool(false);
    } else {
        alt (p.peek()) {
            case (token::LIT_INT(?i)) { p.bump(); lit = ast::lit_int(i); }
            case (token::LIT_UINT(?u)) { p.bump(); lit = ast::lit_uint(u); }
            case (token::LIT_FLOAT(?s)) {
                p.bump();
                lit = ast::lit_float(p.get_str(s));
            }
            case (token::LIT_MACH_INT(?tm, ?i)) {
                p.bump();
                lit = ast::lit_mach_int(tm, i);
            }
            case (token::LIT_MACH_FLOAT(?tm, ?s)) {
                p.bump();
                lit = ast::lit_mach_float(tm, p.get_str(s));
            }
            case (token::LIT_CHAR(?c)) { p.bump(); lit = ast::lit_char(c); }
            case (token::LIT_STR(?s)) {
                p.bump();
                lit = ast::lit_str(p.get_str(s), ast::sk_rc);
            }
            case (token::LPAREN) {
                p.bump();
                expect(p, token::RPAREN);
                lit = ast::lit_nil;
            }
            case (?t) { unexpected(p, t); }
        }
    }
    ret rec(node=lit, span=sp);
}

fn is_ident(token::token t) -> bool {
    alt (t) { case (token::IDENT(_, _)) { ret true; } case (_) { } }
    ret false;
}

fn parse_path(&parser p) -> ast::path {
    auto lo = p.get_lo_pos();
    auto hi = lo;

    auto global;
    if (p.peek() == token::MOD_SEP) {
        global = true; p.bump();
    } else {
        global = false;
    }

    let ast::ident[] ids = ~[];
    while (true) {
        alt (p.peek()) {
            case (token::IDENT(?i, _)) {
                hi = p.get_hi_pos();
                ids += ~[p.get_str(i)];
                hi = p.get_hi_pos();
                p.bump();
                if (p.peek() == token::MOD_SEP) { p.bump(); } else { break; }
            }
            case (_) { break; }
        }
    }
    ret spanned(lo, hi, rec(global=global, idents=ids, types=~[]));
}

fn parse_path_and_ty_param_substs(&parser p) -> ast::path {
    auto lo = p.get_lo_pos();
    auto path = parse_path(p);
    if (p.peek() == token::LBRACKET) {
        auto seq = parse_seq(token::LBRACKET, token::RBRACKET,
                             some(token::COMMA), parse_ty, p);
        auto hi = seq.span.hi;
        path = spanned(lo, hi, rec(global=path.node.global,
                                   idents=path.node.idents,
                                   types=seq.node));
    }
    ret path;
}

fn parse_mutability(&parser p) -> ast::mutability {
    if (eat_word(p, "mutable")) {
        if (p.peek() == token::QUES) { p.bump(); ret ast::maybe_mut; }
        ret ast::mut;
    }
    ret ast::imm;
}

fn parse_field(&parser p, &token::token sep) -> ast::field {
    auto lo = p.get_lo_pos();
    auto m = parse_mutability(p);
    auto i = parse_ident(p);
    expect(p, sep);
    auto e = parse_expr(p);
    ret spanned(lo, e.span.hi, rec(mut=m, ident=i, expr=e));
}

fn mk_expr(&parser p, uint lo, uint hi, &ast::expr_ node) -> @ast::expr {
    ret @rec(id=p.get_id(),
             node=node,
             span=rec(lo=lo, hi=hi));
}

fn mk_mac_expr(&parser p, uint lo, uint hi, &ast::mac_ m) -> @ast::expr {
    ret @rec(id=p.get_id(),
             node=ast::expr_mac(rec(node=m, span=rec(lo=lo, hi=hi))),
             span=rec(lo=lo, hi=hi));
}

fn parse_bottom_expr(&parser p) -> @ast::expr {
    auto lo = p.get_lo_pos();
    auto hi = p.get_hi_pos();
    // FIXME: can only remove this sort of thing when both typestate and
    // alt-exhaustive-match checking are co-operating.

    auto lit = @spanned(lo, hi, ast::lit_nil);
    let ast::expr_ ex = ast::expr_lit(lit);
    if (p.peek() == token::LPAREN) {
        p.bump();
        alt (p.peek()) {
            case (token::RPAREN) {
                hi = p.get_hi_pos();
                p.bump();
                auto lit = @spanned(lo, hi, ast::lit_nil);
                ret mk_expr(p, lo, hi, ast::expr_lit(lit));
            }
            case (_) {/* fall through */ }
        }
        auto e = parse_expr(p);
        hi = p.get_hi_pos();
        expect(p, token::RPAREN);
        ret mk_expr(p, lo, hi, e.node);
    } else if (p.peek() == token::LBRACE) {
        p.bump();
        if (is_word(p, "mutable") ||
            alt p.peek() { token::IDENT(_, false) { true } _ { false } } &&
            p.look_ahead(1u) == token::COLON) {
            auto fields = ~[parse_field(p, token::COLON)];
            auto base = none;
            while p.peek() != token::RBRACE {
                if eat_word(p, "with") {
                    base = some(parse_expr(p));
                    break;
                }
                expect(p, token::COMMA);
                fields += ~[parse_field(p, token::COLON)];
            }
            hi = p.get_hi_pos();
            expect(p, token::RBRACE);
            ex = ast::expr_rec(fields, base);
        } else {
            auto blk = parse_block_tail(p, lo);
            ret mk_expr(p, blk.span.lo, blk.span.hi, ast::expr_block(blk));
        }
    } else if (eat_word(p, "if")) {
        ret parse_if_expr(p);
    } else if (eat_word(p, "for")) {
        ret parse_for_expr(p);
    } else if (eat_word(p, "while")) {
        ret parse_while_expr(p);
    } else if (eat_word(p, "do")) {
        ret parse_do_while_expr(p);
    } else if (eat_word(p, "alt")) {
        ret parse_alt_expr(p);
    } else if (eat_word(p, "spawn")) {
        ret parse_spawn_expr(p);
    } else if (eat_word(p, "fn")) {
        ret parse_fn_expr(p);
    } else if (eat_word(p, "tup")) {
        fn parse_elt(&parser p) -> ast::elt {
            auto m = parse_mutability(p);
            auto e = parse_expr(p);
            ret rec(mut=m, expr=e);
        }
        auto es = parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                            parse_elt, p);
        hi = es.span.hi;
        ex = ast::expr_tup(es.node);
    } else if (p.peek() == token::LBRACKET) {
        p.bump();
        auto mut = parse_mutability(p);
        auto es = parse_seq_to_end(token::RBRACKET, some(token::COMMA),
                                   parse_expr, p);
        ex = ast::expr_vec(es, mut, ast::sk_rc);
    } else if (p.peek() == token::POUND_LT) {
        p.bump();
        auto ty = parse_ty(p);
        expect(p, token::GT);
        /* hack: early return to take advantage of specialized function */
        ret mk_mac_expr(p, lo, p.get_hi_pos(), ast::mac_embed_type(ty))
    } else if (p.peek() == token::POUND_LBRACE) {
        p.bump();
        auto blk = ast::mac_embed_block(parse_block_tail(p, lo));
        ret mk_mac_expr(p, lo, p.get_hi_pos(), blk);
    } else if (p.peek() == token::ELLIPSIS) {
        p.bump();
        ret mk_mac_expr(p, lo, p.get_hi_pos(), ast::mac_ellipsis)
    } else if (p.peek() == token::TILDE) {
        p.bump();
        alt (p.peek()) {
            case (token::LBRACKET) { // unique array (temporary)
                p.bump();
                auto mut = parse_mutability(p);
                auto es = parse_seq_to_end
                    (token::RBRACKET, some(token::COMMA), parse_expr, p);
                ex = ast::expr_vec(es, mut, ast::sk_unique);
            }
            case (token::LIT_STR(?s)) {
                p.bump();
                auto lit =
                    @rec(node=ast::lit_str(p.get_str(s), ast::sk_unique),
                         span=p.get_span());
                ex = ast::expr_lit(lit);
            }
            case (_) {
                p.fatal("unimplemented: unique pointer creation");
            }
        }
    } else if (eat_word(p, "obj")) {
        // Anonymous object

        // Only make people type () if they're actually adding new fields
        let option::t[ast::anon_obj_field[]] fields = none;
        if (p.peek() == token::LPAREN) {
            p.bump();
            fields =
                some(parse_seq_to_end(token::RPAREN, some(token::COMMA),
                                           parse_anon_obj_field, p));
        }
        let (@ast::method)[] meths = ~[];
        let option::t[@ast::expr] with_obj = none;
        expect(p, token::LBRACE);
        while (p.peek() != token::RBRACE) {
            if (eat_word(p, "with")) {
                with_obj = some(parse_expr(p));
            } else {
                meths += ~[parse_method(p)];
            }
        }
        hi = p.get_hi_pos();
        expect(p, token::RBRACE);
        // fields and methods may be *additional* or *overriding* fields
        // and methods if there's a with_obj, or they may be the *only*
        // fields and methods if there's no with_obj.

        // We don't need to pull ".node" out of fields because it's not a
        // "spanned".
        let ast::anon_obj ob =
            rec(fields=fields, methods=meths, with_obj=with_obj);
        ex = ast::expr_anon_obj(ob);
    } else if (eat_word(p, "rec")) {
        expect(p, token::LPAREN);
        auto fields = ~[parse_field(p, token::EQ)];
        auto more = true;
        auto base = none;
        while (more) {
            if (p.peek() == token::RPAREN) {
                hi = p.get_hi_pos();
                p.bump();
                more = false;
            } else if (eat_word(p, "with")) {
                base = some(parse_expr(p));
                hi = p.get_hi_pos();
                expect(p, token::RPAREN);
                more = false;
            } else if (p.peek() == token::COMMA) {
                p.bump();
                fields += ~[parse_field(p, token::EQ)];
            } else { unexpected(p, p.peek()); }
        }
        ex = ast::expr_rec(fields, base);
    } else if (eat_word(p, "bind")) {
        auto e = parse_expr_res(p, RESTRICT_NO_CALL_EXPRS);
        fn parse_expr_opt(&parser p) -> option::t[@ast::expr] {
            alt (p.peek()) {
                case (token::UNDERSCORE) { p.bump(); ret none; }
                case (_) { ret some(parse_expr(p)); }
            }
        }
        auto es = parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                            parse_expr_opt, p);
        hi = es.span.hi;
        ex = ast::expr_bind(e, es.node);
    } else if (p.peek() == token::POUND) {
        auto ex_ext = parse_syntax_ext(p);
        hi = ex_ext.span.hi;
        ex = ex_ext.node;
    } else if (eat_word(p, "fail")) {
        if (can_begin_expr(p.peek())) {
            auto e = parse_expr(p);
            hi = e.span.hi;
            ex = ast::expr_fail(some(e));
        }
        else {
            ex = ast::expr_fail(none);
        }
    } else if (eat_word(p, "log")) {
        auto e = parse_expr(p);
        ex = ast::expr_log(1, e);
        hi = e.span.hi;
    } else if (eat_word(p, "log_err")) {
        auto e = parse_expr(p);
        ex = ast::expr_log(0, e);
        hi = e.span.hi;
    } else if (eat_word(p, "assert")) {
        auto e = parse_expr(p);
        ex = ast::expr_assert(e);
        hi = e.span.hi;
    } else if (eat_word(p, "check")) {
        /* Should be a predicate (pure boolean function) applied to
           arguments that are all either slot variables or literals.
           but the typechecker enforces that. */

        auto e = parse_expr(p);
        hi = e.span.hi;
        ex = ast::expr_check(ast::checked, e);
    } else if (eat_word(p, "claim")) {
        /* Same rules as check, except that if check-claims
         is enabled (a command-line flag), then the parser turns
        claims into check */

        auto e = parse_expr(p);
        hi = e.span.hi;
        ex = ast::expr_check(ast::unchecked, e);
    } else if (eat_word(p, "ret")) {
        if (can_begin_expr(p.peek())) {
            auto e = parse_expr(p);
            hi = e.span.hi;
            ex = ast::expr_ret(some(e));
        }
        else {
            ex = ast::expr_ret(none);
        }
    } else if (eat_word(p, "break")) {
        ex = ast::expr_break;
        hi = p.get_hi_pos();
    } else if (eat_word(p, "cont")) {
        ex = ast::expr_cont;
        hi = p.get_hi_pos();
    } else if (eat_word(p, "put")) {
        alt (p.peek()) {
            case (token::SEMI) { ex = ast::expr_put(none); }
            case (_) {
                auto e = parse_expr(p);
                hi = e.span.hi;
                ex = ast::expr_put(some(e));
            }
        }
    } else if (eat_word(p, "be")) {
        auto e = parse_expr(p);

        // FIXME: Is this the right place for this check?
        if (/*check*/ast::is_call_expr(e)) {
            hi = e.span.hi;
            ex = ast::expr_be(e);
        } else { p.fatal("Non-call expression in tail call"); }
    } else if (eat_word(p, "port")) {
        auto ty = none;
        if(token::LBRACKET == p.peek()) {
            expect(p, token::LBRACKET);
            ty = some(parse_ty(p));
            expect(p, token::RBRACKET);
        }
        expect(p, token::LPAREN);
        expect(p, token::RPAREN);
        hi = p.get_hi_pos();
        ex = ast::expr_port(ty);
    } else if (eat_word(p, "chan")) {
        expect(p, token::LPAREN);
        auto e = parse_expr(p);
        hi = e.span.hi;
        expect(p, token::RPAREN);
        ex = ast::expr_chan(e);
    } else if (eat_word(p, "self")) {
        log "parsing a self-call...";
        expect(p, token::DOT);
        // The rest is a call expression.

        let @ast::expr f = parse_self_method(p);
        auto es = parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                            parse_expr, p);
        hi = es.span.hi;
        ex = ast::expr_call(f, es.node);
    } else if (p.peek() == token::MOD_SEP ||
               (is_ident(p.peek()) && !is_word(p, "true") &&
                !is_word(p, "false"))) {
        check_bad_word(p);
        auto pth = parse_path_and_ty_param_substs(p);
        hi = pth.span.hi;
        ex = ast::expr_path(pth);
    } else {
        auto lit = parse_lit(p);
        hi = lit.span.hi;
        ex = ast::expr_lit(@lit);
    }
    ret mk_expr(p, lo, hi, ex);
}

fn parse_syntax_ext(&parser p) -> @ast::expr {
    auto lo = p.get_lo_pos();
    expect(p, token::POUND);
    ret parse_syntax_ext_naked(p, lo);
}

fn parse_syntax_ext_naked(&parser p, uint lo) -> @ast::expr {
    auto pth = parse_path(p);
    if (ivec::len(pth.node.idents) == 0u) {
        p.fatal("expected a syntax expander name");
    }
    auto es = parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                        parse_expr, p);
    auto hi = es.span.hi;
    ret mk_mac_expr(p, lo, hi, ast::mac_invoc(pth, es.node, none));
}

fn parse_self_method(&parser p) -> @ast::expr {
    auto sp = p.get_span();
    let ast::ident f_name = parse_ident(p);
    ret mk_expr(p, sp.lo, sp.hi, ast::expr_self_method(f_name));
}

fn parse_dot_or_call_expr(&parser p) -> @ast::expr {
    ret parse_dot_or_call_expr_with(p, parse_bottom_expr(p));
}

fn parse_dot_or_call_expr_with(&parser p, @ast::expr e) -> @ast::expr {
    auto lo = e.span.lo;
    auto hi = e.span.hi;
    while (true) {
        alt (p.peek()) {
            case (token::LPAREN) {
                if (p.get_restriction() == RESTRICT_NO_CALL_EXPRS) {
                    ret e;
                } else {
                    // Call expr.

                    auto es = parse_seq(token::LPAREN, token::RPAREN,
                                        some(token::COMMA), parse_expr, p);
                    hi = es.span.hi;
                    e = mk_expr(p, lo, hi, ast::expr_call(e, es.node));
                }
            }
            case (token::DOT) {
                p.bump();
                alt (p.peek()) {
                    case (token::IDENT(?i, _)) {
                        hi = p.get_hi_pos();
                        p.bump();
                        e = mk_expr(p, lo, hi,
                                    ast::expr_field(e, p.get_str(i)));
                    }
                    case (token::LPAREN) {
                        p.bump();
                        auto ix = parse_expr(p);
                        hi = ix.span.hi;
                        expect(p, token::RPAREN);
                        e = mk_expr(p, lo, hi, ast::expr_index(e, ix));
                    }
                    case (?t) { unexpected(p, t); }
                }
            }
            case (_) { ret e; }
        }
    }
    ret e;
}

fn parse_prefix_expr(&parser p) -> @ast::expr {
    if (eat_word(p, "mutable")) {
        p.warn("ignoring deprecated 'mutable' prefix operator");
    }
    auto lo = p.get_lo_pos();
    auto hi = p.get_hi_pos();
    // FIXME: can only remove this sort of thing when both typestate and
    // alt-exhaustive-match checking are co-operating.

    auto lit = @spanned(lo, lo, ast::lit_nil);
    let ast::expr_ ex = ast::expr_lit(lit);
    alt (p.peek()) {
        case (token::NOT) {
            p.bump();
            auto e = parse_prefix_expr(p);
            hi = e.span.hi;
            ex = ast::expr_unary(ast::not, e);
        }
        case (token::BINOP(?b)) {
            alt (b) {
                case (token::MINUS) {
                    p.bump();
                    auto e = parse_prefix_expr(p);
                    hi = e.span.hi;
                    ex = ast::expr_unary(ast::neg, e);
                }
                case (token::STAR) {
                    p.bump();
                    auto e = parse_prefix_expr(p);
                    hi = e.span.hi;
                    ex = ast::expr_unary(ast::deref, e);
                }
                case (_) { ret parse_dot_or_call_expr(p); }
            }
        }
        case (token::AT) {
            p.bump();
            auto m = parse_mutability(p);
            auto e = parse_prefix_expr(p);
            hi = e.span.hi;
            ex = ast::expr_unary(ast::box(m), e);
        }
        case (_) { ret parse_dot_or_call_expr(p); }
    }
    ret mk_expr(p, lo, hi, ex);
}

fn parse_ternary(&parser p) -> @ast::expr {
    auto cond_expr = parse_binops(p);
    if (p.peek() == token::QUES) {
        p.bump();
        auto then_expr = parse_expr(p);
        expect(p, token::COLON);
        auto else_expr = parse_expr(p);
        ret mk_expr(p, cond_expr.span.lo, else_expr.span.hi,
                    ast::expr_ternary(cond_expr, then_expr, else_expr));
    } else {
        ret cond_expr;
    }
}

type op_spec = rec(token::token tok, ast::binop op, int prec);


// FIXME make this a const, don't store it in parser state
fn prec_table() -> @op_spec[] {
    ret @~[rec(tok=token::BINOP(token::STAR), op=ast::mul, prec=11),
           rec(tok=token::BINOP(token::SLASH), op=ast::div, prec=11),
           rec(tok=token::BINOP(token::PERCENT), op=ast::rem, prec=11),
           rec(tok=token::BINOP(token::PLUS), op=ast::add, prec=10),
           rec(tok=token::BINOP(token::MINUS), op=ast::sub, prec=10),
           rec(tok=token::BINOP(token::LSL), op=ast::lsl, prec=9),
           rec(tok=token::BINOP(token::LSR), op=ast::lsr, prec=9),
           rec(tok=token::BINOP(token::ASR), op=ast::asr, prec=9),
           rec(tok=token::BINOP(token::AND), op=ast::bitand, prec=8),
           rec(tok=token::BINOP(token::CARET), op=ast::bitxor, prec=6),
           rec(tok=token::BINOP(token::OR), op=ast::bitor, prec=6),
           // 'as' sits between here with 5
           rec(tok=token::LT, op=ast::lt, prec=4),
           rec(tok=token::LE, op=ast::le, prec=4),
           rec(tok=token::GE, op=ast::ge, prec=4),
           rec(tok=token::GT, op=ast::gt, prec=4),
           rec(tok=token::EQEQ, op=ast::eq, prec=3),
           rec(tok=token::NE, op=ast::ne, prec=3),
           rec(tok=token::ANDAND, op=ast::and, prec=2),
           rec(tok=token::OROR, op=ast::or, prec=1)];
}

fn parse_binops(&parser p) -> @ast::expr {
    ret parse_more_binops(p, parse_prefix_expr(p), 0);
}

const int unop_prec = 100;

const int as_prec = 5;
const int ternary_prec = 0;

fn parse_more_binops(&parser p, @ast::expr lhs, int min_prec) -> @ast::expr {
    auto peeked = p.peek();
    for (op_spec cur in *p.get_prec_table()) {
        if (cur.prec > min_prec && cur.tok == peeked) {
            p.bump();
            auto rhs = parse_more_binops(p, parse_prefix_expr(p), cur.prec);
            auto bin = mk_expr(p, lhs.span.lo, rhs.span.hi,
                               ast::expr_binary(cur.op, lhs, rhs));
            ret parse_more_binops(p, bin, min_prec);
        }
    }
    if (as_prec > min_prec && eat_word(p, "as")) {
        auto rhs = parse_ty(p);
        auto _as = mk_expr(p, lhs.span.lo, rhs.span.hi,
                           ast::expr_cast(lhs, rhs));
        ret parse_more_binops(p, _as, min_prec);
    }
    ret lhs;
}

fn parse_assign_expr(&parser p) -> @ast::expr {
    auto lo = p.get_lo_pos();
    auto lhs = parse_ternary(p);
    alt (p.peek()) {
        case (token::EQ) {
            p.bump();
            auto rhs = parse_expr(p);
            ret mk_expr(p, lo, rhs.span.hi, ast::expr_assign(lhs, rhs));
        }
        case (token::BINOPEQ(?op)) {
            p.bump();
            auto rhs = parse_expr(p);
            auto aop = ast::add;
            alt (op) {
                case (token::PLUS) { aop = ast::add; }
                case (token::MINUS) { aop = ast::sub; }
                case (token::STAR) { aop = ast::mul; }
                case (token::SLASH) { aop = ast::div; }
                case (token::PERCENT) { aop = ast::rem; }
                case (token::CARET) { aop = ast::bitxor; }
                case (token::AND) { aop = ast::bitand; }
                case (token::OR) { aop = ast::bitor; }
                case (token::LSL) { aop = ast::lsl; }
                case (token::LSR) { aop = ast::lsr; }
                case (token::ASR) { aop = ast::asr; }
            }
            ret mk_expr(p, lo, rhs.span.hi,
                        ast::expr_assign_op(aop, lhs, rhs));
        }
        case (token::LARROW) {
            p.bump();
            auto rhs = parse_expr(p);
            ret mk_expr(p, lo, rhs.span.hi, ast::expr_move(lhs, rhs));
        }
        case (token::SEND) {
            p.bump();
            auto rhs = parse_expr(p);
            ret mk_expr(p, lo, rhs.span.hi, ast::expr_send(lhs, rhs));
        }
        case (token::RECV) {
            p.bump();
            auto rhs = parse_expr(p);
            ret mk_expr(p, lo, rhs.span.hi, ast::expr_recv(lhs, rhs));
        }
        case (token::DARROW) {
            p.bump();
            auto rhs = parse_expr(p);
            ret mk_expr(p, lo, rhs.span.hi, ast::expr_swap(lhs, rhs));
        }
        case (_) {/* fall through */ }
    }
    ret lhs;
}

fn parse_if_expr_1(&parser p) -> tup(@ast::expr,
                                     ast::blk, option::t[@ast::expr],
                                     uint, uint) {
    auto lo = p.get_last_lo_pos();
    auto cond = parse_expr(p);
    auto thn = parse_block(p);
    let option::t[@ast::expr] els = none;
    auto hi = thn.span.hi;
    if (eat_word(p, "else")) {
        auto elexpr = parse_else_expr(p);
        els = some(elexpr);
        hi = elexpr.span.hi;
    }
    ret tup(cond, thn, els, lo, hi);
}

fn parse_if_expr(&parser p) -> @ast::expr {
    if (eat_word(p, "check")) {
            auto q = parse_if_expr_1(p);
            ret mk_expr(p, q._3, q._4, ast::expr_if_check(q._0, q._1, q._2));
    }
    else {
        auto q = parse_if_expr_1(p);
        ret mk_expr(p, q._3, q._4, ast::expr_if(q._0, q._1, q._2));
    }
}

fn parse_fn_expr(&parser p) -> @ast::expr {
    auto lo = p.get_last_lo_pos();
    auto decl = parse_fn_decl(p, ast::impure_fn);
    auto body = parse_block(p);
    auto _fn = rec(decl=decl, proto=ast::proto_fn, body=body);
    ret mk_expr(p, lo, body.span.hi, ast::expr_fn(_fn));
}

fn parse_else_expr(&parser p) -> @ast::expr {
    if (eat_word(p, "if")) {
        ret parse_if_expr(p);
    } else {
        auto blk = parse_block(p);
        ret mk_expr(p, blk.span.lo, blk.span.hi, ast::expr_block(blk));
    }
}

fn parse_head_local(&parser p) -> @ast::local {
    if (is_word(p, "auto")) {
        ret parse_auto_local(p);
    } else {
        ret parse_typed_local(p);
    }
}

fn parse_for_expr(&parser p) -> @ast::expr {
    auto lo = p.get_last_lo_pos();
    auto is_each = eat_word(p, "each");
    expect(p, token::LPAREN);
    auto decl = parse_head_local(p);
    expect_word(p, "in");
    auto seq = parse_expr(p);
    expect(p, token::RPAREN);
    auto body = parse_block(p);
    auto hi = body.span.hi;
    if (is_each) {
        ret mk_expr(p, lo, hi, ast::expr_for_each(decl, seq, body));
    } else {
        ret mk_expr(p, lo, hi, ast::expr_for(decl, seq, body));
    }
}

fn parse_while_expr(&parser p) -> @ast::expr {
    auto lo = p.get_last_lo_pos();
    auto cond = parse_expr(p);
    auto body = parse_block(p);
    auto hi = body.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_while(cond, body));
}

fn parse_do_while_expr(&parser p) -> @ast::expr {
    auto lo = p.get_last_lo_pos();
    auto body = parse_block(p);
    expect_word(p, "while");
    auto cond = parse_expr(p);
    auto hi = cond.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_do_while(body, cond));
}

fn parse_alt_expr(&parser p) -> @ast::expr {
    auto lo = p.get_last_lo_pos();
    auto discriminant = parse_expr(p);
    expect(p, token::LBRACE);
    let ast::arm[] arms = ~[];
    while (p.peek() != token::RBRACE) {
        // Optionally eat the case keyword.
        // FIXME remove this (and the optional parens) once we've updated our
        // code to not use the old syntax
        eat_word(p, "case");
        auto parens = false;
        if (p.peek() == token::LPAREN) { parens = true; p.bump(); }
        auto pats = parse_pats(p);
        if (parens) { expect(p, token::RPAREN); }
        auto blk = parse_block(p);
        arms += ~[rec(pats=pats, block=blk)];
    }
    auto hi = p.get_hi_pos();
    p.bump();
    ret mk_expr(p, lo, hi, ast::expr_alt(discriminant, arms));
}

fn parse_spawn_expr(&parser p) -> @ast::expr {
    auto lo = p.get_last_lo_pos();
    // FIXME: Parse domain and name
    // FIXME: why no full expr?

    auto fn_expr = parse_bottom_expr(p);
    auto es = parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                        parse_expr, p);
    auto hi = es.span.hi;
    ret mk_expr(p, lo, hi, ast::expr_spawn
                (ast::dom_implicit, option::none, fn_expr, es.node));
}

fn parse_expr(&parser p) -> @ast::expr {
    ret parse_expr_res(p, UNRESTRICTED);
}

fn parse_expr_res(&parser p, restriction r) -> @ast::expr {
    auto old = p.get_restriction();
    p.restrict(r);
    auto e = parse_assign_expr(p);
    p.restrict(old);
    ret e;
}

fn parse_initializer(&parser p) -> option::t[ast::initializer] {
    alt (p.peek()) {
        case (token::EQ) {
            p.bump();
            ret some(rec(op=ast::init_assign, expr=parse_expr(p)));
        }
        case (token::LARROW) {
            p.bump();
            ret some(rec(op=ast::init_move, expr=parse_expr(p)));
        }
        // Now that the the channel is the first argument to receive,
        // combining it with an initializer doesn't really make sense.
        // case (token::RECV) {
        //     p.bump();
        //     ret some(rec(op = ast::init_recv,
        //                  expr = parse_expr(p)));
        // }
        case (_) {
            ret none;
        }
    }
}

fn parse_pats(&parser p) -> (@ast::pat)[] {
    auto pats = ~[];
    while (true) {
        pats += ~[parse_pat(p)];
        if (p.peek() == token::BINOP(token::OR)) {
            p.bump();
        } else {
            break;
        }
    }
    ret pats;
}

fn parse_pat(&parser p) -> @ast::pat {
    auto lo = p.get_lo_pos();
    auto hi = p.get_hi_pos();
    auto pat;
    alt (p.peek()) {
        case (token::UNDERSCORE) {
            p.bump();
            pat = ast::pat_wild;
        }
        case (token::QUES) {
            p.bump();
            alt (p.peek()) {
                case (token::IDENT(?id, _)) {
                    hi = p.get_hi_pos();
                    p.bump();
                    pat = ast::pat_bind(p.get_str(id));
                }
                case (?tok) {
                    p.fatal("expected identifier after '?' in pattern but " +
                              "found " + token::to_str(p.get_reader(), tok));
                    fail;
                }
            }
        }
        case (token::AT) {
            p.bump();
            auto sub = parse_pat(p);
            pat = ast::pat_box(sub);
            hi = sub.span.hi;
        }
        case (token::LBRACE) {
            p.bump();
            auto fields = ~[];
            auto etc = false;
            auto first = true;
            while (p.peek() != token::RBRACE) {
                if (first) { first = false; }
                else { expect(p, token::COMMA); }

                if (p.peek() == token::UNDERSCORE) {
                    p.bump();
                    if (p.peek() != token::RBRACE) {
                        p.fatal("expecting }, found " +
                                token::to_str(p.get_reader(), p.peek()));
                    }
                    etc = true;
                    break;
                }

                auto fieldname = parse_ident(p);
                auto subpat;
                if (p.peek() == token::COLON) {
                    p.bump();
                    subpat = parse_pat(p);
                } else {
                    if (p.get_bad_expr_words().contains_key(fieldname)) {
                        p.fatal("found " + fieldname +
                                " in binding position");
                    }
                    subpat = @rec(id=p.get_id(),
                                  node=ast::pat_bind(fieldname),
                                  span=rec(lo=lo, hi=hi));
                }
                fields += ~[rec(ident=fieldname, pat=subpat)];
            }
            hi = p.get_hi_pos();
            p.bump();
            pat = ast::pat_rec(fields, etc);
        }
        case (?tok) {
            if (!is_ident(tok) || is_word(p, "true") || is_word(p, "false")) {
                auto lit = parse_lit(p);
                hi = lit.span.hi;
                pat = ast::pat_lit(@lit);
            } else {
                auto tag_path = parse_path_and_ty_param_substs(p);
                hi = tag_path.span.hi;
                let (@ast::pat)[] args;
                alt (p.peek()) {
                    case (token::LPAREN) {
                        auto a = parse_seq(token::LPAREN, token::RPAREN,
                                           some(token::COMMA), parse_pat, p);
                        args = a.node;
                        hi = a.span.hi;
                    }
                    case (_) { args = ~[]; }
                }
                pat = ast::pat_tag(tag_path, args);
            }
        }
    }
    ret @rec(id=p.get_id(), node=pat, span=rec(lo=lo, hi=hi));
}

fn parse_local_full(&option::t[@ast::ty] tyopt, &parser p)
    -> @ast::local {
    auto lo = p.get_lo_pos();
    auto ident = parse_value_ident(p);
    auto init = parse_initializer(p);
    ret @spanned(lo, p.get_hi_pos(),
                 rec(ty=tyopt,
                     infer=false,
                     ident=ident,
                     init=init,
                     id=p.get_id()));
}

fn parse_typed_local(&parser p) -> @ast::local {
    auto ty = parse_ty(p);
    ret parse_local_full(some(ty), p);
}

fn parse_auto_local(&parser p) -> @ast::local {
    ret parse_local_full(none, p);
}

// FIXME simplify when old syntax is no longer supported
fn parse_let(&parser p) -> @ast::decl {
    if alt p.peek() { token::IDENT(_, false) { true } _ { false } } {
        alt p.look_ahead(1u) {
          token::COLON | token::SEMI | token::COMMA | token::EQ |
          token::LARROW {
            ret parse_let_modern(p);
          }
          _ {}
        }
    }
    auto lo = p.get_last_lo_pos();
    auto locals = ~[parse_typed_local(p)];
    while p.peek() == token::COMMA {
        p.bump();
        locals += ~[parse_typed_local(p)];
    }
    ret @spanned(lo, p.get_hi_pos(), ast::decl_local(locals));
}

fn parse_let_modern(&parser p) -> @ast::decl {
    fn parse_local(&parser p) -> @ast::local {
        auto lo = p.get_lo_pos();
        auto ident = parse_value_ident(p);
        auto ty = none;
        if p.peek() == token::COLON {
            p.bump();
            ty = some(parse_ty(p));
        }
        auto init = parse_initializer(p);
        ret @spanned(lo, p.get_last_hi_pos(),
                     rec(ty=ty, infer=false, ident=ident,
                         init=init, id=p.get_id()));
    }
    auto lo = p.get_lo_pos();
    auto locals = ~[parse_local(p)];
    while p.peek() == token::COMMA {
        p.bump();
        locals += ~[parse_local(p)];
    }
    ret @spanned(lo, p.get_last_hi_pos(), ast::decl_local(locals));
}

fn parse_auto(&parser p) -> @ast::decl {
    auto lo = p.get_last_lo_pos();
    auto locals = ~[parse_auto_local(p)];
    while p.peek() == token::COMMA {
        p.bump();
        locals += ~[parse_auto_local(p)];
    }
    ret @spanned(lo, p.get_hi_pos(), ast::decl_local(locals));
}

fn parse_stmt(&parser p) -> @ast::stmt {
    if (p.get_file_type() == SOURCE_FILE) {
        ret parse_source_stmt(p);
    } else { ret parse_crate_stmt(p); }
}

fn parse_crate_stmt(&parser p) -> @ast::stmt {
    auto cdir = parse_crate_directive(p, ~[]);
    ret @spanned(cdir.span.lo, cdir.span.hi,
                 ast::stmt_crate_directive(@cdir));
}

fn parse_source_stmt(&parser p) -> @ast::stmt {
    auto lo = p.get_lo_pos();
    if (eat_word(p, "let")) {
        auto decl = parse_let(p);
        ret @spanned(lo, decl.span.hi, ast::stmt_decl(decl, p.get_id()));
    } else if (eat_word(p, "auto")) {
        auto decl = parse_auto(p);
        ret @spanned(lo, decl.span.hi, ast::stmt_decl(decl, p.get_id()));
    } else {

        auto item_attrs;
        alt (parse_outer_attrs_or_ext(p)) {
            case (none) {
                item_attrs = ~[];
            }
            case (some(left(?attrs))) {
                item_attrs = attrs;
            }
            case (some(right(?ext))) {
                ret @spanned(lo, ext.span.hi,
                             ast::stmt_expr(ext, p.get_id()));
            }
        }

        auto maybe_item = parse_item(p, item_attrs);

        // If we have attributes then we should have an item
        if (ivec::len(item_attrs) > 0u) {
            alt (maybe_item) {
                case (got_item(_)) { /* fallthrough */ }
                case (_) {
                    ret p.fatal("expected item");
                }
            }
        }

        alt (maybe_item) {
            case (got_item(?i)) {
                auto hi = i.span.hi;
                auto decl = @spanned(lo, hi, ast::decl_item(i));
                ret @spanned(lo, hi, ast::stmt_decl(decl, p.get_id()));
            }
            case (fn_no_item) { // parse_item will have already skipped "fn"

                auto e = parse_fn_expr(p);
                e = parse_dot_or_call_expr_with(p, e);
                ret @spanned(lo, e.span.hi, ast::stmt_expr(e, p.get_id()));
            }
            case (no_item) {
                // Remainder are line-expr stmts.

                auto e = parse_expr(p);
                ret @spanned(lo, e.span.hi, ast::stmt_expr(e, p.get_id()));
            }
        }
    }
    p.fatal("expected statement");
    fail;
}

fn stmt_to_expr(@ast::stmt stmt) -> option::t[@ast::expr] {
    ret alt (stmt.node) {
            case (ast::stmt_expr(?e, _)) { some(e) }
            case (_) { none }
        };
}

fn stmt_ends_with_semi(&ast::stmt stmt) -> bool {
    alt (stmt.node) {
        case (ast::stmt_decl(?d, _)) {
            ret alt (d.node) {
                case (ast::decl_local(_)) { true }
                case (ast::decl_item(_)) { false }
            }
        }
        case (ast::stmt_expr(?e, _)) {
            ret alt (e.node) {
                case (ast::expr_vec(_, _, _)) { true }
                case (ast::expr_tup(_)) { true }
                case (ast::expr_rec(_, _)) { true }
                case (ast::expr_call(_, _)) { true }
                case (ast::expr_self_method(_)) { false }
                case (ast::expr_bind(_, _)) { true }
                case (ast::expr_spawn(_, _, _, _)) { true }
                case (ast::expr_binary(_, _, _)) { true }
                case (ast::expr_unary(_, _)) { true }
                case (ast::expr_lit(_)) { true }
                case (ast::expr_cast(_, _)) { true }
                case (ast::expr_if(_, _, _)) { false }
                case (ast::expr_ternary(_, _, _)) { true }
                case (ast::expr_for(_, _, _)) { false }
                case (ast::expr_for_each(_, _, _)) { false }
                case (ast::expr_while(_, _)) { false }
                case (ast::expr_do_while(_, _)) { false }
                case (ast::expr_alt(_, _)) { false }
                case (ast::expr_fn(_)) { false }
                case (ast::expr_block(_)) { false }
                case (ast::expr_move(_, _)) { true }
                case (ast::expr_assign(_, _)) { true }
                case (ast::expr_swap(_, _)) { true }
                case (ast::expr_assign_op(_, _, _)) { true }
                case (ast::expr_send(_, _)) { true }
                case (ast::expr_recv(_, _)) { true }
                case (ast::expr_field(_, _)) { true }
                case (ast::expr_index(_, _)) { true }
                case (ast::expr_path(_)) { true }
                case (ast::expr_mac(_)) { true }
                case (ast::expr_fail(_)) { true }
                case (ast::expr_break) { true }
                case (ast::expr_cont) { true }
                case (ast::expr_ret(_)) { true }
                case (ast::expr_put(_)) { true }
                case (ast::expr_be(_)) { true }
                case (ast::expr_log(_, _)) { true }
                case (ast::expr_check(_, _)) { true }
                case (ast::expr_if_check(_, _, _)) { false }
                case (ast::expr_port(_)) { true }
                case (ast::expr_chan(_)) { true }
                case (ast::expr_anon_obj(_)) { false }
                case (ast::expr_assert(_)) { true }
            }
        }
        // We should not be calling this on a cdir.
        case (ast::stmt_crate_directive(?cdir)) {
            fail;
        }
    }
}

fn parse_block(&parser p) -> ast::blk {
    auto lo = p.get_lo_pos();
    expect(p, token::LBRACE);
    be parse_block_tail(p, lo);
}

// some blocks start with "#{"...
fn parse_block_tail(&parser p, uint lo) -> ast::blk {
    let (@ast::stmt)[] stmts = ~[];
    let option::t[@ast::expr] expr = none;
    while (p.peek() != token::RBRACE) {
        alt (p.peek()) {
            case (token::SEMI) {
                p.bump(); // empty
            }
            case (_) {
                auto stmt = parse_stmt(p);
                alt (stmt_to_expr(stmt)) {
                    case (some(?e)) {
                        alt (p.peek()) {
                            case (token::SEMI) { p.bump(); stmts += ~[stmt]; }
                            case (token::RBRACE) { expr = some(e); }
                            case (?t) {
                                if (stmt_ends_with_semi(*stmt)) {
                                    p.fatal("expected ';' or '}' after " +
                                              "expression but found " +
                                              token::to_str(p.get_reader(),
                                                            t));
                                    fail;
                                }
                                stmts += ~[stmt];
                            }
                        }
                    }
                    case (none) {
                        // Not an expression statement.
                        stmts += ~[stmt];

                        if (p.get_file_type() == SOURCE_FILE
                            && stmt_ends_with_semi(*stmt)) {
                            expect(p, token::SEMI);
                        }
                    }
                }
            }
        }
    }
    auto hi = p.get_hi_pos();
    p.bump();
    auto bloc = rec(stmts=stmts, expr=expr, id=p.get_id());
    ret spanned(lo, hi, bloc);
}

fn parse_ty_param(&parser p) -> ast::ty_param { ret parse_ident(p); }

fn parse_ty_params(&parser p) -> ast::ty_param[] {
    let ast::ty_param[] ty_params = ~[];
    if (p.peek() == token::LBRACKET) {
        ty_params = parse_seq(token::LBRACKET, token::RBRACKET,
                              some(token::COMMA), parse_ty_param, p).node;
    }
    ret ty_params;
}

fn parse_fn_decl(&parser p, ast::purity purity) -> ast::fn_decl {
    let ast::spanned[ast::arg[]] inputs =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                       parse_arg, p);
    let ty_or_bang rslt;
// Use the args list to translate each bound variable
// mentioned in a constraint to an arg index.
// Seems weird to do this in the parser, but I'm not sure how else to.
    auto constrs = ~[];
    if (p.peek() == token::COLON) {
        p.bump();
        constrs = parse_constrs(bind parse_ty_constr(inputs.node,_), p);
    }
    if (p.peek() == token::RARROW) {
        p.bump();
        rslt = parse_ty_or_bang(p);
    } else {
        rslt = a_ty(@spanned(inputs.span.lo, inputs.span.hi, ast::ty_nil));
    }
    alt (rslt) {
        case (a_ty(?t)) {
            ret rec(inputs=inputs.node,
                    output=t,
                    purity=purity,
                    cf=ast::return,
                    constraints=constrs);
        }
        case (a_bang) {
            ret rec(inputs=inputs.node,
                    output=@spanned(p.get_lo_pos(), p.get_hi_pos(),
                                    ast::ty_bot),
                    purity=purity,
                    cf=ast::noreturn,
                    constraints=constrs);
        }
    }
}

fn parse_fn(&parser p, ast::proto proto, ast::purity purity) -> ast::_fn {
    auto decl = parse_fn_decl(p, purity);
    auto body = parse_block(p);
    ret rec(decl=decl, proto=proto, body=body);
}

fn parse_fn_header(&parser p) -> tup(ast::ident, ast::ty_param[]) {
    auto id = parse_value_ident(p);
    auto ty_params = parse_ty_params(p);
    ret tup(id, ty_params);
}

fn mk_item(&parser p, uint lo, uint hi, &ast::ident ident, &ast::item_ node,
           &ast::attribute[] attrs) -> @ast::item {
    ret @rec(ident=ident,
             attrs=attrs,
             id=p.get_id(),
             node=node,
             span=rec(lo=lo, hi=hi));
}

fn parse_item_fn_or_iter(&parser p, ast::purity purity, ast::proto proto,
                         &ast::attribute[] attrs) -> @ast::item {
    auto lo = p.get_last_lo_pos();
    auto t = parse_fn_header(p);
    auto f = parse_fn(p, proto, purity);
    ret mk_item(p, lo, f.body.span.hi, t._0, ast::item_fn(f, t._1), attrs);
}

fn parse_obj_field(&parser p) -> ast::obj_field {
    auto mut = parse_mutability(p);
    auto ty = parse_ty(p);
    auto ident = parse_value_ident(p);
    ret rec(mut=mut, ty=ty, ident=ident, id=p.get_id());
}

fn parse_anon_obj_field(&parser p) -> ast::anon_obj_field {
    auto mut = parse_mutability(p);
    auto ty = parse_ty(p);
    auto ident = parse_value_ident(p);
    expect(p, token::EQ);
    auto expr = parse_expr(p);
    ret rec(mut=mut, ty=ty, expr=expr, ident=ident, id=p.get_id());
}

fn parse_method(&parser p) -> @ast::method {
    auto lo = p.get_lo_pos();
    auto proto = parse_proto(p);
    auto ident = parse_value_ident(p);
    auto f = parse_fn(p, proto, ast::impure_fn);
    auto meth = rec(ident=ident, meth=f, id=p.get_id());
    ret @spanned(lo, f.body.span.hi, meth);
}

fn parse_dtor(&parser p) -> @ast::method {
    auto lo = p.get_last_lo_pos();
    let ast::blk b = parse_block(p);
    let ast::arg[] inputs = ~[];
    let @ast::ty output = @spanned(lo, lo, ast::ty_nil);
    let ast::fn_decl d =
        rec(inputs=inputs,
            output=output,
            purity=ast::impure_fn,
            cf=ast::return,

            // I guess dtors can't have constraints?
            constraints=~[]);
    let ast::_fn f = rec(decl=d, proto=ast::proto_fn, body=b);
    let ast::method_ m =
        rec(ident="drop", meth=f, id=p.get_id());
    ret @spanned(lo, f.body.span.hi, m);
}

fn parse_item_obj(&parser p, ast::layer lyr, &ast::attribute[] attrs) ->
   @ast::item {
    auto lo = p.get_last_lo_pos();
    auto ident = parse_value_ident(p);
    auto ty_params = parse_ty_params(p);
    let ast::spanned[ast::obj_field[]] fields =
        parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                       parse_obj_field, p);
    let (@ast::method)[] meths = ~[];
    let option::t[@ast::method] dtor = none;
    expect(p, token::LBRACE);
    while (p.peek() != token::RBRACE) {
        if (eat_word(p, "drop")) {
            dtor = some(parse_dtor(p));
        } else { meths += ~[parse_method(p)]; }
    }
    auto hi = p.get_hi_pos();
    expect(p, token::RBRACE);
    let ast::_obj ob = rec(fields=fields.node, methods=meths, dtor=dtor);
    ret mk_item(p, lo, hi, ident, ast::item_obj(ob, ty_params,
                                                p.get_id()), attrs);
}

fn parse_item_res(&parser p, ast::layer lyr, &ast::attribute[] attrs) ->
   @ast::item {
    auto lo = p.get_last_lo_pos();
    auto ident = parse_value_ident(p);
    auto ty_params = parse_ty_params(p);
    expect(p, token::LPAREN);
    auto t = parse_ty(p);
    auto arg_ident = parse_value_ident(p);
    expect(p, token::RPAREN);
    auto dtor = parse_block(p);
    auto decl = rec(inputs=~[rec(mode=ast::alias(false), ty=t,
                                 ident=arg_ident, id=p.get_id())],
                    output=@spanned(lo, lo, ast::ty_nil),
                    purity=ast::impure_fn,
                    cf=ast::return,
                    constraints=~[]);
    auto f = rec(decl=decl, proto=ast::proto_fn, body=dtor);
    ret mk_item(p, lo, dtor.span.hi, ident,
                ast::item_res(f, p.get_id(), ty_params, p.get_id()), attrs);
}

fn parse_mod_items(&parser p, token::token term,
                   &ast::attribute[] first_item_attrs) -> ast::_mod {
    auto view_items = if (ivec::len(first_item_attrs) == 0u) {
        parse_view(p)
    } else {
        // Shouldn't be any view items since we've already parsed an item attr
        ~[]
    };
    let (@ast::item)[] items = ~[];
    auto initial_attrs = first_item_attrs;
    while (p.peek() != term) {
        auto attrs = initial_attrs + parse_outer_attributes(p);
        initial_attrs = ~[];
        alt (parse_item(p, attrs)) {
            case (got_item(?i)) { items += ~[i]; }
            case (_) {
                p.fatal("expected item but found " +
                          token::to_str(p.get_reader(), p.peek()));
            }
        }
    }
    ret rec(view_items=view_items, items=items);
}

fn parse_item_const(&parser p, &ast::attribute[] attrs) -> @ast::item {
    auto lo = p.get_last_lo_pos();
    auto ty = parse_ty(p);
    auto id = parse_value_ident(p);
    expect(p, token::EQ);
    auto e = parse_expr(p);
    auto hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret mk_item(p, lo, hi, id, ast::item_const(ty, e), attrs);
}

fn parse_item_mod(&parser p, &ast::attribute[] attrs) -> @ast::item {
    auto lo = p.get_last_lo_pos();
    auto id = parse_ident(p);
    expect(p, token::LBRACE);
    auto inner_attrs = parse_inner_attrs_and_next(p);
    auto first_item_outer_attrs = inner_attrs._1;
    auto m = parse_mod_items(p, token::RBRACE, first_item_outer_attrs);
    auto hi = p.get_hi_pos();
    expect(p, token::RBRACE);
    ret mk_item(p, lo, hi, id, ast::item_mod(m), attrs + inner_attrs._0);
}

fn parse_item_native_type(&parser p, &ast::attribute[] attrs)
        -> @ast::native_item {
    auto t = parse_type_decl(p);
    auto hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret @rec(ident=t._1,
             attrs=attrs,
             node=ast::native_item_ty,
             id=p.get_id(),
             span=rec(lo=t._0, hi=hi));
}

fn parse_item_native_fn(&parser p, &ast::attribute[] attrs)
        -> @ast::native_item {
    auto lo = p.get_last_lo_pos();
    auto t = parse_fn_header(p);
    auto decl = parse_fn_decl(p, ast::impure_fn);
    auto link_name = none;
    if (p.peek() == token::EQ) {
        p.bump();
        link_name = some(parse_str(p));
    }
    auto hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret @rec(ident=t._0,
             attrs=attrs,
             node=ast::native_item_fn(link_name, decl, t._1),
             id=p.get_id(),
             span=rec(lo=lo, hi=hi));
}

fn parse_native_item(&parser p, &ast::attribute[] attrs)
        -> @ast::native_item {
    parse_layer(p);
    if (eat_word(p, "type")) {
        ret parse_item_native_type(p, attrs);
    } else if (eat_word(p, "fn")) {
        ret parse_item_native_fn(p, attrs);
    } else { unexpected(p, p.peek()); fail; }
}

fn parse_native_mod_items(&parser p, &str native_name, ast::native_abi abi,
                          &ast::attribute[] first_item_attrs)
        -> ast::native_mod {
    auto view_items = if (ivec::len(first_item_attrs) == 0u) {
        parse_native_view(p)
    } else {
        // Shouldn't be any view items since we've already parsed an item attr
        ~[]
    };
    let (@ast::native_item)[] items = ~[];
    auto initial_attrs = first_item_attrs;
    while (p.peek() != token::RBRACE) {
        auto attrs = initial_attrs + parse_outer_attributes(p);
        initial_attrs = ~[];
        items += ~[parse_native_item(p, attrs)];
    }
    ret rec(native_name=native_name,
            abi=abi,
            view_items=view_items,
            items=items);
}

fn parse_item_native_mod(&parser p, &ast::attribute[] attrs) -> @ast::item {
    auto lo = p.get_last_lo_pos();
    auto abi = ast::native_abi_cdecl;
    if (!is_word(p, "mod")) {
        auto t = parse_str(p);
        if (str::eq(t, "cdecl")) {
        } else if (str::eq(t, "rust")) {
            abi = ast::native_abi_rust;
        } else if (str::eq(t, "llvm")) {
            abi = ast::native_abi_llvm;
        } else if (str::eq(t, "rust-intrinsic")) {
            abi = ast::native_abi_rust_intrinsic;
        } else if (str::eq(t, "x86stdcall")) {
            abi = ast::native_abi_x86stdcall;
        } else { p.fatal("unsupported abi: " + t); fail; }
    }
    expect_word(p, "mod");
    auto id = parse_ident(p);
    auto native_name;
    if (p.peek() == token::EQ) {
        expect(p, token::EQ);
        native_name = parse_str(p);
    } else {
        native_name = id;
    }
    expect(p, token::LBRACE);
    auto more_attrs = parse_inner_attrs_and_next(p);
    auto inner_attrs = more_attrs._0;
    auto first_item_outer_attrs = more_attrs._1;
    auto m = parse_native_mod_items(p, native_name, abi,
                                    first_item_outer_attrs);
    auto hi = p.get_hi_pos();
    expect(p, token::RBRACE);
    ret mk_item(p, lo, hi, id, ast::item_native_mod(m), attrs + inner_attrs);
}

fn parse_type_decl(&parser p) -> tup(uint, ast::ident) {
    auto lo = p.get_last_lo_pos();
    auto id = parse_ident(p);
    ret tup(lo, id);
}

fn parse_item_type(&parser p, &ast::attribute[] attrs) -> @ast::item {
    auto t = parse_type_decl(p);
    auto tps = parse_ty_params(p);
    expect(p, token::EQ);
    auto ty = parse_ty(p);
    auto hi = p.get_hi_pos();
    expect(p, token::SEMI);
    ret mk_item(p, t._0, hi, t._1, ast::item_ty(ty, tps), attrs);
}

fn parse_item_tag(&parser p, &ast::attribute[] attrs) -> @ast::item {
    auto lo = p.get_last_lo_pos();
    auto id = parse_ident(p);
    auto ty_params = parse_ty_params(p);
    let ast::variant[] variants = ~[];
    // Newtype syntax
    if (p.peek() == token::EQ) {
        if (p.get_bad_expr_words().contains_key(id)) {
            p.fatal("found " + id + " in tag constructor position");
        }
        p.bump();
        auto ty = parse_ty(p);
        expect(p, token::SEMI);
        auto variant = spanned(ty.span.lo, ty.span.hi,
                               rec(name=id,
                                   args=~[rec(ty=ty, id=p.get_id())],
                                   id=p.get_id()));
        ret mk_item(p, lo, ty.span.hi, id,
                    ast::item_tag(~[variant], ty_params), attrs);
    }
    expect(p, token::LBRACE);
    while (p.peek() != token::RBRACE) {
        auto tok = p.peek();
        alt (tok) {
            case (token::IDENT(?name, _)) {
                check_bad_word(p);
                auto vlo = p.get_lo_pos();
                p.bump();
                let ast::variant_arg[] args = ~[];
                auto vhi = p.get_hi_pos();
                alt (p.peek()) {
                    case (token::LPAREN) {
                        auto arg_tys =
                            parse_seq(token::LPAREN, token::RPAREN,
                                           some(token::COMMA), parse_ty, p);
                        for (@ast::ty ty in arg_tys.node) {
                            args += ~[rec(ty=ty, id=p.get_id())];
                        }
                        vhi = arg_tys.span.hi;
                    }
                    case (_) {/* empty */ }
                }
                expect(p, token::SEMI);
                p.get_id();
                auto vr =
                    rec(name=p.get_str(name),
                        args=args,
                        id=p.get_id());
                variants += ~[spanned(vlo, vhi, vr)];
            }
            case (token::RBRACE) {/* empty */ }
            case (_) {
                p.fatal("expected name of variant or '}' but found " +
                          token::to_str(p.get_reader(), tok));
            }
        }
    }
    auto hi = p.get_hi_pos();
    p.bump();
    ret mk_item(p, lo, hi, id, ast::item_tag(variants, ty_params), attrs);
}

fn parse_layer(&parser p) -> ast::layer {
    if (eat_word(p, "state")) {
        ret ast::layer_state;
    } else if (eat_word(p, "gc")) {
        ret ast::layer_gc;
    } else { ret ast::layer_value; }
    fail;
}

fn parse_auth(&parser p) -> ast::_auth {
    if (eat_word(p, "unsafe")) {
        ret ast::auth_unsafe;
    } else { unexpected(p, p.peek()); }
    fail;
}

tag parsed_item { got_item(@ast::item); no_item; fn_no_item; }

fn parse_item(&parser p, &ast::attribute[] attrs) -> parsed_item {
    if (eat_word(p, "const")) {
        ret got_item(parse_item_const(p, attrs));
    } else if (eat_word(p, "fn")) {
        // This is an anonymous function

        if (p.peek() == token::LPAREN) { ret fn_no_item; }
        ret got_item(parse_item_fn_or_iter(p, ast::impure_fn, ast::proto_fn,
                                           attrs));
    } else if (eat_word(p, "pred")) {
        ret got_item(parse_item_fn_or_iter(p, ast::pure_fn, ast::proto_fn,
                                           attrs));
    } else if (eat_word(p, "iter")) {
        ret got_item(parse_item_fn_or_iter(p, ast::impure_fn, ast::proto_iter,
                                           attrs));
    } else if (eat_word(p, "mod")) {
        ret got_item(parse_item_mod(p, attrs));
    } else if (eat_word(p, "native")) {
        ret got_item(parse_item_native_mod(p, attrs));
    }
    auto lyr = parse_layer(p);
    if (eat_word(p, "type")) {
        ret got_item(parse_item_type(p, attrs));
    } else if (eat_word(p, "tag")) {
        ret got_item(parse_item_tag(p, attrs));
    } else if (eat_word(p, "obj")) {
        ret got_item(parse_item_obj(p, lyr, attrs));
    } else if (eat_word(p, "resource")) {
        ret got_item(parse_item_res(p, lyr, attrs));
    } else { ret no_item; }
}

// A type to distingush between the parsing of item attributes or syntax
// extensions, which both begin with token.POUND
type attr_or_ext = option::t[either::t[ast::attribute[], @ast::expr]];

fn parse_outer_attrs_or_ext(&parser p) -> attr_or_ext {
    if (p.peek() == token::POUND) {
        auto lo = p.get_lo_pos();
        p.bump();
        if (p.peek() == token::LBRACKET) {
            auto first_attr = parse_attribute_naked(p, ast::attr_outer, lo);
            ret some(left(~[first_attr] + parse_outer_attributes(p)));
        } else if (! (p.peek() == token::LT || p.peek() == token::LBRACKET)) {
            ret some(right(parse_syntax_ext_naked(p, lo)));
        } else {
            ret none;
        }
    } else {
        ret none;
    }
}

// Parse attributes that appear before an item
fn parse_outer_attributes(&parser p) -> ast::attribute[] {
    let ast::attribute[] attrs = ~[];
    while (p.peek() == token::POUND) {
        attrs += ~[parse_attribute(p, ast::attr_outer)];
    }
    ret attrs;
}

fn parse_attribute(&parser p, ast::attr_style style) -> ast::attribute {
    auto lo = p.get_lo_pos();
    expect(p, token::POUND);
    ret parse_attribute_naked(p, style, lo);
}

fn parse_attribute_naked(&parser p, ast::attr_style style,
                         uint lo) -> ast::attribute {
    expect(p, token::LBRACKET);
    auto meta_item = parse_meta_item(p);
    expect(p, token::RBRACKET);
    auto hi = p.get_hi_pos();
    ret spanned(lo, hi, rec(style=style, value=*meta_item));
}

// Parse attributes that appear after the opening of an item, each terminated
// by a semicolon. In addition to a vector of inner attributes, this function
// also returns a vector that may contain the first outer attribute of the
// next item (since we can't know whether the attribute is an inner attribute
// of the containing item or an outer attribute of the first contained item
// until we see the semi).
fn parse_inner_attrs_and_next(&parser p) -> tup(ast::attribute[],
                                                ast::attribute[]) {
    let ast::attribute[] inner_attrs = ~[];
    let ast::attribute[] next_outer_attrs = ~[];
    while (p.peek() == token::POUND) {
        auto attr = parse_attribute(p, ast::attr_inner);
        if (p.peek() == token::SEMI) {
            p.bump();
            inner_attrs += ~[attr];
        } else {
            // It's not really an inner attribute
            auto outer_attr = spanned(attr.span.lo,
                                      attr.span.hi,
                                      rec(style=ast::attr_outer,
                                          value=attr.node.value));
            next_outer_attrs += ~[outer_attr];
            break;
        }
    }
    ret tup(inner_attrs, next_outer_attrs);
}

fn parse_meta_item(&parser p) -> @ast::meta_item {
    auto lo = p.get_lo_pos();
    auto ident = parse_ident(p);
    alt (p.peek()) {
        case (token::EQ) {
            p.bump();
            auto lit = parse_lit(p);
            auto hi = p.get_hi_pos();
            ret @spanned(lo, hi, ast::meta_name_value(ident, lit));
        }
        case (token::LPAREN) {
            auto inner_items = parse_meta_seq(p);
            auto hi = p.get_hi_pos();
            ret @spanned(lo, hi, ast::meta_list(ident, inner_items));
        }
        case (_) {
            auto hi = p.get_hi_pos();
            ret @spanned(lo, hi, ast::meta_word(ident));
        }
    }
}

fn parse_meta_seq(&parser p) -> (@ast::meta_item)[] {
    ret parse_seq(token::LPAREN, token::RPAREN, some(token::COMMA),
                  parse_meta_item, p).node;
}

fn parse_optional_meta(&parser p) -> (@ast::meta_item)[] {
    alt (p.peek()) {
        case (token::LPAREN) { ret parse_meta_seq(p); }
        case (_) { ret ~[]; }
    }
}

fn parse_use(&parser p) -> ast::view_item_ {
    auto ident = parse_ident(p);
    auto metadata = parse_optional_meta(p);
    ret ast::view_item_use(ident, metadata, p.get_id());
}

fn parse_rest_import_name(&parser p, ast::ident first,
                          option::t[ast::ident] def_ident) ->
   ast::view_item_ {
    let ast::ident[] identifiers = ~[first];
    let bool glob = false;
    while (true) {
        alt (p.peek()) {
            case (token::SEMI) { break; }
            case (token::MOD_SEP) {
                if (glob) { p.fatal("cannot path into a glob"); }
                p.bump();
            }
            case (_) { p.fatal("expecting '::' or ';'"); }
        }
        alt (p.peek()) {
            case (token::IDENT(_, _)) { identifiers += ~[parse_ident(p)]; }
            //the lexer can't tell the different kinds of stars apart ) :
            case (token::BINOP(token::STAR)) {
                glob = true;
                p.bump();
            }
            case (_) { p.fatal("expecting an identifier, or '*'"); }
        }
    }
    alt (def_ident) {
        case (some(?i)) {
            if (glob) { p.fatal("globbed imports can't be renamed"); }
            ret ast::view_item_import(i, identifiers, p.get_id());
        }
        case (_) {
            if (glob) {
                ret ast::view_item_import_glob(identifiers, p.get_id());
            } else {
                auto len = ivec::len(identifiers);
                ret ast::view_item_import(identifiers.(len - 1u), identifiers,
                                          p.get_id());
            }
        }
    }
}

fn parse_full_import_name(&parser p, ast::ident def_ident) ->
   ast::view_item_ {
    alt (p.peek()) {
        case (token::IDENT(?i, _)) {
            p.bump();
            ret parse_rest_import_name(p, p.get_str(i), some(def_ident));
        }
        case (_) { p.fatal("expecting an identifier"); }
    }
    fail;
}

fn parse_import(&parser p) -> ast::view_item_ {
    alt (p.peek()) {
        case (token::IDENT(?i, _)) {
            p.bump();
            alt (p.peek()) {
                case (token::EQ) {
                    p.bump();
                    ret parse_full_import_name(p, p.get_str(i));
                }
                case (_) {
                    ret parse_rest_import_name(p, p.get_str(i), none);
                }
            }
        }
        case (_) { p.fatal("expecting an identifier"); }
    }
    fail;
}

fn parse_export(&parser p) -> ast::view_item_ {
    auto id = parse_ident(p);
    ret ast::view_item_export(id, p.get_id());
}

fn parse_view_item(&parser p) -> @ast::view_item {
    auto lo = p.get_lo_pos();
    auto the_item = if (eat_word(p, "use")) { parse_use(p) }
                    else if (eat_word(p, "import")) { parse_import(p) }
                    else if (eat_word(p, "export")) { parse_export(p) }
                    else { fail };
    auto hi = p.get_lo_pos();
    expect(p, token::SEMI);
    ret @spanned(lo, hi, the_item);
}

fn is_view_item(&parser p) -> bool {
    alt (p.peek()) {
        case (token::IDENT(?sid, false)) {
            auto st = p.get_str(sid);
            ret str::eq(st, "use") || str::eq(st, "import") ||
                    str::eq(st, "export");
        }
        case (_) { ret false; }
    }
    ret false;
}

fn parse_view(&parser p) -> (@ast::view_item)[] {
    let (@ast::view_item)[] items = ~[];
    while (is_view_item(p)) { items += ~[parse_view_item(p)]; }
    ret items;
}

fn parse_native_view(&parser p) -> (@ast::view_item)[] {
    let (@ast::view_item)[] items = ~[];
    while (is_view_item(p)) { items += ~[parse_view_item(p)]; }
    ret items;
}

fn parse_crate_from_source_file(&str input, &ast::crate_cfg cfg,
                                &parse_sess sess) -> @ast::crate {
    auto p = new_parser_from_file(sess, cfg, input, 0u, 0u);
    ret parse_crate_mod(p, cfg, sess);
}

fn parse_crate_from_source_str(&str name, &str source, &ast::crate_cfg cfg,
                               &codemap::codemap cm) -> @ast::crate {
    auto sess = @rec(cm=cm, mutable next_id=0);
    auto ftype = SOURCE_FILE;
    auto filemap = codemap::new_filemap(name, 0u, 0u);
    sess.cm.files += ~[filemap];
    auto itr = @interner::mk(str::hash, str::eq);
    auto rdr = lexer::new_reader(sess.cm, source, filemap, itr);
    auto p = new_parser(sess, cfg, rdr, ftype);
    ret parse_crate_mod(p, cfg, sess);
}

// Parses a source module as a crate
fn parse_crate_mod(&parser p, &ast::crate_cfg cfg, parse_sess sess)
    -> @ast::crate {
    auto lo = p.get_lo_pos();
    auto crate_attrs = parse_inner_attrs_and_next(p);
    auto first_item_outer_attrs = crate_attrs._1;
    auto m = parse_mod_items(p, token::EOF,
                             first_item_outer_attrs);
    ret @spanned(lo, p.get_lo_pos(), rec(directives=~[],
                                         module=m,
                                         attrs=crate_attrs._0,
                                         config=p.get_cfg()));
}

fn parse_str(&parser p) -> ast::ident {
    alt (p.peek()) {
        case (token::LIT_STR(?s)) {
            p.bump();
            ret p.get_str(s);
        }
        case (_) { fail; }
    }
}

// Logic for parsing crate files (.rc)
//
// Each crate file is a sequence of directives.
//
// Each directive imperatively extends its environment with 0 or more items.
fn parse_crate_directive(&parser p, &ast::attribute[] first_outer_attr)
    -> ast::crate_directive {

    // Collect the next attributes
    auto outer_attrs = first_outer_attr + parse_outer_attributes(p);
    // In a crate file outer attributes are only going to apply to mods
    auto expect_mod = ivec::len(outer_attrs) > 0u;

    auto lo = p.get_lo_pos();
    if (expect_mod || is_word(p, "mod")) {
        expect_word(p, "mod");
        auto id = parse_ident(p);
        auto file_opt =
            alt (p.peek()) {
                case (token::EQ) {
                    p.bump();
                    some(parse_str(p))
                }
                case (_) { none }
            };
        alt (p.peek()) {
            case (
                 // mod x = "foo.rs";
                 token::SEMI) {
                auto hi = p.get_hi_pos();
                p.bump();
                ret spanned(lo, hi, ast::cdir_src_mod(id, file_opt,
                                                      outer_attrs));
            }
            case (
                 // mod x = "foo_dir" { ...directives... }
                 token::LBRACE) {
                p.bump();
                auto inner_attrs = parse_inner_attrs_and_next(p);
                auto mod_attrs = outer_attrs + inner_attrs._0;
                auto next_outer_attr = inner_attrs._1;
                auto cdirs = parse_crate_directives(p, token::RBRACE,
                                                    next_outer_attr);
                auto hi = p.get_hi_pos();
                expect(p, token::RBRACE);
                ret spanned(lo, hi, ast::cdir_dir_mod(id, file_opt, cdirs,
                                                      mod_attrs));
            }
            case (?t) { unexpected(p, t); }
        }
    } else if (eat_word(p, "auth")) {
        auto n = parse_path(p);
        expect(p, token::EQ);
        auto a = parse_auth(p);
        auto hi = p.get_hi_pos();
        expect(p, token::SEMI);
        ret spanned(lo, hi, ast::cdir_auth(n, a));
    } else if (is_view_item(p)) {
        auto vi = parse_view_item(p);
        ret spanned(lo, vi.span.hi, ast::cdir_view_item(vi));
    } else {
        ret p.fatal("expected crate directive");
    }
}

fn parse_crate_directives(&parser p, token::token term,
                          &ast::attribute[] first_outer_attr)
        -> (@ast::crate_directive)[] {

    // This is pretty ugly. If we have an outer attribute then we can't accept
    // seeing the terminator next, so if we do see it then fail the same way
    // parse_crate_directive would
    if (ivec::len(first_outer_attr) > 0u && p.peek() == term) {
        expect_word(p, "mod");
    }

    let (@ast::crate_directive)[] cdirs = ~[];
    while (p.peek() != term) {
        auto cdir = @parse_crate_directive(p, first_outer_attr);
        cdirs += ~[cdir];
    }
    ret cdirs;
}

fn parse_crate_from_crate_file(&str input, &ast::crate_cfg cfg,
                               &parse_sess sess) -> @ast::crate {
    auto p = new_parser_from_file(sess, cfg, input, 0u, 0u);
    auto lo = p.get_lo_pos();
    auto prefix = std::fs::dirname(p.get_filemap().name);
    auto leading_attrs = parse_inner_attrs_and_next(p);
    auto crate_attrs = leading_attrs._0;
    auto first_cdir_attr = leading_attrs._1;
    auto cdirs = parse_crate_directives(p, token::EOF, first_cdir_attr);
    let str[] deps = ~[];
    auto cx = @rec(p=p,
                   mode=eval::mode_parse,
                   mutable deps=deps,
                   sess=sess,
                   mutable chpos=p.get_chpos(),
                   mutable byte_pos=p.get_byte_pos(),
                   cfg = p.get_cfg());
    auto m = eval::eval_crate_directives_to_mod(cx, cdirs, prefix);
    auto hi = p.get_hi_pos();
    expect(p, token::EOF);
    ret @spanned(lo, hi, rec(directives=cdirs,
                             module=m,
                             attrs=crate_attrs,
                             config=p.get_cfg()));
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

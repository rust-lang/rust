#[doc = "The main parser interface"];
import dvec::extensions;

export parse_sess;
export next_node_id;
export new_parser_from_file;
export new_parser_from_source_str;
export parse_crate_from_file;
export parse_crate_from_crate_file;
export parse_crate_from_source_str;
export parse_expr_from_source_str;
export parse_item_from_source_str;
export parse_from_source_str;

import parser::parser;
import attr::parser_attr;
import common::parser_common;
import ast::node_id;
import util::interner;
import lexer::reader;

type parse_sess = @{
    cm: codemap::codemap,
    mut next_id: node_id,
    span_diagnostic: diagnostic::span_handler,
    // these two must be kept up to date
    mut chpos: uint,
    mut byte_pos: uint
};

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

fn parse_crate_from_crate_file(input: str, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, parser::CRATE_FILE);
    let lo = p.span.lo;
    let prefix = path::dirname(p.reader.filemap.name);
    let leading_attrs = p.parse_inner_attrs_and_next();
    let { inner: crate_attrs, next: first_cdir_attr } = leading_attrs;
    let cdirs = p.parse_crate_directives(token::EOF, first_cdir_attr);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    let cx = @{sess: sess, cfg: /* FIXME: bad */ copy p.cfg};
    let (companionmod, _) = path::splitext(path::basename(input));
    let (m, attrs) = eval::eval_crate_directives_to_mod(
        cx, cdirs, prefix, option::some(companionmod));
    let mut hi = p.span.hi;
    p.expect(token::EOF);
    ret @ast_util::respan(ast_util::mk_sp(lo, hi),
                          {directives: cdirs,
                           module: m,
                           attrs: crate_attrs + attrs,
                           config: /* FIXME: bad */ copy p.cfg});
}

fn parse_crate_from_source_file(input: str, cfg: ast::crate_cfg,
                                sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, parser::SOURCE_FILE);
    let r = p.parse_crate_mod(cfg);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn parse_crate_from_source_str(name: str, source: @str, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_source_str(
        sess, cfg, name, codemap::fss_none, source);
    let r = p.parse_crate_mod(cfg);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn parse_expr_from_source_str(name: str, source: @str, cfg: ast::crate_cfg,
                              sess: parse_sess) -> @ast::expr {
    let p = new_parser_from_source_str(
        sess, cfg, name, codemap::fss_none, source);
    let r = p.parse_expr();
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn parse_item_from_source_str(name: str, source: @str, cfg: ast::crate_cfg,
                              +attrs: [ast::attribute], vis: ast::visibility,
                              sess: parse_sess) -> option<@ast::item> {
    let p = new_parser_from_source_str(
        sess, cfg, name, codemap::fss_none, source);
    let r = p.parse_item(attrs, vis);
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
    if !p.reader.is_eof() {
        p.reader.fatal("expected end-of-string");
    }
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn next_node_id(sess: parse_sess) -> node_id {
    let rv = sess.next_id;
    sess.next_id += 1;
    // ID 0 is reserved for the crate and doesn't actually exist in the AST
    assert rv != 0;
    ret rv;
}

fn new_parser_from_source_str(sess: parse_sess, cfg: ast::crate_cfg,
                              +name: str, +ss: codemap::file_substr,
                              source: @str) -> parser {
    let ftype = parser::SOURCE_FILE;
    let filemap = codemap::new_filemap_w_substr
        (name, ss, source, sess.chpos, sess.byte_pos);
    sess.cm.files.push(filemap);
    let itr = @interner::mk::<@str>(
        {|x|str::hash(*x)},
        {|x,y|str::eq(*x, *y)}
    );
    let rdr = lexer::new_reader(sess.span_diagnostic,
                                filemap, itr);
    ret parser(sess, cfg, rdr, ftype);
}

fn new_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg, +path: str,
                        ftype: parser::file_type) ->
   parser {
    let res = io::read_whole_file_str(path);
    alt res {
      result::ok(_) { /* Continue. */ }
      result::err(e) { sess.span_diagnostic.handler().fatal(e); }
    }
    let src = @result::unwrap(res);
    let filemap = codemap::new_filemap(path, src, sess.chpos, sess.byte_pos);
    sess.cm.files.push(filemap);
    let itr = @interner::mk::<@str>(
        {|x|str::hash(*x)},
        {|x,y|str::eq(*x, *y)}
    );
    let rdr = lexer::new_reader(sess.span_diagnostic, filemap, itr);
    ret parser(sess, cfg, rdr, ftype);
}

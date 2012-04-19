export parse_sess;
export next_node_id;
export new_parser_from_file;
export new_parser_from_source_str;
export parse_crate_from_file;
export parse_crate_from_crate_file;
export parse_crate_from_source_str;
export parse_expr_from_source_str;
export parse_from_source_str;

import parser::parser;
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

fn next_node_id(sess: parse_sess) -> node_id {
    let rv = sess.next_id;
    sess.next_id += 1;
    // ID 0 is reserved for the crate and doesn't actually exist in the AST
    assert rv != 0;
    ret rv;
}

fn new_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg, path: str,
                        ftype: parser::file_type) ->
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
    let rdr = lexer::new_reader(sess.span_diagnostic, filemap, itr);
    ret new_parser(sess, cfg, rdr, ftype);
}

fn new_parser_from_source_str(sess: parse_sess, cfg: ast::crate_cfg,
                              name: str, ss: codemap::file_substr,
                              source: @str) -> parser {
    let ftype = parser::SOURCE_FILE;
    let filemap = codemap::new_filemap_w_substr
        (name, ss, source, sess.chpos, sess.byte_pos);
    sess.cm.files += [filemap];
    let itr = @interner::mk(str::hash, str::eq);
    let rdr = lexer::new_reader(sess.span_diagnostic,
                                filemap, itr);
    ret new_parser(sess, cfg, rdr, ftype);
}

fn new_parser(sess: parse_sess, cfg: ast::crate_cfg, rdr: lexer::reader,
              ftype: parser::file_type) -> parser {
    let tok0 = lexer::next_token(rdr);
    let span0 = ast_util::mk_sp(tok0.chpos, rdr.chpos);
    @{sess: sess,
      cfg: cfg,
      file_type: ftype,
      mut token: tok0.tok,
      mut span: span0,
      mut last_span: span0,
      mut buffer: [],
      mut restriction: parser::UNRESTRICTED,
      reader: rdr,
      binop_precs: prec::binop_prec_table(),
      keywords: token::keyword_table(),
      bad_expr_words: token::bad_expr_word_table()}
}

fn parse_crate_from_crate_file(input: str, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, parser::CRATE_FILE);
    let lo = p.span.lo;
    let prefix = path::dirname(p.reader.filemap.name);
    let leading_attrs = parser::parse_inner_attrs_and_next(p);
    let crate_attrs = leading_attrs.inner;
    let first_cdir_attr = leading_attrs.next;
    let cdirs = parser::parse_crate_directives(
        p, token::EOF, first_cdir_attr);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    let cx =
        @{p: p,
          sess: sess,
          cfg: p.cfg};
    let (companionmod, _) = path::splitext(path::basename(input));
    let (m, attrs) = eval::eval_crate_directives_to_mod(
        cx, cdirs, prefix, option::some(companionmod));
    let mut hi = p.span.hi;
    parser::expect(p, token::EOF);
    ret @ast_util::respan(ast_util::mk_sp(lo, hi),
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

fn parse_crate_from_source_file(input: str, cfg: ast::crate_cfg,
                                sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_file(sess, cfg, input, parser::SOURCE_FILE);
    let r = parser::parse_crate_mod(p, cfg);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn parse_expr_from_source_str(name: str, source: @str, cfg: ast::crate_cfg,
                              sess: parse_sess) -> @ast::expr {
    let p = new_parser_from_source_str(
        sess, cfg, name, codemap::fss_none, source);
    let r = parser::parse_expr(p);
    sess.chpos = p.reader.chpos;
    sess.byte_pos = sess.byte_pos + p.reader.pos;
    ret r;
}

fn parse_crate_from_source_str(name: str, source: @str, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_source_str(
        sess, cfg, name, codemap::fss_none, source);
    let r = parser::parse_crate_mod(p, cfg);
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

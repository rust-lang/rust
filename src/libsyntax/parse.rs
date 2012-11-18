//! The main parser interface

export parse_sess;
export new_parse_sess, new_parse_sess_special_handler;
export next_node_id;
export new_parser_from_file, new_parser_etc_from_file;
export new_parser_from_source_str;
export new_parser_from_tt;
export new_sub_parser_from_file;
export parse_crate_from_file, parse_crate_from_crate_file;
export parse_crate_from_source_str;
export parse_expr_from_source_str, parse_item_from_source_str;
export parse_stmt_from_source_str;
export parse_from_source_str;

use parser::Parser;
use attr::parser_attr;
use common::parser_common;
use ast::node_id;
use util::interner;
use diagnostic::{span_handler, mk_span_handler, mk_handler, emitter};
use lexer::{reader, string_reader};
use parse::token::{ident_interner, mk_ident_interner};
use codemap::{span, CodeMap, FileMap, CharPos, BytePos};

type parse_sess = @{
    cm: @codemap::CodeMap,
    mut next_id: node_id,
    span_diagnostic: span_handler,
    interner: @ident_interner,
};

fn new_parse_sess(demitter: Option<emitter>) -> parse_sess {
    let cm = @CodeMap::new();
    return @{cm: cm,
             mut next_id: 1,
             span_diagnostic: mk_span_handler(mk_handler(demitter), cm),
             interner: mk_ident_interner(),
            };
}

fn new_parse_sess_special_handler(sh: span_handler, cm: @codemap::CodeMap)
    -> parse_sess {
    return @{cm: cm,
             mut next_id: 1,
             span_diagnostic: sh,
             interner: mk_ident_interner(),
             };
}

fn parse_crate_from_file(input: &Path, cfg: ast::crate_cfg,
                         sess: parse_sess) -> @ast::crate {
    if input.filetype() == Some(~".rc") {
        parse_crate_from_crate_file(input, cfg, sess)
    } else if input.filetype() == Some(~".rs") {
        parse_crate_from_source_file(input, cfg, sess)
    } else {
        sess.span_diagnostic.handler().fatal(~"unknown input file type: " +
                                             input.to_str())
    }
}

fn parse_crate_from_crate_file(input: &Path, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_crate_parser_from_file(sess, cfg, input,
                                       parser::CRATE_FILE);
    let lo = p.span.lo;
    let prefix = input.dir_path();
    let leading_attrs = p.parse_inner_attrs_and_next();
    let { inner: crate_attrs, next: first_cdir_attr } = leading_attrs;
    let cdirs = p.parse_crate_directives(token::EOF, first_cdir_attr);
    let cx = @{sess: sess, cfg: /* FIXME (#2543) */ copy p.cfg};
    let companionmod = input.filestem().map(|s| Path(*s));
    let (m, attrs) = eval::eval_crate_directives_to_mod(
        cx, cdirs, &prefix, &companionmod);
    let mut hi = p.span.hi;
    p.expect(token::EOF);
    p.abort_if_errors();
    return @ast_util::respan(ast_util::mk_sp(lo, hi),
                          {directives: cdirs,
                           module: m,
                           attrs: vec::append(crate_attrs, attrs),
                           config: /* FIXME (#2543) */ copy p.cfg});
}

fn parse_crate_from_source_file(input: &Path, cfg: ast::crate_cfg,
                                sess: parse_sess) -> @ast::crate {
    let p = new_crate_parser_from_file(sess, cfg, input,
                                       parser::SOURCE_FILE);
    let r = p.parse_crate_mod(cfg);
    return r;
}

fn parse_crate_from_source_str(name: ~str, source: @~str, cfg: ast::crate_cfg,
                               sess: parse_sess) -> @ast::crate {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_crate_mod(cfg);
    p.abort_if_errors();
    return r;
}

fn parse_expr_from_source_str(name: ~str, source: @~str, cfg: ast::crate_cfg,
                              sess: parse_sess) -> @ast::expr {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_expr();
    p.abort_if_errors();
    return r;
}

fn parse_item_from_source_str(name: ~str, source: @~str, cfg: ast::crate_cfg,
                              +attrs: ~[ast::attribute],
                              sess: parse_sess) -> Option<@ast::item> {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_item(attrs);
    p.abort_if_errors();
    return r;
}

fn parse_stmt_from_source_str(name: ~str, source: @~str, cfg: ast::crate_cfg,
                              +attrs: ~[ast::attribute],
                              sess: parse_sess) -> @ast::stmt {
    let p = new_parser_from_source_str(sess, cfg, name,
                                       codemap::FssNone, source);
    let r = p.parse_stmt(attrs);
    p.abort_if_errors();
    return r;
}

fn parse_from_source_str<T>(f: fn (p: Parser) -> T,
                            name: ~str, ss: codemap::FileSubstr,
                            source: @~str, cfg: ast::crate_cfg,
                            sess: parse_sess)
    -> T
{
    let p = new_parser_from_source_str(sess, cfg, name, ss,
                                       source);
    let r = f(p);
    if !p.reader.is_eof() {
        p.reader.fatal(~"expected end-of-string");
    }
    p.abort_if_errors();
    move r
}

fn next_node_id(sess: parse_sess) -> node_id {
    let rv = sess.next_id;
    sess.next_id += 1;
    // ID 0 is reserved for the crate and doesn't actually exist in the AST
    assert rv != 0;
    return rv;
}

fn new_parser_from_source_str(sess: parse_sess, cfg: ast::crate_cfg,
                              +name: ~str, +ss: codemap::FileSubstr,
                              source: @~str) -> Parser {
    let ftype = parser::SOURCE_FILE;
    let filemap = sess.cm.new_filemap_w_substr(name, ss, source);
    let srdr = lexer::new_string_reader(sess.span_diagnostic, filemap,
                                        sess.interner);
    return Parser(sess, cfg, srdr as reader, ftype);
}

fn new_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg,
                        path: &Path,
                        ftype: parser::file_type) -> Result<Parser, ~str> {
    match io::read_whole_file_str(path) {
      result::Ok(move src) => {
          let filemap = sess.cm.new_filemap(path.to_str(), @move src);
          let srdr = lexer::new_string_reader(sess.span_diagnostic, filemap,
                                              sess.interner);

          Ok(Parser(sess, cfg, srdr as reader, ftype))

      }
      result::Err(move e) => Err(move e)
    }
}

/// Create a new parser for an entire crate, handling errors as appropriate
/// if the file doesn't exist
fn new_crate_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg,
                              path: &Path,
                              ftype: parser::file_type) -> Parser {
    match new_parser_from_file(sess, cfg, path, ftype) {
        Ok(move parser) => move parser,
        Err(move e) => {
            sess.span_diagnostic.handler().fatal(e)
        }
    }
}

/// Create a new parser based on a span from an existing parser. Handles
/// error messages correctly when the file does not exist.
fn new_sub_parser_from_file(sess: parse_sess, cfg: ast::crate_cfg,
                            path: &Path, ftype: parser::file_type,
                            sp: span) -> Parser {
    match new_parser_from_file(sess, cfg, path, ftype) {
        Ok(move parser) => move parser,
        Err(move e) => {
            sess.span_diagnostic.span_fatal(sp, e)
        }
    }
}

fn new_parser_from_tt(sess: parse_sess, cfg: ast::crate_cfg,
                      tt: ~[ast::token_tree]) -> Parser {
    let trdr = lexer::new_tt_reader(sess.span_diagnostic, sess.interner,
                                    None, tt);
    return Parser(sess, cfg, trdr as reader, parser::SOURCE_FILE)
}


import front::attr;
import core::{option, result};
import std::{io, fs};
import option::{some, none};
import syntax::ast;
import syntax::parse::token;
import syntax::parse::parser::{parser, new_parser_from_file,
                               parse_inner_attrs_and_next,
                               parse_mod_items, SOURCE_FILE};

export eval_crate_directives_to_mod;

type ctx =
    @{p: parser,
      sess: parser::parse_sess,
      mutable chpos: uint,
      mutable byte_pos: uint,
      cfg: ast::crate_cfg};

fn eval_crate_directives(cx: ctx, cdirs: [@ast::crate_directive], prefix: str,
                         &view_items: [@ast::view_item],
                         &items: [@ast::item]) {
    for sub_cdir: @ast::crate_directive in cdirs {
        eval_crate_directive(cx, sub_cdir, prefix, view_items, items);
    }
}

fn eval_crate_directives_to_mod(cx: ctx, cdirs: [@ast::crate_directive],
                                prefix: str, suffix: option::t<str>)
    -> (ast::_mod, [ast::attribute]) {
    #debug("eval crate prefix: %s", prefix);
    #debug("eval crate suffix: %s",
           option::from_maybe("none", suffix));
    let (cview_items, citems, cattrs)
        = parse_companion_mod(cx, prefix, suffix);
    let view_items: [@ast::view_item] = [];
    let items: [@ast::item] = [];
    eval_crate_directives(cx, cdirs, prefix, view_items, items);
    ret ({view_items: view_items + cview_items,
          items: items + citems},
         cattrs);
}

/*
The 'companion mod'. So .rc crates and directory mod crate directives define
modules but not a .rs file to fill those mods with stuff. The companion mod is
a convention for location a .rs file to go with them.  For .rc files the
companion mod is a .rs file with the same name; for directory mods the
companion mod is a .rs file with the same name as the directory.

We build the path to the companion mod by combining the prefix and the
optional suffix then adding the .rs extension.
*/
fn parse_companion_mod(cx: ctx, prefix: str, suffix: option::t<str>)
    -> ([@ast::view_item], [@ast::item], [ast::attribute]) {

    fn companion_file(prefix: str, suffix: option::t<str>) -> str {
        alt suffix {
          option::some(s) { fs::connect(prefix, s) }
          option::none. { prefix }
        } + ".rs"
    }

    fn file_exists(path: str) -> bool {
        // Crude, but there's no lib function for this and I'm not
        // up to writing it just now
        alt io::file_reader(path) {
          result::ok(_) { true }
          result::err(_) { false }
        }
    }

    let modpath = companion_file(prefix, suffix);
    #debug("looking for companion mod %s", modpath);
    if file_exists(modpath) {
        #debug("found companion mod");
        let p0 = new_parser_from_file(cx.sess, cx.cfg, modpath,
                                     cx.chpos, cx.byte_pos, SOURCE_FILE);
        let inner_attrs = parse_inner_attrs_and_next(p0);
        let first_item_outer_attrs = inner_attrs.next;
        let m0 = parse_mod_items(p0, token::EOF, first_item_outer_attrs);
        cx.chpos = p0.get_chpos();
        cx.byte_pos = p0.get_byte_pos();
        ret (m0.view_items, m0.items, inner_attrs.inner);
    } else {
        ret ([], [], []);
    }
}

fn cdir_path_opt(id: str, attrs: [ast::attribute]) -> str {
    alt attr::get_meta_item_value_str_by_name(attrs, "path") {
      some(d) {
        ret d;
      }
      none. { ret id; }
    }
}

fn eval_crate_directive(cx: ctx, cdir: @ast::crate_directive, prefix: str,
                        &view_items: [@ast::view_item],
                        &items: [@ast::item]) {
    alt cdir.node {
      ast::cdir_src_mod(id, attrs) {
        let file_path = cdir_path_opt(id + ".rs", attrs);
        let full_path =
            if std::fs::path_is_absolute(file_path) {
                file_path
            } else { prefix + std::fs::path_sep() + file_path };
        let p0 =
            new_parser_from_file(cx.sess, cx.cfg, full_path, cx.chpos,
                                 cx.byte_pos, SOURCE_FILE);
        let inner_attrs = parse_inner_attrs_and_next(p0);
        let mod_attrs = attrs + inner_attrs.inner;
        let first_item_outer_attrs = inner_attrs.next;
        let m0 = parse_mod_items(p0, token::EOF, first_item_outer_attrs);

        let i =
            syntax::parse::parser::mk_item(p0, cdir.span.lo, cdir.span.hi, id,
                                           ast::item_mod(m0), mod_attrs);
        // Thread defids, chpos and byte_pos through the parsers
        cx.chpos = p0.get_chpos();
        cx.byte_pos = p0.get_byte_pos();
        items += [i];
      }
      ast::cdir_dir_mod(id, cdirs, attrs) {
        let path = cdir_path_opt(id, attrs);
        let full_path =
            if std::fs::path_is_absolute(path) {
                path
            } else { prefix + std::fs::path_sep() + path };
        let (m0, a0) = eval_crate_directives_to_mod(
            cx, cdirs, full_path, none);
        let i =
            @{ident: id,
              attrs: attrs + a0,
              id: cx.sess.next_id,
              node: ast::item_mod(m0),
              span: cdir.span};
        cx.sess.next_id += 1;
        items += [i];
      }
      ast::cdir_view_item(vi) { view_items += [vi]; }
      ast::cdir_syntax(pth) { }
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

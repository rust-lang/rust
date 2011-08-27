
import std::str;
import std::istr;
import std::option;
import std::option::some;
import std::option::none;
import syntax::ast;
import syntax::parse::token;
import syntax::parse::parser::parser;
import syntax::parse::parser::new_parser_from_file;
import syntax::parse::parser::parse_inner_attrs_and_next;
import syntax::parse::parser::parse_mod_items;
import syntax::parse::parser::SOURCE_FILE;

export eval_crate_directives_to_mod;
export mode_parse;

tag eval_mode { mode_depend; mode_parse; }

type ctx =
    @{p: parser,
      mode: eval_mode,
      mutable deps: [istr],
      sess: parser::parse_sess,
      mutable chpos: uint,
      mutable byte_pos: uint,
      cfg: ast::crate_cfg};

fn eval_crate_directives(cx: ctx, cdirs: &[@ast::crate_directive],
                         prefix: &istr,
                         view_items: &mutable [@ast::view_item],
                         items: &mutable [@ast::item]) {
    for sub_cdir: @ast::crate_directive in cdirs {
        eval_crate_directive(cx, sub_cdir, prefix, view_items, items);
    }
}

fn eval_crate_directives_to_mod(cx: ctx, cdirs: &[@ast::crate_directive],
                                prefix: &istr) -> ast::_mod {
    let view_items: [@ast::view_item] = [];
    let items: [@ast::item] = [];
    eval_crate_directives(cx, cdirs, prefix, view_items, items);
    ret {view_items: view_items, items: items};
}

fn eval_crate_directive(cx: ctx, cdir: @ast::crate_directive, prefix: &istr,
                        view_items: &mutable [@ast::view_item],
                        items: &mutable [@ast::item]) {
    alt cdir.node {
      ast::cdir_src_mod(id, file_opt, attrs) {
        let file_path = id + ~".rs";
        alt file_opt {
          some(f) {
            file_path = f;
          }
          none. { }
        }
        let full_path = if std::fs::path_is_absolute(file_path) {
            file_path
        } else {
            prefix + std::fs::path_sep() + file_path
        };
        if cx.mode == mode_depend { cx.deps += [full_path]; ret; }
        let p0 =
            new_parser_from_file(cx.sess, cx.cfg,
                                 full_path, cx.chpos,
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
      ast::cdir_dir_mod(id, dir_opt, cdirs, attrs) {
        let path = id;
        alt dir_opt {
          some(d) {
            path = d;
          }
          none. { }
        }
        let full_path =
            if std::fs::path_is_absolute(path) {
                path
            } else {
            prefix + std::fs::path_sep() + path
        };
        let m0 = eval_crate_directives_to_mod(cx, cdirs, full_path);
        let i =
            @{ident: id,
              attrs: attrs,
              id: cx.sess.next_id,
              node: ast::item_mod(m0),
              span: cdir.span};
        cx.sess.next_id += 1;
        items += [i];
      }
      ast::cdir_view_item(vi) { view_items += [vi]; }
      ast::cdir_syntax(pth) { }
      ast::cdir_auth(pth, eff) { }
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

use parser::{parser, SOURCE_FILE};
use attr::parser_attr;

export eval_crate_directives_to_mod;

type ctx =
    @{sess: parse::parse_sess,
      cfg: ast::crate_cfg};

fn eval_crate_directives(cx: ctx,
                         cdirs: ~[@ast::crate_directive],
                         prefix: &Path,
                         &view_items: ~[@ast::view_item],
                         &items: ~[@ast::item]) {
    for cdirs.each |sub_cdir| {
        eval_crate_directive(cx, sub_cdir, prefix, view_items, items);
    }
}

fn eval_crate_directives_to_mod(cx: ctx, cdirs: ~[@ast::crate_directive],
                                prefix: &Path, suffix: &Option<Path>)
    -> (ast::_mod, ~[ast::attribute]) {
    let (cview_items, citems, cattrs)
        = parse_companion_mod(cx, prefix, suffix);
    let mut view_items: ~[@ast::view_item] = ~[];
    let mut items: ~[@ast::item] = ~[];
    eval_crate_directives(cx, cdirs, prefix, view_items, items);
    return ({view_items: vec::append(view_items, cview_items),
          items: vec::append(items, citems)},
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
fn parse_companion_mod(cx: ctx, prefix: &Path, suffix: &Option<Path>)
    -> (~[@ast::view_item], ~[@ast::item], ~[ast::attribute]) {

    fn companion_file(prefix: &Path, suffix: &Option<Path>) -> Path {
        return match *suffix {
          option::Some(s) => prefix.push_many(s.components),
          option::None => copy *prefix
        }.with_filetype("rs");
    }

    fn file_exists(path: &Path) -> bool {
        // Crude, but there's no lib function for this and I'm not
        // up to writing it just now
        match io::file_reader(path) {
          result::Ok(_) => true,
          result::Err(_) => false
        }
    }

    let modpath = &companion_file(prefix, suffix);
    if file_exists(modpath) {
        debug!("found companion mod");
        let (p0, r0) = new_parser_etc_from_file(cx.sess, cx.cfg,
                                                modpath, SOURCE_FILE);
        let inner_attrs = p0.parse_inner_attrs_and_next();
        let m0 = p0.parse_mod_items(token::EOF, inner_attrs.next);
        cx.sess.chpos = r0.chpos;
        cx.sess.byte_pos = cx.sess.byte_pos + r0.pos;
        return (m0.view_items, m0.items, inner_attrs.inner);
    } else {
        return (~[], ~[], ~[]);
    }
}

fn cdir_path_opt(default: ~str, attrs: ~[ast::attribute]) -> ~str {
    match ::attr::first_attr_value_str_by_name(attrs, ~"path") {
      Some(d) => d,
      None => default
    }
}

fn eval_crate_directive(cx: ctx, cdir: @ast::crate_directive, prefix: &Path,
                        &view_items: ~[@ast::view_item],
                        &items: ~[@ast::item]) {
    match cdir.node {
      ast::cdir_src_mod(id, attrs) => {
        let file_path = Path(cdir_path_opt(
            cx.sess.interner.get(id) + ~".rs", attrs));
        let full_path = if file_path.is_absolute {
            copy file_path
        } else {
            prefix.push_many(file_path.components)
        };
        let (p0, r0) =
            new_parser_etc_from_file(cx.sess, cx.cfg,
                                     &full_path, SOURCE_FILE);
        let inner_attrs = p0.parse_inner_attrs_and_next();
        let mod_attrs = vec::append(attrs, inner_attrs.inner);
        let first_item_outer_attrs = inner_attrs.next;
        let m0 = p0.parse_mod_items(token::EOF, first_item_outer_attrs);

        let i = p0.mk_item(cdir.span.lo, cdir.span.hi,
                           /* FIXME (#2543) */ copy id,
                           ast::item_mod(m0), ast::public, mod_attrs);
        // Thread defids, chpos and byte_pos through the parsers
        cx.sess.chpos = r0.chpos;
        cx.sess.byte_pos = cx.sess.byte_pos + r0.pos;
        vec::push(items, i);
      }
      ast::cdir_dir_mod(id, cdirs, attrs) => {
        let path = Path(cdir_path_opt(*cx.sess.interner.get(id), attrs));
        let full_path = if path.is_absolute {
            copy path
        } else {
            prefix.push_many(path.components)
        };
        let (m0, a0) = eval_crate_directives_to_mod(
            cx, cdirs, &full_path, &None);
        let i =
            @{ident: /* FIXME (#2543) */ copy id,
              attrs: vec::append(attrs, a0),
              id: cx.sess.next_id,
              node: ast::item_mod(m0),
              vis: ast::public,
              span: cdir.span};
        cx.sess.next_id += 1;
        vec::push(items, i);
      }
      ast::cdir_view_item(vi) => vec::push(view_items, vi),
      ast::cdir_syntax(*) => ()
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

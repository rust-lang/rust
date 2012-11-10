use parser::{Parser, SOURCE_FILE};
use attr::parser_attr;
use ast_util::mk_sp;

export eval_crate_directives_to_mod;
export eval_src_mod;

type ctx =
    @{sess: parse::parse_sess,
      cfg: ast::crate_cfg};

fn eval_crate_directives(cx: ctx,
                         cdirs: ~[@ast::crate_directive],
                         prefix: &Path,
                         view_items: &mut~[@ast::view_item],
                         items: &mut~[@ast::item]) {
    for cdirs.each |sub_cdir| {
        eval_crate_directive(cx, *sub_cdir, prefix, view_items, items);
    }
}

fn eval_crate_directives_to_mod(cx: ctx, cdirs: ~[@ast::crate_directive],
                                prefix: &Path, suffix: &Option<Path>)
    -> (ast::_mod, ~[ast::attribute]) {
    let (cview_items, citems, cattrs)
        = parse_companion_mod(cx, prefix, suffix);
    let mut view_items: ~[@ast::view_item] = ~[];
    let mut items: ~[@ast::item] = ~[];
    eval_crate_directives(cx, cdirs, prefix, &mut view_items, &mut items);
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
        let p0 = new_parser_from_file(cx.sess, cx.cfg,
                                      modpath, SOURCE_FILE);
        let inner_attrs = p0.parse_inner_attrs_and_next();
        let m0 = p0.parse_mod_items(token::EOF, inner_attrs.next);
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

fn eval_src_mod(cx: ctx, prefix: &Path, id: ast::ident,
                outer_attrs: ~[ast::attribute]) -> (ast::item_, ~[ast::attribute]) {
    let file_path = Path(cdir_path_opt(
        cx.sess.interner.get(id) + ~".rs", outer_attrs));
    let full_path = if file_path.is_absolute {
        copy file_path
    } else {
        prefix.push_many(file_path.components)
    };
    let p0 =
        new_parser_from_file(cx.sess, cx.cfg,
                             &full_path, SOURCE_FILE);
    let inner_attrs = p0.parse_inner_attrs_and_next();
    let mod_attrs = vec::append(outer_attrs, inner_attrs.inner);
    let first_item_outer_attrs = inner_attrs.next;
    let m0 = p0.parse_mod_items(token::EOF, first_item_outer_attrs);
    return (ast::item_mod(m0), mod_attrs);
}

// XXX: Duplicated from parser.rs
fn mk_item(ctx: ctx, lo: BytePos, hi: BytePos, +ident: ast::ident,
           +node: ast::item_, vis: ast::visibility,
           +attrs: ~[ast::attribute]) -> @ast::item {
    return @{ident: ident,
             attrs: attrs,
             id: next_node_id(ctx.sess),
             node: node,
             vis: vis,
             span: mk_sp(lo, hi)};
}

fn eval_crate_directive(cx: ctx, cdir: @ast::crate_directive, prefix: &Path,
                        view_items: &mut ~[@ast::view_item],
                        items: &mut ~[@ast::item]) {
    match cdir.node {
      ast::cdir_src_mod(vis, id, attrs) => {
        let (m, mod_attrs) = eval_src_mod(cx, prefix, id, attrs);
        let i = mk_item(cx, cdir.span.lo, cdir.span.hi,
                           /* FIXME (#2543) */ copy id,
                           m, vis, mod_attrs);
        items.push(i);
      }
      ast::cdir_dir_mod(vis, id, cdirs, attrs) => {
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
              vis: vis,
              span: cdir.span};
        cx.sess.next_id += 1;
        items.push(i);
      }
      ast::cdir_view_item(vi) => view_items.push(vi),
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

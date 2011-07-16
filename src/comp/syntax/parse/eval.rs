
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import syntax::ast;
import syntax::parse::token;
import syntax::parse::parser::parser;
import syntax::parse::parser::new_parser_from_file;
import syntax::parse::parser::parse_inner_attrs_and_next;
import syntax::parse::parser::parse_mod_items;

export eval_crate_directives_to_mod;
export mode_parse;

tag eval_mode { mode_depend; mode_parse; }

type ctx =
    @rec(parser p,
         eval_mode mode,
         mutable str[] deps,
         parser::parse_sess sess,
         mutable uint chpos,
         mutable uint byte_pos,
         ast::crate_cfg cfg);

fn eval_crate_directives(ctx cx, &(@ast::crate_directive)[] cdirs,
                         str prefix, &mutable (@ast::view_item)[] view_items,
                         &mutable (@ast::item)[] items) {
    for (@ast::crate_directive sub_cdir in cdirs) {
        eval_crate_directive(cx, sub_cdir, prefix, view_items, items);
    }
}

fn eval_crate_directives_to_mod(ctx cx, &(@ast::crate_directive)[] cdirs,
                                str prefix) -> ast::_mod {
    let (@ast::view_item)[] view_items = ~[];
    let (@ast::item)[] items = ~[];
    eval_crate_directives(cx, cdirs, prefix, view_items, items);
    ret rec(view_items=view_items, items=items);
}

fn eval_crate_directive(ctx cx, @ast::crate_directive cdir, str prefix,
                        &mutable (@ast::view_item)[] view_items,
                        &mutable (@ast::item)[] items) {
    alt (cdir.node) {
        case (ast::cdir_src_mod(?id, ?file_opt, ?attrs)) {
            auto file_path = id + ".rs";
            alt (file_opt) {
                case (some(?f)) { file_path = f; }
                case (none) { }
            }
            auto full_path = if (std::fs::path_is_absolute(file_path)) {
                file_path
            } else {
                prefix + std::fs::path_sep() + file_path
            };
            if (cx.mode == mode_depend) { cx.deps += ~[full_path]; ret; }
            auto p0 =
                new_parser_from_file(cx.sess, cx.cfg, full_path, cx.chpos,
                                     cx.byte_pos);
            auto inner_attrs = parse_inner_attrs_and_next(p0);
            auto mod_attrs = attrs + inner_attrs._0;
            auto first_item_outer_attrs = inner_attrs._1;
            auto m0 = parse_mod_items(p0, token::EOF, first_item_outer_attrs);

            auto i = syntax::parse::parser::mk_item
                (p0, cdir.span.lo, cdir.span.hi, id, ast::item_mod(m0),
                 mod_attrs);
            // Thread defids, chpos and byte_pos through the parsers
            cx.chpos = p0.get_chpos();
            cx.byte_pos = p0.get_byte_pos();
            items += ~[i];
        }
        case (ast::cdir_dir_mod(?id, ?dir_opt, ?cdirs, ?attrs)) {
            auto path = id;
            alt (dir_opt) { case (some(?d)) { path = d; } case (none) { } }
            auto full_path = if (std::fs::path_is_absolute(path)) {
                path
            } else {
                prefix + std::fs::path_sep() + path
            };
            auto m0 = eval_crate_directives_to_mod(cx, cdirs, full_path);
            auto i = @rec(ident=id,
                          attrs=attrs,
                          id=cx.sess.next_id,
                          node=ast::item_mod(m0),
                          span=cdir.span);
            cx.sess.next_id += 1;
            items += ~[i];
        }
        case (ast::cdir_view_item(?vi)) { view_items += ~[vi]; }
        case (ast::cdir_syntax(?pth)) { }
        case (ast::cdir_auth(?pth, ?eff)) { }
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

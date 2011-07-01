
import std::vec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::map::hashmap;
import driver::session;
import ast::ident;
import front::parser::parser;
import front::parser::spanned;
import front::parser::new_parser;
import front::parser::parse_inner_attrs_and_next;
import front::parser::parse_mod_items;
import util::common;
import util::common::filename;
import util::common::span;
import util::common::new_str_hash;


tag eval_mode { mode_depend; mode_parse; }

type ctx =
    @rec(parser p,
         eval_mode mode,
         mutable vec[str] deps,
         session::session sess,
         mutable uint chpos,
         mutable int next_id,
         ast::crate_cfg cfg);

fn eval_crate_directives(ctx cx, vec[@ast::crate_directive] cdirs,
                         str prefix, &mutable vec[@ast::view_item] view_items,
                         &mutable vec[@ast::item] items) {
    for (@ast::crate_directive sub_cdir in cdirs) {
        eval_crate_directive(cx, sub_cdir, prefix, view_items, items);
    }
}

fn eval_crate_directives_to_mod(ctx cx,
                                vec[@ast::crate_directive] cdirs, str prefix)
   -> ast::_mod {
    let vec[@ast::view_item] view_items = [];
    let vec[@ast::item] items = [];
    eval_crate_directives(cx, cdirs, prefix, view_items, items);
    ret rec(view_items=view_items, items=items);
}

fn eval_crate_directive_block(ctx cx, &ast::block blk, str prefix,
                              &mutable vec[@ast::view_item] view_items,
                              &mutable vec[@ast::item] items) {
    for (@ast::stmt s in blk.node.stmts) {
        alt (s.node) {
            case (ast::stmt_crate_directive(?cdir)) {
                eval_crate_directive(cx, cdir, prefix, view_items, items);
            }
            case (_) {
                cx.sess.span_fatal(s.span,
                                 "unsupported stmt in crate-directive block");
            }
        }
    }
}

fn eval_crate_directive(ctx cx, @ast::crate_directive cdir, str prefix,
                        &mutable vec[@ast::view_item] view_items,
                        &mutable vec[@ast::item] items) {
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
            if (cx.mode == mode_depend) { cx.deps += [full_path]; ret; }
            auto p0 =
                new_parser(cx.sess, cx.cfg, full_path, cx.chpos,
                           cx.next_id);
            auto inner_attrs = parse_inner_attrs_and_next(p0);
            auto mod_attrs = attrs + inner_attrs._0;
            auto first_item_outer_attrs = inner_attrs._1;
            auto m0 = parse_mod_items(p0, token::EOF, first_item_outer_attrs);

            auto i = front::parser::mk_item(p0, cdir.span.lo, cdir.span.hi,
                                            id, ast::item_mod(m0),
                                            mod_attrs);
            // Thread defids and chpos through the parsers
            cx.chpos = p0.get_chpos();
            cx.next_id = p0.next_id();
            vec::push[@ast::item](items, i);
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
                          id=cx.next_id,
                          node=ast::item_mod(m0),
                          span=cdir.span);
            cx.next_id += 1;
            vec::push[@ast::item](items, i);
        }
        case (ast::cdir_view_item(?vi)) {
            vec::push[@ast::view_item](view_items, vi);
        }
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

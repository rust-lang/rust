import std::vec;
import std::vec::map;
import std::option;
import std::option::some;
import std::option::none;

import ast::*;
import codemap::span;
import codemap::filename;

tag ast_node {
    branch(node_name, option::t<span>, (@ast_node)[]);
    i_seq((@ast_node)[]);
    i_opt(option::t<@ast_node>);
    l_bool(bool);
    l_ident(ident);
    l_fn_ident(fn_ident);
    l_path(path); //doesn't have to be a leaf, but is effectively atomic
    l_crate_num(crate_num);
    l_node_id(node_id);
    l_def_id(def_id);
    l_ty_param(ty_param);
    l_mutability(mutability);
    l__auth(_auth); //singleton, but needed for uniformity
    l_proto(proto);
    l_binop(binop);
    // moving this to node_name would make the macro system able to abstract
    // over @/@mutable, but that poses parsing problems.
    l_unop(unop);
    l_mode(mode); //same here
    l_init_op(init_op);
    l_inlineness(inlineness);
    l_native_abi(native_abi);
    l_str(str);
    l_char(char);
    l_int(int);
    l_uint(uint);
    l_spawn_dom;
    l_check_mode;
    l_seq_kind;
    l_ty_mach;
    l_purity;
    l_controlflow;
    l_attr_style;

    // these could be avoided, at the cost of making #br_* more convoluted
    l_optional_filename(option::t<filename>);
    l_optional_string(option::t<str>);
    l_seq_ident(ident[]);
    l_seq_ty_param(ty_param[]);

}


tag node_name {
    n_crate_cfg;
    n_crate;

    //crate_directive:
    n_cdir_src_mod;
    n_cdir_dir_mod;
    n_cdir_view_item;
    n_cdir_syntax;
    n_cdir_auth;

    //meta_item:
    n_meta_word;
    n_meta_list;
    n_meta_name_value;

    n_blk;

    //pat: (be sure to add node_id)
    n_pat_wild;
    n_pat_bind;
    n_pat_lit;
    n_pat_tag;
    n_pat_rec;
    n_pat_box;

    n_field_pat;
    //stmt:
    n_stmt_decl;
    n_stmt_expr;
    n_stmt_crate_directive;

    n_initializer;
    n_local;

    //decl:
    n_decl_local;
    n_decl_item;

    n_arm;
    //n_elt; //not used
    n_field;
    //expr: (be sure to add node_id)
    n_expr_vec;
    n_expr_rec;
    n_expr_call;
    n_expr_self_method;
    n_expr_bind;
    n_expr_spawn;
    n_expr_binary;
    n_expr_unary;
    n_expr_lit;
    n_expr_cast;
    n_expr_if;
    n_expr_ternary;
    n_expr_while;
    n_expr_for;
    n_expr_for_each;
    n_expr_do_while;
    n_expr_alt;
    n_expr_fn;
    n_expr_block;
    n_expr_move;
    n_expr_assign;
    n_expr_swap;
    n_expr_assign_op;
    n_expr_send;
    n_expr_recv;
    n_expr_field;
    n_expr_index;
    n_expr_path;
    n_expr_fail;
    n_expr_break;
    n_expr_cont;
    n_expr_ret;
    n_expr_put;
    n_expr_be;
    n_expr_log;
    n_expr_assert;
    n_expr_check;
    n_expr_if_check;
    n_expr_port;
    n_expr_chan;
    n_expr_anon_obj;
    n_expr_mac;

    /*
    n_mac_invoc; //must be expanded away
    n_embed_type; //must be expanded away
    n_embed_bloc; //must be expanded away
    n_ellipsis; //must be expanded away
    */

    //lit:
    n_lit_str;
    n_lit_char;
    n_lit_int;
    n_lit_uint;
    n_lit_mach_int;
    n_lit_float;
    n_lit_mach_float;
    n_lit_nil;
    n_lit_bool;

    n_mt;
    n_ty_field;
    n_ty_arg;
    n_ty_method;

    //ty:
    n_ty_nil;
    //ty_bot; //"there is no syntax for this type"
    n_ty_bool;
    n_ty_int;
    n_ty_uint;
    n_ty_float;
    n_ty_machine;
    n_ty_char;
    n_ty_str;
    n_ty_istr;
    n_ty_box;
    n_ty_vec;
    n_ty_vec;
    n_ty_ptr;
    n_ty_task;
    n_ty_port;
    n_ty_chan;
    n_ty_rec;
    n_ty_fn;
    n_ty_obj;
    n_ty_path;
    n_ty_type;
    n_ty_constr;
    //n_ty_mac; //should be expanded away

    //carg_* and constr are a little funny, since they're type-parametric
    //constr_arg:
    n_carg_base;
    n_carg_ident;
    n_carg_lit;

    n_constr;
    //ack, there's a type and tag with the same name:
    n_ty_constr_theactualconstraint;
    n_arg;
    n_fn_decl;
    n__fn;
    n_method;
    n_obj_field;
    n_anon_obj_field;
    n__obj;
    n_anon_obj;
    n__mod;
    n_native_mod;
    n_variant_arg;
    n_variant;

    //view_item: (it might be necessary to restrict the generation of these)
    n_view_item_use;
    n_view_item_import;
    n_view_item_import_glob;
    n_view_item_export;

    //n_obj_def_ids; //not used
    n_attribute;

    //special case; has ident, attrs, node_id, and span
    n_item;
    //item_:
    n_item_const;
    n_item_fn;
    n_item_mod;
    n_item_native_mod;
    n_item_ty;
    n_item_tag;
    n_item_obj;
    n_item_res;

    //special case; has ident, attrs, node_id, and span
    n_native_item;
    //native_item_:
    n_native_item_ty;
    n_native_item_fn;
}

type ctx = {
    ff: ff,
    sess: driver::session::session, //maybe remove
    next_id: fn() -> node_id
};

/** Type of failure function: to be invoked if typification fails.
    It's hopefully a bug for this to be invoked without a span. */
type ff = fn(sp: option::t<span>, msg: str) -> !;

fn dummy() {

    // arguments explicitly structured here (instead of writing `args, ...`)
    // to make error messages easier to deal with.
    #macro[[#br_alt[val, ctx, [[tagname, n_tagname, [elt, ...]],
                              more_tags, ...]],
            #br_alt_gen[do_span, val, ctx, [[tagname, n_tagname, [elt, ...]],
                                           more_tags, ...]]]];
    #macro[[#br_alt_no_span[val, ctx,
                            [[tagname, n_tagname, [elt, ...]],
                             more_tags, ...]],
            #br_alt_gen[do_not_span, val, ctx,
                        [[tagname, n_tagname, [elt, ...]], more_tags, ...]]]];

    // the silly `n_tagname` will be replacable with an invocation of
    // #concat_idents after subtitution into arbitrary AST nodes works
    #macro[[#br_alt_gen[possibly_span, val, ctx,
                        [[tagname, n_tagname, [elt, ...]], more_tags, ...]],
            alt val {
              branch(n_tagname., sp, chldrn) {
                #possibly_span[sp, #br_alt_core[ctx, sp, tagname,
                                                chldrn, 0u, [elt, ...], []]]
              }
              //replace this explicit recursion with `...`, once it works
              //over all kinds of AST nodes.
              _ { #br_alt_gen[possibly_span, val, ctx, [more_tags, ...]] }
            }
           ],
           [#br_alt_gen[possibly_span, val, ctx, []],
            alt val {
              branch(_, sp, _) {
                ctx.ff(sp, "expected " + #ident_to_str[tagname])
              }
              _ { ctx.ff(none, "expected " + #ident_to_str[tagname]) }
            }
           ]];

    // this wackiness is just because we need indices
    #macro[[#br_alt_core[ctx, sp, ctor, kids, offset, [h, more_hs, ...],
                         [accum, ...]],
            #br_alt_core[ctx, sp, ctor, kids, offset+1u, [more_hs, ...],
                         [accum, ...,
                          #extract_elt[ctx, sp, kids, offset, h]]]],
           [#br_alt_core[ctx, sp, ctor, kids, offset, [], [accum, ...]],
            ctor(accum, ...)]];



    #macro[[#br_rec[args, ...], #br_rec_gen[do_span, args, ...]]];
    #macro[[#br_rec_no_span[args, ...], #br_rec_gen[do_not_span, args, ...]]];


    #macro[[#br_rec_gen[possibly_span, val, ctx, tagname, n_tagname, fields],
            alt val {
              branch(n_tagname., sp, chldrn) {
                #possibly_span[sp, #br_rec_core[ctx, sp, chldrn, fields]]
              }
              branch(_, sp, _) {
                ctx.ff(sp, "expected " + #ident_to_str[tagname])
              }
              _ { ctx.ff(option::none, "expected " + #ident_to_str[tagname]) }
            }]];

    #macro[[#do_span[sp, node_val],
            {node: node_val,
             span: alt sp {
               some(s) { s }
               none. { ctx.ff(none, "needed a span"); }
             }
            }]];
    #macro[[#do_not_span[sp, node_val], node_val]];


    //this abomination can go away when `...` works properly over
    //all kinds of AST nodes.
    #macro[[#br_rec_core[ctx, sp, kids, [[f0, fh0]]],
            {f0: #extract_elt[ctx, sp, kids, 0u, fh0]}],
           [#br_rec_core[ctx, sp, kids, [[f0, fh0], [f1, fh1]]],
            {f0: #extract_elt[ctx, sp, kids, 0u, fh0],
             f1: #extract_elt[ctx, sp, kids, 1u, fh1]}],
           [#br_rec_core[ctx, sp, kids, [[f0, fh0], [f1, fh1], [f2, fh2]]],
            {f0: #extract_elt[ctx, sp, kids, 0u, fh0],
             f1: #extract_elt[ctx, sp, kids, 1u, fh1],
             f2: #extract_elt[ctx, sp, kids, 2u, fh2]}],
           [#br_rec_core[ctx, sp, kids,
                         [[f0, fh0], [f1, fh1], [f2, fh2], [f3, fh3]]],
            {f0: #extract_elt[ctx, sp, kids, 0u, fh0],
             f1: #extract_elt[ctx, sp, kids, 1u, fh1],
             f2: #extract_elt[ctx, sp, kids, 2u, fh2],
             f3: #extract_elt[ctx, sp, kids, 3u, fh3]}],
           [#br_rec_core[ctx, sp, kids,
                         [[f0, fh0], [f1, fh1], [f2, fh2], [f3, fh3],
                          [f4, fh4]]],
            {f0: #extract_elt[ctx, sp, kids, 0u, fh0],
             f1: #extract_elt[ctx, sp, kids, 1u, fh1],
             f2: #extract_elt[ctx, sp, kids, 2u, fh2],
             f3: #extract_elt[ctx, sp, kids, 3u, fh3],
             f4: #extract_elt[ctx, sp, kids, 4u, fh4]}],
           [#br_rec_core[ctx, sp, kids,
                         [[f0, fh0], [f1, fh1], [f2, fh2], [f3, fh3],
                          [f4, fh4], [f5, fh5]]],
            {f0: #extract_elt[ctx, sp, kids, 0u, fh0],
             f1: #extract_elt[ctx, sp, kids, 1u, fh1],
             f2: #extract_elt[ctx, sp, kids, 2u, fh2],
             f3: #extract_elt[ctx, sp, kids, 3u, fh3],
             f4: #extract_elt[ctx, sp, kids, 4u, fh4],
             f5: #extract_elt[ctx, sp, kids, 5u, fh5]}]];

    #macro[ //keywords would make these two nicer:
        [#extract_elt[ctx, sp, elts, idx, []], ctx.next_id()],
        [#extract_elt[ctx, cp, elts, idx, [[]]], sp],
        [#extract_elt[ctx, sp, elts, idx, [leaf_destructure]],
         alt *elts.(idx) {
           leaf_destructure(x) { x }
           _ {
             ctx.ff(sp, #fmt["expected %s in position %u",
                             #ident_to_str[leaf_destructure], idx])
           }
         }],
        [#extract_elt[ctx, sp, elts, idx, extract_fn],
         extract_fn(ctx, elts.(idx))]];
}

fn seq_cv<T>(conversion: fn (&ctx, &@ast_node) -> T)
    -> fn (&ctx, @ast_node) -> T[] {
    ret lambda(ctx: &ctx, ut: @ast_node) -> T[] {
        ret alt *ut {
          i_seq(uts) { map(bind conversion(ctx, _), uts) }
          branch(_, sp, _) {
            ctx.ff(sp, "expected a sequence, found a branch");
          }
          _ { ctx.ff(none, "expected a sequence"); }
        }
    }
}

fn opt_cv<T>(conversion: fn (&ctx, &@ast_node) -> T)
    -> fn (&ctx, @ast_node) -> option::t<T> {
    ret lambda(ctx: &ctx, ut: @ast_node) -> option::t<T> {
        ret alt *ut {
          i_opt(ut_maybe) { option::map(bind conversion(ctx, _), ut_maybe) }
          branch(_, sp, _) {
            ctx.ff(sp, "expected a sequence, found a branch");
          }
          _ { ctx.ff(none, "expected a sequence"); }
        }
    }
}


// expect leaf/branch fn × scalar/vec/option

fn cv_crate(ctx: &ctx, ut: &@ast_node) -> @crate {
    ret @#br_rec[*ut, ctx, crate, n_crate,
                 [[directives, seq_cv(cv_crate_directive)],
                  [module, cv__mod],
                  [attrs, seq_cv(cv_attribute)],
                  [config, cv_crate_cfg]]];
}

fn cv_crate_cfg(ctx: &ctx, ut: &@ast_node) -> crate_cfg {
    ret alt *ut {
      branch(n_crate_cfg., _, meta_items) {
        vec::map(bind cv_meta_item(ctx,_), meta_items)
      }
      branch(_, sp, _) { ctx.ff(sp,"Invalid crate_cfg") }
    };
}

fn cv_crate_directive(ctx: &ctx, ut: &@ast_node) -> @crate_directive {
    ret @#br_alt[*ut, ctx,
                 [[cdir_src_mod, n_cdir_src_mod,
                   [[l_ident], [l_optional_filename], seq_cv(cv_attribute)]],
                  [cdir_dir_mod, n_cdir_dir_mod,
                   [[l_ident], [l_optional_filename],
                    seq_cv(cv_crate_directive),
                    seq_cv(cv_attribute)]],
                  [cdir_view_item, n_cdir_view_item, [cv_view_item]],
                  [cdir_syntax, n_cdir_syntax,       [[l_path]]],
                  [cdir_auth, n_cdir_auth,           [[l_path], [l__auth]]]]];
}

fn cv_meta_item(ctx: &ctx, ut: &@ast_node) -> @meta_item {
    ret @#br_alt[*ut, ctx,
                 [[meta_word, n_meta_word,             [[l_ident]]],
                  [meta_list, n_meta_list,
                   [[l_ident], seq_cv(cv_meta_item)]],
                  [meta_name_value, n_meta_name_value, [[l_ident], cv_lit]]]];
}

fn cv_blk(ctx: &ctx, ut: &@ast_node) -> blk {
    ret #br_rec[*ut, ctx, blk, n_blk,
                [[stmts, seq_cv(cv_stmt)],
                 [expr,  opt_cv(cv_stmt)],
                 [id,    [l_node_id]]]];
}

fn cv_pat(ctx: &ctx, ut: &@ast_node) -> @pat {
    ret @{id: ctx.next_id(),
          node: #br_alt_no_span
              [*ut, ctx,
               [[pat_wild, n_pat_wild, []],
                [pat_bind, n_pat_bind, [[l_ident]]],
                [pat_tag, n_pat_tag,   [[l_ident], seq_cv(cv_pat)]],
                [pat_rec, n_pat_rec,   [seq_cv(cv_field_pat),
                                        [l_bool]]],
                [pat_box, n_pat_box,   [cv_pat]]]],
          span: alt *ut {
            branch(_,some(sp),_) { sp }
            none. { ctx.ff("pat needs a span"); }
          }
         }
}

fn cv_field_pat(ctx: &ctx, ut: &@ast_node) -> field_pat {
    ret #br_rec[*ut, ctx, field_pat, n_field_pat,
                [[ident, [l_ident]], [pat, cv_pat]]];
}

fn cv_stmt(ctx: &ctx, ut: &@ast_node) -> @stmt {
    ret @#br_alt[*ut, ctx,
                 [[stmt_decl, n_stmt_decl, [cv_decl, []]],
                  [stmt_expr, n_stmt_expr, [cv_expr, []]],
                  [stmt_crate_directive, n_stmt_crate_directive,
                   [cv_crate_directive]]]];
}

fn cv_initializer(ctx: &ctx, ut: &@ast_node) -> initializer {
    ret #br_rec[*ut, ctx, initializer, n_initializer,
               [[op,   [l_init_op]],
                [expr, cv_expr]]];
}

fn cv_local(ctx: &ctx, ut: &@ast_node) -> @local {
    ret @#br_rec[*ut, ctx, local, n_local,
                 [[ty,   opt_cv(cv_ty)],
                  [pat,  cv_pat],
                  [init, opt_cv(cv_initializer)],
                  [id,   []]]]
}

fn cv_decl(ctx: &ctx, ut: &@ast_node) -> @decl {
    ret @#br_alt[*ut, ctx,
                 [[decl_local, n_decl_local, [cv_local]],
                  [decl_item, n_decl_item, [cv_item]]]];
}

fn cv_arm(ctx: &ctx, ut: &@ast_node) -> arm {
    ret #br_rec[*ut, ctx, arm, n_arm,
                [[pats, seq_cv(cv_pat)],
                 [body, cv_blk]]];
}

fn cv_field(ctx: &ctx, ut: &@ast_node) -> field {
    ret #br_rec[*ut, ctx, field, n_field,
                [[mut,   [l_mutability]],
                 [ident, [l_ident]],
                 [expr,  cv_expr]]];
}

fn cv_expr(ctx: &ctx, ut: &@ast_node) -> @expr {
    ret @{id: ctx.next_id(),
          node: #br_alt_no_span
              [*ut, ctx,
               [[expr_vec, n_expr_vec,
                 [seq_cv(cv_expr), [l_mutability], [l_seq_kind]]],
                [expr_rec, n_expr_rec,
                 [seq_cv(cv_field), opt_cv(cv_expr)]],
                [expr_call, n_expr_call,
                 [cv_expr, seq_cv(cv_expr)]],
                [expr_self_method, n_expr_self_method, [[l_ident]]],
                [expr_bind, n_expr_bind,
                 [cv_expr, seq_cv(opt_cv(cv_expr))]],
                [expr_spawn, n_expr_spawn,
                 [[l_spawn_dom], [l_optional_string], cv_expr,
                  seq_cv(cv_expr)]],
                [expr_binary, n_expr_binary, [[l_binop], cv_expr, cv_expr]],
                [expr_unary, n_expr_unary, [[l_unop], cv_expr]],
                [expr_lit, n_expr_lit, [cv_lit]],
                [expr_cast, n_expr_cast, [cv_expr, cv_ty]],
                [expr_if, n_expr_if, [cv_expr, cv_blk, opt_cv(cv_expr)]],
                [expr_ternary, n_expr_ternary, [cv_expr, cv_expr, cv_expr]],
                [expr_while, n_expr_while, [cv_expr, cv_blk]],
                [expr_for, n_expr_for, [cv_local, cv_expr, cv_blk]],
                [expr_for_each, n_expr_for_each, [cv_local, cv_expr, cv_blk]],
                [expr_do_while, n_expr_do_while, [cv_blk, cv_expr]],
                [expr_alt, n_expr_alt, [cv_expr, seq_cv(cv_arm)]],
                [expr_fn, n_expr_fn, [cv__fn]],
                [expr_block, n_expr_block, [cv_expr]],
                [expr_move, n_expr_move, [cv_expr, cv_expr]],
                [expr_assign, n_expr_assign, [cv_expr, cv_expr]],
                [expr_swap, n_expr_swap, [cv_expr, cv_expr]],
                [expr_assign_op, n_expr_assign_op,
                 [[l_binop], cv_expr, cv_expr]],
                [expr_send, n_expr_send, [cv_expr, cv_expr]],
                [expr_recv, n_expr_recv, [cv_expr, cv_expr]],
                [expr_field, n_expr_field, [cv_expr, [l_ident]]],
                [expr_index, n_expr_index, [cv_expr, cv_expr]],
                [expr_path, n_expr_path, [[l_path]]],
                [expr_fail, n_expr_fail, [opt_cv(cv_expr)]],
                [expr_break, n_expr_break, []],
                [expr_cont, n_expr_cont, []],
                [expr_ret, n_expr_ret, [opt_cv(cv_expr)]],
                [expr_put, n_expr_put, [opt_cv(cv_expr)]],
                [expr_be, n_expr_be, [cv_expr]],
                [expr_log, n_expr_log, [cv_expr]],
                [expr_assert, n_expr_assert, [cv_expr]],
                [expr_check, n_expr_check, [cv_expr]],
                [expr_if_check, n_expr_if_check,
                 [cv_expr, cv_blk, opt_cv(cv_expr)]],
                [expr_port, n_expr_port, [opt_cv(cv_ty)]],
                [expr_chan, n_expr_chan, [cv_expr]],
                [expr_anon_obj, n_expr_anon_obj, [cv_anon_obj]]]
              ],
          span: alt *ut {
            branch(_,some(sp),_) { sp }
            none. { ctx.ff("pat needs a span"); }
          }
         }
}


fn cv_lit(ctx: &ctx, ut: &@ast_node) -> @lit {
    ret @#br_alt[*ut, ctx,
                 [[lit_str, n_lit_str, [[l_str], [l_seq_kind]]],
                  [lit_char, n_lit_char, [[l_char]]],
                  [lit_int, n_lit_int, [[l_int]]],
                  [lit_uint, n_lit_uint, [[l_uint]]],
                  [lit_mach_int, n_lit_mach_int, [[l_ty_mach], [l_int]]],
                  [lit_float, n_lit_float, [[l_str]]],
                  [lit_mach_float, n_lit_mach_float, [[l_ty_mach], [l_str]]],
                  [lit_nil, n_lit_nil, []],
                  [lit_bool, n_lit_bool, [[l_bool]]]]];
}

fn cv_mt(ctx: &ctx, ut: &@ast_node) -> mt {
    ret #br_rec_no_span[*ut, ctx, mt, n_mt,
                        [[ty, cv_ty], [mut, [l_mutability]]]];
}

fn cv_ty_field(ctx: &ctx, ut: &@ast_node) -> ty_field {
    ret #br_rec[*ut, ctx, ty_field, n_ty_field,
                [[ident, [l_ident]], [mt, cv_mt]]];
}

fn cv_ty_arg(ctx: &ctx, ut: &@ast_node) -> ty_arg {
    ret #br_rec[*ut, ctx, ty_arg, n_ty_arg,
                [[mode, [l_mode]], [ty, cv_ty]]];
}

fn cv_ty_method(ctx: &ctx, ut: &@ast_node) -> ty_method {
    ret #br_rec[*ut, ctx, ty_method, n_ty_method,
                [[proto, [l_proto]],
                 [ident, [l_ident]],
                 [inputs, seq_cv(cv_ty_arg)],
                 [output, cv_ty],
                 [cf, [l_controlflow]],
                 [constrs, seq_cv(cv_constr)]]];
}

fn cv_ty(ctx: &ctx, ut: &@ast_node) -> @ty {
    ret @#br_alt[*ut, ctx,
                 [[ty_nil, n_ty_nil, []],
                  [ty_bool, n_ty_bool, []],
                  [ty_int, n_ty_int, []],
                  [ty_uint, n_ty_uint, []],
                  [ty_float, n_ty_float, []],
                  [ty_machine, n_ty_machine, [[l_ty_mach]]],
                  [ty_char, n_ty_char, []],
                  [ty_str, n_ty_str, []],
                  [ty_istr, n_ty_istr, []],
                  [ty_box, n_ty_box, [cv_mt]],
                  [ty_vec, n_ty_vec, [cv_mt]],
                  [ty_vec, n_ty_vec, [cv_mt]],
                  [ty_ptr, n_ty_ptr, [cv_mt]],
                  [ty_task, n_ty_task, []],
                  [ty_port, n_ty_port, [cv_ty]],
                  [ty_chan, n_ty_chan, [cv_ty]],
                  [ty_rec, n_ty_rec, [seq_cv(cv_ty_field)]],
                  [ty_fn, n_ty_fn,
                   [[l_proto], seq_cv(cv_arg), cv_ty, [l_controlflow],
                    seq_cv(cv_constr)]],
                  [ty_obj, n_ty_obj, [seq_cv(cv_ty_method)]],
                  [ty_path, n_ty_path, [[l_path], [] /*node_id*/]],
                  [ty_type, n_ty_type, []],
                  [ty_constr, n_ty_constr, [cv_ty, seq_cv(cv_constr)]]]]
}

/* these four are expanded from the type-parametric code in ast.rs */

fn cv_carg_uint(ctx: &ctx, ut: &@ast_node) -> constr_arg_general_[uint] {
    ret #br_alt[*ut, ctx,
                [[carg_base, n_carg_base, []],
                 [carg_ident, n_carg_ident, [[l_uint]]],
                 [carg_lit, n_carg_lit, [cv_lit]]]];
}
fn cv_carg_path(ctx: &ctx, ut: &@ast_node) -> constr_arg_general_[path] {
    ret #br_alt[*ut, ctx,
                [[carg_base, n_carg_base, []],
                 [carg_ident, n_carg_ident, [[l_path]]],
                 [carg_lit, n_carg_lit, [cv_lit]]]];
}

fn cv_constr(ctx: &ctx, ut: &@ast_node) -> @constr {
    ret @#br_rec[*ut, ctx, constr, n_constr,
                 [[path, [l_path]],
                  [args, cv_carg_uint],
                  [id, [] /*node_id*/]]];
}
fn cv_typed_constr(ctx: &ctx, ut: &@ast_node) -> @ty_constr {
    ret @#br_rec[*ut, ctx, ty_constr, n_ty_constr_theactualconstraint,
                 [[path, [l_path]],
                  [args, cv_carg_path],
                  [id, [] /*node_id*/]]];
}


fn cv_arg(ctx: &ctx, ut: &@ast_node) -> arg {
    ret #br_rec_no_span[*ut, ctx, arg, n_arg,
                        [[mode, [l_mode]],
                         [ty, cv_ty],
                         [ident, [l_ident]],
                         [id, [] /*node_id*/]]];
}

fn cv_fn_decl(ctx: &ctx, ut: &@ast_node) -> fn_decl {
    ret #br_rec_no_span[*ut, ctx, fn_decl, n_fn_decl,
                        [[inputs, seq_cv(cv_arg)],
                         [output, cv_ty],
                         [il, [l_inlineness]],
                         [cf, [l_controlflow]],
                         [constraints, seq_cv(cv_constr)]]];
}

fn cv__fn(ctx: &ctx, ut: &@ast_node) -> _fn {
    ret #br_rec_no_span[*ut, ctx, _fn, n__fn,
                        [[decl, cv_fn_decl],
                         [proto, [l_proto]],
                         [body, cv_blk]]];
}

fn cv_method(ctx: &ctx, ut: &@ast_node) -> @method {
    ret @#br_rec[*ut, ctx, method, n_method,
                 [[ident, [l_ident]],
                  [meth, cv__fn],
                  [id, [] /*node_id*/]]];
}

fn cv_obj_field(ctx: &ctx, ut: &@ast_node) -> obj_field {
    ret #br_rec[*ut, ctx, obj_field, n_obj_field,
                [[mut, [l_mutability]],
                 [ty, cv_ty],
                 [ident, [l_ident]],
                 [id, [] /*node_id*/]]];
}

fn cv_anon_obj_field(ctx: &ctx, ut: &@ast_node) -> anon_obj_field {
    ret #br_rec[*ut, ctx, anon_obj_field, n_anon_obj_field,
                [[mut, [l_mutability]],
                 [ty, cv_ty],
                 [expr, cv_expr],
                 [ident, [l_ident]],
                 [id, [] /*node_id*/]]];
}

fn cv__obj(ctx: &ctx, ut: &@ast_node) ->  _obj {
    ret #br_rec_no_span[*ut, ctx, _obj, n__obj,
                        [[fields, seq_cv(cv_obj_field)],
                         [methods, seq_cv(cv_method)]]];
}

fn cv_anon_obj(ctx: &ctx, ut: &@ast_node) ->  anon_obj {
    ret #br_rec_no_span[*ut, ctx, anon_obj, n_anon_obj,
                        [[fields, opt_cv(seq_cv(cv_anon_obj_field))],
                         [methods, seq_cv(cv_method)],
                         [inner_obj, opt_cv(cv_expr)]]];
}

fn cv__mod(ctx: &ctx, ut: &@ast_node) -> _mod {
    ret #br_rec_no_span[*ut, ctx, _mod, n__mod,
                        [[view_items, seq_cv(cv_view_item)],
                         [items, seq_cv(cv_item)]]];
}

fn cv_native_mod(ctx: &ctx, ut: &@ast_node) -> native_mod {
    ret #br_rec_no_span[*ut, ctx, native_mod, n_native_mod,
                        [[native_name, [l_str]],
                         [abi, [l_native_abi]],
                         [view_items, seq_cv(cv_view_item)],
                         [items, seq_cv(cv_item)]]];
}

fn cv_variant_arg(ctx: &ctx, ut: &@ast_node) -> variant_arg {
    ret #br_rec_no_span[*ut, ctx, variant_arg, n_variant_arg,
                        [[ty, cv_ty],
                         [id, [] /*node_id*/]]];
}

fn cv_variant(ctx: &ctx, ut: &@ast_node) -> variant {
    ret #br_rec[*ut, ctx, variant, n_variant,
                [[name, [l_str]],
                 [args, seq_cv(cv_variant_arg)],
                 [id, [] /*node_id*/]]];
}

fn cv_view_item(ctx: &ctx, ut: &@ast_node) -> @view_item {
    ret @#br_alt[*ut, ctx,
                 [[view_item_use, n_view_item_use,
                   [[l_ident], seq_cv(cv_meta_item), [] /*node_id*/]],
                  [view_item_import, n_view_item_import,
                   [[l_ident], [l_seq_ident], []]],
                  [view_item_import_glob, n_view_item_import_glob,
                   [[l_seq_ident], []]],
                  [view_item_export, n_view_item_export,
                   [[l_ident], []]]]];
}

fn cv_attribute(ctx: &ctx, ut: &@ast_node) -> attribute {
    ret #br_rec[*ut, ctx, attribute, n_attribute,
                [[style, [l_attr_style]],
                 [value, cv_meta_item]]];
}

/* item and native_item have large enough wrappers that their underscored
   components get separate handling */

fn cv_item(ctx: &ctx, ut: &@ast_node) -> @item {
    ret @#br_rec[*ut, ctx, item, n_item,
                 [[ident, [l_ident]],
                  [attrs, seq_cv(cv_attribute)],
                  [id, [] /*node_id*/],
                  [node, cv_item_],
                  [span, [[]] /*span*/]]];
}

fn cv_item_(ctx: &ctx, ut: &@ast_node) -> item_ {
    ret #br_alt_no_span[*ut, ctx,
                        [[item_const, n_item_const, [cv_ty, cv_expr]],
                         [item_fn, n_item_fn, [cv__fn, [l_seq_ty_param]]],
                         [item_mod, n_item_mod, [cv__mod]],
                         [item_native_mod, n_item_native_mod,
                          [cv_native_mod]],
                         [item_ty, n_item_ty, [cv_ty, [l_seq_ty_param]]],
                         [item_tag, n_item_tag,
                          [seq_cv(cv_variant), [l_seq_ty_param]]],
                         [item_obj, n_item_obj,
                          [cv__obj, [l_seq_ty_param], [] /*node_id*/]],
                         [item_res, n_item_res,
                          [cv__fn, [] /*node_id*/, [l_seq_ty_param],
                           [] /*node_id*/]]]];
}

fn cv_native_item(ctx: &ctx, ut: &@ast_node) -> @native_item {
    ret @#br_rec[*ut, ctx, native_item, n_native_item,
                 [[ident, [l_ident]],
                  [attrs, seq_cv(cv_attribute)],
                  [node, cv_native_item_],
                  [id, [] /*node_id*/],
                  [span, [[]] /*span*/]]];
}

fn cv_native_item_(ctx: &ctx, ut: &@ast_node) -> native_item_ {
    ret #br_alt[*ut, ctx,
                [[native_item_ty, n_native_item_ty, []],
                 [native_item_fn, n_native_item_fn,
                  [[l_optional_string], cv_fn_decl, [l_seq_ty_param]]]]];
}

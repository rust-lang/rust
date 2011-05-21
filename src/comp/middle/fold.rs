import std::map::hashmap;
import std::option;
import std::option::some;
import std::option::none;

import util::common::new_str_hash;
import util::common::spanned;
import util::common::span;
import util::common::ty_mach;
import middle::tstate::ann::ts_ann;

import front::ast;
import front::ast::fn_decl;
import front::ast::ident;
import front::ast::path;
import front::ast::mutability;
import front::ast::controlflow;
import front::ast::ty;
import front::ast::expr;
import front::ast::stmt;
import front::ast::block;
import front::ast::item;
import front::ast::view_item;
import front::ast::meta_item;
import front::ast::native_item;
import front::ast::arg;
import front::ast::pat;
import front::ast::decl;
import front::ast::arm;
import front::ast::def;
import front::ast::def_id;
import front::ast::ann;
import front::ast::mt;
import front::ast::purity;

import std::uint;
import std::vec;

type ast_fold[ENV] =
    @rec
    (
     // Path fold:
     (fn(&ENV e, &span sp, &ast::path_ p) 
      -> path)                                    fold_path,

     // Type folds.
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_nil,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_bot,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_bool,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_int,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_uint,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_float,
     (fn(&ENV e, &span sp, ty_mach tm) -> @ty)    fold_ty_machine,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_char,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_str,
     (fn(&ENV e, &span sp, &mt tm) -> @ty)        fold_ty_box,
     (fn(&ENV e, &span sp, &mt tm) -> @ty)        fold_ty_vec,

     (fn(&ENV e, &span sp, &vec[mt] elts) -> @ty) fold_ty_tup,

     (fn(&ENV e, &span sp,
         &vec[ast::ty_field] elts) -> @ty)        fold_ty_rec,

     (fn(&ENV e, &span sp,
         &vec[ast::ty_method] meths) -> @ty)      fold_ty_obj,

     (fn(&ENV e, &span sp,
         ast::proto proto,
         &vec[rec(ast::mode mode, @ty ty)] inputs,
         &@ty output, &controlflow cf) -> @ty)    fold_ty_fn,

     (fn(&ENV e, &span sp, &ast::path p,
         &ann a) -> @ty)                          fold_ty_path,

     (fn(&ENV e, &span sp, &@ty t) -> @ty)        fold_ty_chan,
     (fn(&ENV e, &span sp, &@ty t) -> @ty)        fold_ty_port,

     // Expr folds.
     (fn(&ENV e, &span sp,
         &vec[@expr] es, ast::mutability mut,
         &ann a) -> @expr)                        fold_expr_vec,

     (fn(&ENV e, &span sp,
         &vec[ast::elt] es, &ann a) -> @expr)     fold_expr_tup,

     (fn(&ENV e, &span sp,
         &vec[ast::field] fields,
         &option::t[@expr] base, &ann a) -> @expr) fold_expr_rec,

     (fn(&ENV e, &span sp,
         &@expr f, &vec[@expr] args,
         &ann a) -> @expr)                        fold_expr_call,

     (fn(&ENV e, &span sp,
         &ident id, &ann a) -> @expr)             fold_expr_self_method,

     (fn(&ENV e, &span sp,
         &@expr f, &vec[option::t[@expr]] args,
         &ann a) -> @expr)                        fold_expr_bind,

     (fn(&ENV e, &span sp,
         ast::spawn_dom dom, &option::t[str] name,
         &@expr f, &vec[@expr] args,
         &ann a) -> @expr)                        fold_expr_spawn,

     (fn(&ENV e, &span sp,
         ast::binop,
         &@expr lhs, &@expr rhs,
         &ann a) -> @expr)                        fold_expr_binary,

     (fn(&ENV e, &span sp,
         ast::unop, &@expr e,
         &ann a) -> @expr)                        fold_expr_unary,

     (fn(&ENV e, &span sp,
         &@ast::lit, &ann a) -> @expr)            fold_expr_lit,

     (fn(&ENV e, &span sp,
         &@ast::expr e, &@ast::ty ty,
         &ann a) -> @expr)                        fold_expr_cast,

     (fn(&ENV e, &span sp,
         &@expr cond, &block thn,
         &option::t[@expr] els,
         &ann a) -> @expr)                        fold_expr_if,

     (fn(&ENV e, &span sp,
         &@decl decl, &@expr seq, &block body,
         &ann a) -> @expr)                        fold_expr_for,

     (fn(&ENV e, &span sp,
         &@decl decl, &@expr seq, &block body,
         &ann a) -> @expr)                        fold_expr_for_each,

     (fn(&ENV e, &span sp,
         &@expr cond, &block body,
         &ann a) -> @expr)                        fold_expr_while,

     (fn(&ENV e, &span sp,
         &block body, &@expr cond,
         &ann a) -> @expr)                        fold_expr_do_while,

     (fn(&ENV e, &span sp,
         &@expr e, &vec[arm] arms,
         &ann a) -> @expr)                        fold_expr_alt,

     (fn(&ENV e, &span sp,
         &block blk, &ann a) -> @expr)            fold_expr_block,

     (fn(&ENV e, &span sp,
         &@expr lhs, &@expr rhs,
         &ann a) -> @expr)                        fold_expr_assign,

     (fn(&ENV e, &span sp,
         ast::binop,
         &@expr lhs, &@expr rhs,
         &ann a) -> @expr)                        fold_expr_assign_op,

     (fn(&ENV e, &span sp,
         &@expr lhs, &@expr rhs,
         &ann a) -> @expr)                        fold_expr_send,

     (fn(&ENV e, &span sp,
         &@expr lhs, &@expr rhs,
         &ann a) -> @expr)                        fold_expr_recv,

     (fn(&ENV e, &span sp,
         &@expr e, &ident i,
         &ann a) -> @expr)                        fold_expr_field,

     (fn(&ENV e, &span sp,
         &@expr e, &@expr ix,
         &ann a) -> @expr)                        fold_expr_index,

     (fn(&ENV e, &span sp,
         &path p,
         &ann a) -> @expr)                        fold_expr_path,

     (fn(&ENV e, &span sp,
         &path p, &vec[@expr] args,
         &option::t[str] body,
         &@expr expanded,
         &ann a) -> @expr)                        fold_expr_ext,

     (fn(&ENV e, &span sp, &ann a) -> @expr)      fold_expr_fail,

     (fn(&ENV e, &span sp, &ann a) -> @expr)      fold_expr_break,

     (fn(&ENV e, &span sp, &ann a) -> @expr)      fold_expr_cont,

     (fn(&ENV e, &span sp,
         &option::t[@expr] rv, &ann a) -> @expr)  fold_expr_ret,

     (fn(&ENV e, &span sp,
         &option::t[@expr] rv, &ann a) -> @expr)  fold_expr_put,

     (fn(&ENV e, &span sp,
         &@expr e, &ann a) -> @expr)              fold_expr_be,

     (fn(&ENV e, &span sp, int lvl,
         &@expr e, &ann a) -> @expr)              fold_expr_log,

     (fn(&ENV e, &span sp,
         &@expr e, &ann a) -> @expr)              fold_expr_check,

     (fn(&ENV e, &span sp,
         &@expr e, &ann a) -> @expr)              fold_expr_assert,

     (fn(&ENV e, &span sp,
         &ann a) -> @expr)                        fold_expr_port,

     (fn(&ENV e, &span sp,
         &@expr e, &ann a) -> @expr)              fold_expr_chan,

     (fn(&ENV e, &span sp,
         &ast::anon_obj ob,
         &vec[ast::ty_param] tps,
         &ast::obj_def_ids odid, 
         &ann a) -> @expr)                        fold_expr_anon_obj,

     // Decl folds.
     (fn(&ENV e, &span sp,
         &@ast::local local) -> @decl)            fold_decl_local,

     (fn(&ENV e, &span sp,
         &@item item) -> @decl)                   fold_decl_item,


     // Pat folds.
     (fn(&ENV e, &span sp,
         &ann a) -> @pat)                         fold_pat_wild,

     (fn(&ENV e, &span sp,
         &@ast::lit lit, &ann a) -> @pat)         fold_pat_lit,

     (fn(&ENV e, &span sp,
         &ident i, &def_id did, &ann a) -> @pat)  fold_pat_bind,

     (fn(&ENV e, &span sp,
         &path p, &vec[@pat] args,
         &ann a) -> @pat)                         fold_pat_tag,


     // Stmt folds.
     (fn(&ENV e, &span sp,
         &@decl decl, &ann a)
      -> @stmt)                                   fold_stmt_decl,

     (fn(&ENV e, &span sp,
         &@expr e, &ann a)
      -> @stmt)                                   fold_stmt_expr,

     // Item folds.
     (fn(&ENV e, &span sp, &ident ident,
         &@ty t, &@expr e,
         &def_id id, &ann a) -> @item)            fold_item_const,

     (fn(&ENV e, &span sp, &ident ident,
         &ast::_fn f,
         &vec[ast::ty_param] ty_params,
         &def_id id, &ann a) -> @item)            fold_item_fn,

     (fn(&ENV e, &span sp, &ident ident,
         &option::t[str] link_name,
         &ast::fn_decl decl,
         &vec[ast::ty_param] ty_params,
         &def_id id, &ann a) -> @native_item)     fold_native_item_fn,

     (fn(&ENV e, &span sp, &ident ident,
         &ast::_mod m, &def_id id) -> @item)      fold_item_mod,

     (fn(&ENV e, &span sp, &ident ident,
         &ast::native_mod m, &def_id id) 
      -> @item)                                   fold_item_native_mod,

     (fn(&ENV e, &span sp, &ident ident,
         &@ty t, &vec[ast::ty_param] ty_params,
         &def_id id, &ann a) -> @item)            fold_item_ty,

     (fn(&ENV e, &span sp, &ident ident,
         &def_id id) -> @native_item)             fold_native_item_ty,

     (fn(&ENV e, &span sp, &ident ident,
         &vec[ast::variant] variants,
         &vec[ast::ty_param] ty_params,
         &def_id id, &ann a) -> @item)            fold_item_tag,

     (fn(&ENV e, &span sp, &ident ident,
         &ast::_obj ob,
         &vec[ast::ty_param] ty_params,
         &ast::obj_def_ids odid, &ann a) 
      -> @item)                                   fold_item_obj,

     // View Item folds.
     (fn(&ENV e, &span sp, &ident ident,
         &vec[@meta_item] meta_items,
         &def_id id,
         &option::t[int]) -> @view_item)          fold_view_item_use,

     (fn(&ENV e, &span sp, &ident i,
         &vec[ident] idents,
         &def_id id) -> @view_item)               fold_view_item_import,

     (fn(&ENV e, &span sp,
         &ident i) -> @view_item)                 fold_view_item_export,

     // Annotation folds.
     (fn(&ENV e, &ann a) -> ann)                  fold_ann,

     // Additional nodes.

     (fn(&ENV e, &fn_decl decl,
         ast::proto proto,
         &block body) -> ast::_fn)                fold_fn,

     (fn(&ENV e,
         &vec[arg] inputs,
         &@ty output,
         &purity p, &controlflow c) -> ast::fn_decl) fold_fn_decl,

     (fn(&ENV e, &ast::_mod m) -> ast::_mod)      fold_mod,

     (fn(&ENV e, &ast::native_mod m) 
      -> ast::native_mod)                         fold_native_mod,

     (fn(&ENV e, &span sp,
         &vec[@ast::crate_directive] cdirs,
         &ast::_mod m) -> @ast::crate)            fold_crate,

     (fn(&ENV e,
         &vec[ast::obj_field] fields,
         &vec[@ast::method] methods,
         &option::t[@ast::method] dtor)
      -> ast::_obj)                               fold_obj,

     (fn(&ENV e,
         &option::t[vec[ast::obj_field]] fields,
         &vec[@ast::method] methods,
         &option::t[@ast::expr] with_obj) 
      -> ast::anon_obj)                           fold_anon_obj,

     // Env updates.
     (fn(&ENV e, &@ast::crate c) -> ENV) update_env_for_crate,
     (fn(&ENV e, &@item i) -> ENV) update_env_for_item,
     (fn(&ENV e, &@native_item i) -> ENV) update_env_for_native_item,
     (fn(&ENV e, &@view_item i) -> ENV) update_env_for_view_item,
     (fn(&ENV e, &block b) -> ENV) update_env_for_block,
     (fn(&ENV e, &@stmt s) -> ENV) update_env_for_stmt,
     (fn(&ENV e, &@decl i) -> ENV) update_env_for_decl,
     (fn(&ENV e, &@pat p) -> ENV) update_env_for_pat,
     (fn(&ENV e, &arm a) -> ENV) update_env_for_arm,
     (fn(&ENV e, &@expr x) -> ENV) update_env_for_expr,
     (fn(&ENV e, &@ty t) -> ENV) update_env_for_ty,

     // Traversal control.
     (fn(&ENV v) -> bool) keep_going
     );


//// Fold drivers.

fn fold_path[ENV](&ENV env, &ast_fold[ENV] fld, &path p) -> path {
    let vec[@ast::ty] tys_ = [];
    for (@ast::ty t in p.node.types) {
        vec::push[@ast::ty](tys_, fold_ty(env, fld, t));
    }
    let ast::path_ p_ = rec(idents=p.node.idents, types=tys_);
    ret fld.fold_path(env, p.span, p_);
}

fn fold_ty[ENV](&ENV env, &ast_fold[ENV] fld, &@ty t) -> @ty {
    let ENV env_ = fld.update_env_for_ty(env, t);

    if (!fld.keep_going(env_)) {
        ret t;
    }

    alt (t.node) {
        case (ast::ty_nil) { ret fld.fold_ty_nil(env_, t.span); }
        case (ast::ty_bot) { ret fld.fold_ty_bot(env_, t.span); }
        case (ast::ty_bool) { ret fld.fold_ty_bool(env_, t.span); }
        case (ast::ty_int) { ret fld.fold_ty_int(env_, t.span); }
        case (ast::ty_uint) { ret fld.fold_ty_uint(env_, t.span); }
        case (ast::ty_float) { ret fld.fold_ty_float(env_, t.span); }

        case (ast::ty_machine(?m)) {
            ret fld.fold_ty_machine(env_, t.span, m);
        }

        case (ast::ty_char) { ret fld.fold_ty_char(env_, t.span); }
        case (ast::ty_str) { ret fld.fold_ty_str(env_, t.span); }

        case (ast::ty_box(?tm)) {
            auto ty_ = fold_ty(env, fld, tm.ty);
            ret fld.fold_ty_box(env_, t.span, rec(ty=ty_, mut=tm.mut));
        }

        case (ast::ty_vec(?tm)) {
            auto ty_ = fold_ty(env, fld, tm.ty);
            ret fld.fold_ty_vec(env_, t.span, rec(ty=ty_, mut=tm.mut));
        }

        case (ast::ty_tup(?elts)) {
            let vec[mt] elts_ = [];
            for (mt elt in elts) {
                auto ty_ = fold_ty(env, fld, elt.ty);
                vec::push[mt](elts_, rec(ty=ty_, mut=elt.mut));
            }
            ret fld.fold_ty_tup(env_, t.span, elts_);
        }

        case (ast::ty_rec(?flds)) {
            let vec[ast::ty_field] flds_ = [];
            for (ast::ty_field f in flds) {
                auto ty_ = fold_ty(env, fld, f.mt.ty);
                vec::push[ast::ty_field]
                    (flds_, rec(mt=rec(ty=ty_, mut=f.mt.mut) with f));
            }
            ret fld.fold_ty_rec(env_, t.span, flds_);
        }

        case (ast::ty_obj(?meths)) {
            let vec[ast::ty_method] meths_ = [];
            for (ast::ty_method m in meths) {
                auto tfn = fold_ty_fn(env_, fld, t.span, m.proto,
                                      m.inputs, m.output, m.cf);
                alt (tfn.node) {
                    case (ast::ty_fn(?p, ?ins, ?out, ?cf)) {
                        vec::push[ast::ty_method]
                            (meths_, rec(proto=p, inputs=ins,
                                         output=out, cf=cf with m));
                    }
                }
            }
            ret fld.fold_ty_obj(env_, t.span, meths_);
        }

        case (ast::ty_path(?pth, ?ann)) {
            auto pth_ = fold_path(env, fld, pth);
            ret fld.fold_ty_path(env_, t.span, pth_, ann);
        }

        case (ast::ty_fn(?proto, ?inputs, ?output, ?cf)) {
            ret fold_ty_fn(env_, fld, t.span, proto, inputs, output, cf);
        }

        case (ast::ty_chan(?ty)) {
            auto ty_ = fold_ty(env, fld, ty);
            ret fld.fold_ty_chan(env_, t.span, ty_);
        }

        case (ast::ty_port(?ty)) {
            auto ty_ = fold_ty(env, fld, ty);
            ret fld.fold_ty_port(env_, t.span, ty_);
        }
    }
}

fn fold_ty_fn[ENV](&ENV env, &ast_fold[ENV] fld, &span sp,
                   ast::proto proto,
                   &vec[rec(ast::mode mode, @ty ty)] inputs,
                   &@ty output, &controlflow cf) -> @ty {
    auto output_ = fold_ty(env, fld, output);
    let vec[rec(ast::mode mode, @ty ty)] inputs_ = [];
    for (rec(ast::mode mode, @ty ty) input in inputs) {
        auto ty_ = fold_ty(env, fld, input.ty);
        auto input_ = rec(ty=ty_ with input);
        inputs_ += [input_];
    }
    ret fld.fold_ty_fn(env, sp, proto, inputs_, output_, cf);
}

fn fold_decl[ENV](&ENV env, &ast_fold[ENV] fld, &@decl d) -> @decl {
    let ENV env_ = fld.update_env_for_decl(env, d);

    if (!fld.keep_going(env_)) {
        ret d;
    }

    alt (d.node) {
        case (ast::decl_local(?local)) {
            auto ty_ = none[@ast::ty];
            auto init_ = none[ast::initializer];
            alt (local.ty) {
                case (some[@ast::ty](?t)) {
                    ty_ = some[@ast::ty](fold_ty(env, fld, t));
                }
                case (_) { /* fall through */  }
            }
            alt (local.init) {
                case (some[ast::initializer](?init)) {
                    auto e =  fold_expr(env, fld, init.expr);
                    init_ = some[ast::initializer](rec(expr = e with init));
                }
                case (_) { /* fall through */  }
            }
            auto ann_ = fld.fold_ann(env_, local.ann);
            let @ast::local local_ =
                @rec(ty=ty_, init=init_, ann=ann_ with *local);
            ret fld.fold_decl_local(env_, d.span, local_);
        }

        case (ast::decl_item(?item)) {
            auto item_ = fold_item(env_, fld, item);
            ret fld.fold_decl_item(env_, d.span, item_);
        }
    }

    fail;
}

fn fold_pat[ENV](&ENV env, &ast_fold[ENV] fld, &@ast::pat p) -> @ast::pat {
    let ENV env_ = fld.update_env_for_pat(env, p);

    if (!fld.keep_going(env_)) {
        ret p;
    }

    alt (p.node) {
        case (ast::pat_wild(?t)) { ret fld.fold_pat_wild(env_, p.span, t); }
        case (ast::pat_lit(?lt, ?t)) {
            ret fld.fold_pat_lit(env_, p.span, lt, t);
        }
        case (ast::pat_bind(?id, ?did, ?t)) {
            ret fld.fold_pat_bind(env_, p.span, id, did, t);
        }
        case (ast::pat_tag(?path, ?pats, ?t)) {
            auto ppath = fold_path(env, fld, path);

            let vec[@ast::pat] ppats = [];
            for (@ast::pat pat in pats) {
                ppats += [fold_pat(env_, fld, pat)];
            }

            ret fld.fold_pat_tag(env_, p.span, ppath, ppats, t);
        }
    }
}

fn fold_exprs[ENV](&ENV env, &ast_fold[ENV] fld,
                   &vec[@expr] es) -> vec[@expr] {
    let vec[@expr] exprs = [];
    for (@expr e in es) {
        vec::push[@expr](exprs, fold_expr(env, fld, e));
    }
    ret exprs;
}

fn fold_tup_elt[ENV](&ENV env, &ast_fold[ENV] fld, &ast::elt e) -> ast::elt {
    ret rec(expr=fold_expr(env, fld, e.expr) with e);
}

fn fold_rec_field[ENV](&ENV env, &ast_fold[ENV] fld, &ast::field f)
    -> ast::field {
    ret rec(expr=fold_expr(env, fld, f.expr) with f);
}

fn fold_expr[ENV](&ENV env, &ast_fold[ENV] fld, &@expr e) -> @expr {

    let ENV env_ = fld.update_env_for_expr(env, e);

    if (!fld.keep_going(env_)) {
        ret e;
    }

    alt (e.node) {
        case (ast::expr_vec(?es, ?mut, ?t)) {
            auto ees = fold_exprs(env_, fld, es);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_vec(env_, e.span, ees, mut, t2);
        }

        case (ast::expr_tup(?es, ?t)) {
            let vec[ast::elt] elts = [];
            for (ast::elt e in es) {
                elts += [fold_tup_elt[ENV](env, fld, e)];
            }
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_tup(env_, e.span, elts, t2);
        }

        case (ast::expr_rec(?fs, ?base, ?t)) {
            let vec[ast::field] fields = [];
            let option::t[@expr] b = none[@expr];
            for (ast::field f in fs) {
                fields += [fold_rec_field(env, fld, f)];
            }
            alt (base) {
                case (none[@ast::expr]) { }
                case (some[@ast::expr](?eb)) {
                    b = some[@expr](fold_expr(env_, fld, eb));
                }
            }
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_rec(env_, e.span, fields, b, t2);
        }

        case (ast::expr_call(?f, ?args, ?t)) {
            auto ff = fold_expr(env_, fld, f);
            auto aargs = fold_exprs(env_, fld, args);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_call(env_, e.span, ff, aargs, t2);
        }

        case (ast::expr_self_method(?ident, ?t)) {
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_self_method(env_, e.span, ident, t2);
        }

        case (ast::expr_bind(?f, ?args_opt, ?t)) {
            auto ff = fold_expr(env_, fld, f);
            let vec[option::t[@ast::expr]] aargs_opt = [];
            for (option::t[@ast::expr] t_opt in args_opt) {
                alt (t_opt) {
                    case (none[@ast::expr]) {
                        aargs_opt += [none[@ast::expr]];
                    }
                    case (some[@ast::expr](?e)) {
                        aargs_opt += [some(fold_expr(env_, fld, e))];
                    }
                    case (none[@ast::expr]) { /* empty */ }
                }
            }
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_bind(env_, e.span, ff, aargs_opt, t2);
        }

        case (ast::expr_spawn(?dom, ?name, ?f, ?args, ?t)) {
            auto ff = fold_expr(env_, fld, f);
            auto aargs = fold_exprs(env_, fld, args);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_spawn(env_, e.span, dom, name, ff, aargs, t2);
        }

        case (ast::expr_binary(?op, ?a, ?b, ?t)) {
            auto aa = fold_expr(env_, fld, a);
            auto bb = fold_expr(env_, fld, b);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_binary(env_, e.span, op, aa, bb, t2);
        }

        case (ast::expr_unary(?op, ?a, ?t)) {
            auto aa = fold_expr(env_, fld, a);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_unary(env_, e.span, op, aa, t2);
        }

        case (ast::expr_lit(?lit, ?t)) {
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_lit(env_, e.span, lit, t2);
        }

        case (ast::expr_cast(?e, ?t, ?at)) {
            auto ee = fold_expr(env_, fld, e);
            auto tt = fold_ty(env, fld, t);
            auto at2 = fld.fold_ann(env_, at);
            ret fld.fold_expr_cast(env_, e.span, ee, tt, at2);
        }

        case (ast::expr_if(?cnd, ?thn, ?els, ?t)) {
            auto ccnd = fold_expr(env_, fld, cnd);
            auto tthn = fold_block(env_, fld, thn);
            auto eels = none[@expr];
            alt (els) {
                case (some[@expr](?e)) {
                    eels = some(fold_expr(env_, fld, e));
                }
                case (_) { /* fall through */  }
            }
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_if(env_, e.span, ccnd, tthn, eels, t2);
        }

        case (ast::expr_for(?decl, ?seq, ?body, ?t)) {
            auto ddecl = fold_decl(env_, fld, decl);
            auto sseq = fold_expr(env_, fld, seq);
            auto bbody = fold_block(env_, fld, body);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_for(env_, e.span, ddecl, sseq, bbody, t2);
        }

        case (ast::expr_for_each(?decl, ?seq, ?body, ?t)) {
            auto ddecl = fold_decl(env_, fld, decl);
            auto sseq = fold_expr(env_, fld, seq);
            auto bbody = fold_block(env_, fld, body);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_for_each(env_, e.span, ddecl, sseq, bbody, t2);
        }

        case (ast::expr_while(?cnd, ?body, ?t)) {
            auto ccnd = fold_expr(env_, fld, cnd);
            auto bbody = fold_block(env_, fld, body);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_while(env_, e.span, ccnd, bbody, t2);
        }

        case (ast::expr_do_while(?body, ?cnd, ?t)) {
            auto bbody = fold_block(env_, fld, body);
            auto ccnd = fold_expr(env_, fld, cnd);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_do_while(env_, e.span, bbody, ccnd, t2);
        }

        case (ast::expr_alt(?expr, ?arms, ?t)) {
            auto eexpr = fold_expr(env_, fld, expr);
            let vec[ast::arm] aarms = [];
            for (ast::arm a in arms) {
                aarms += [fold_arm(env_, fld, a)];
            }
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_alt(env_, e.span, eexpr, aarms, t2);
        }

        case (ast::expr_block(?b, ?t)) {
            auto bb = fold_block(env_, fld, b);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_block(env_, e.span, bb, t2);
        }

        case (ast::expr_assign(?lhs, ?rhs, ?t)) {
            auto llhs = fold_expr(env_, fld, lhs);
            auto rrhs = fold_expr(env_, fld, rhs);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_assign(env_, e.span, llhs, rrhs, t2);
        }

        case (ast::expr_assign_op(?op, ?lhs, ?rhs, ?t)) {
            auto llhs = fold_expr(env_, fld, lhs);
            auto rrhs = fold_expr(env_, fld, rhs);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_assign_op(env_, e.span, op, llhs, rrhs, t2);
        }

        case (ast::expr_send(?lhs, ?rhs, ?t)) {
            auto llhs = fold_expr(env_, fld, lhs);
            auto rrhs = fold_expr(env_, fld, rhs);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_send(env_, e.span, llhs, rrhs, t2);
        }

        case (ast::expr_recv(?lhs, ?rhs, ?t)) {
            auto llhs = fold_expr(env_, fld, lhs);
            auto rrhs = fold_expr(env_, fld, rhs);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_recv(env_, e.span, llhs, rrhs, t2);
        }

        case (ast::expr_field(?e, ?i, ?t)) {
            auto ee = fold_expr(env_, fld, e);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_field(env_, e.span, ee, i, t2);
        }

        case (ast::expr_index(?e, ?ix, ?t)) {
            auto ee = fold_expr(env_, fld, e);
            auto iix = fold_expr(env_, fld, ix);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_index(env_, e.span, ee, iix, t2);
        }

        case (ast::expr_path(?p, ?t)) {
            auto p_ = fold_path(env_, fld, p);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_path(env_, e.span, p_, t2);
        }

        case (ast::expr_ext(?p, ?args, ?body, ?expanded, ?t)) {
            // Only fold the expanded expression, not the
            // expressions involved in syntax extension
            auto exp = fold_expr(env_, fld, expanded);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_ext(env_, e.span, p, args, body,
                                  exp, t2);
        }

        case (ast::expr_fail(?t)) {
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_fail(env_, e.span, t2);
        }

        case (ast::expr_break(?t)) {
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_break(env_, e.span, t2);
        }

        case (ast::expr_cont(?t)) {
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_cont(env_, e.span, t2);
        }

        case (ast::expr_ret(?oe, ?t)) {
            auto oee = none[@expr];
            alt (oe) {
                case (some[@expr](?x)) {
                    oee = some(fold_expr(env_, fld, x));
                }
                case (_) { /* fall through */  }
            }
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_ret(env_, e.span, oee, t2);
        }

        case (ast::expr_put(?oe, ?t)) {
            auto oee = none[@expr];
            alt (oe) {
                case (some[@expr](?x)) {
                    oee = some(fold_expr(env_, fld, x));
                }
                case (_) { /* fall through */  }
            }
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_put(env_, e.span, oee, t2);
        }

        case (ast::expr_be(?x, ?t)) {
            auto ee = fold_expr(env_, fld, x);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_be(env_, e.span, ee, t2);
        }

        case (ast::expr_log(?l, ?x, ?t)) {
            auto ee = fold_expr(env_, fld, x);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_log(env_, e.span, l, ee, t2);
        }

        case (ast::expr_check(?x, ?t)) {
            auto ee = fold_expr(env_, fld, x);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_check(env_, e.span, ee, t2);
        }

        case (ast::expr_assert(?x, ?t)) {
            auto ee = fold_expr(env_, fld, x);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_assert(env_, e.span, ee, t2);
        }

        case (ast::expr_port(?t)) {
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_port(env_, e.span, t2);
        }

        case (ast::expr_chan(?x, ?t)) {
            auto ee = fold_expr(env_, fld, x);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_chan(env_, e.span, ee, t2);
        }

        case (ast::expr_anon_obj(?ob, ?tps, ?odid, ?t)) {
            auto ee = fold_anon_obj(env_, fld, ob);
            auto t2 = fld.fold_ann(env_, t);
            ret fld.fold_expr_anon_obj(env_, e.span, ee, tps, odid, t2);
        }
    }

    fail;
}


fn fold_stmt[ENV](&ENV env, &ast_fold[ENV] fld, &@stmt s) -> @stmt {

    let ENV env_ = fld.update_env_for_stmt(env, s);

    if (!fld.keep_going(env_)) {
        ret s;
    }

    alt (s.node) {
        case (ast::stmt_decl(?d, ?a)) {
            auto dd = fold_decl(env_, fld, d);
            auto aa = fld.fold_ann(env_, a);
            ret fld.fold_stmt_decl(env_, s.span, dd, aa);
        }

        case (ast::stmt_expr(?e, ?a)) {
            auto ee = fold_expr(env_, fld, e);
            auto aa = fld.fold_ann(env_, a);
            ret fld.fold_stmt_expr(env_, s.span, ee, aa);
        }
    }
    fail;
}

fn fold_block[ENV](&ENV env, &ast_fold[ENV] fld, &block blk) -> block {

    let ENV env_ = fld.update_env_for_block(env, blk);

    if (!fld.keep_going(env_)) {
        ret blk;
    }

    let vec[@ast::stmt] stmts = [];
    for (@ast::stmt s in blk.node.stmts) {
        auto new_stmt = fold_stmt[ENV](env_, fld, s);
        vec::push[@ast::stmt](stmts, new_stmt);
    }

    auto expr = none[@ast::expr];
    alt (blk.node.expr) {
        case (some[@ast::expr](?e)) {
            expr = some[@ast::expr](fold_expr[ENV](env_, fld, e));
        }
        case (none[@ast::expr]) {
            // empty
        }
    }

    auto aa = fld.fold_ann(env, blk.node.a);
    ret respan(blk.span, rec(stmts=stmts, expr=expr, a=aa));
}

fn fold_arm[ENV](&ENV env, &ast_fold[ENV] fld, &arm a) -> arm {
    let ENV env_ = fld.update_env_for_arm(env, a);
    auto ppat = fold_pat(env_, fld, a.pat);
    auto bblock = fold_block(env_, fld, a.block);
    ret rec(pat=ppat, block=bblock);
}

fn fold_arg[ENV](&ENV env, &ast_fold[ENV] fld, &arg a) -> arg {
    auto ty = fold_ty(env, fld, a.ty);
    ret rec(ty=ty with a);
}

fn fold_fn_decl[ENV](&ENV env, &ast_fold[ENV] fld,
                     &ast::fn_decl decl) -> ast::fn_decl {
    let vec[ast::arg] inputs = [];
    for (ast::arg a in decl.inputs) {
        inputs += [fold_arg(env, fld, a)];
    }
    auto output = fold_ty[ENV](env, fld, decl.output);
    ret fld.fold_fn_decl(env, inputs, output, decl.purity, decl.cf);
}

fn fold_fn[ENV](&ENV env, &ast_fold[ENV] fld, &ast::_fn f) -> ast::_fn {
    auto decl = fold_fn_decl(env, fld, f.decl);

    auto body = fold_block[ENV](env, fld, f.body);

    ret fld.fold_fn(env, decl, f.proto, body);
}


fn fold_obj_field[ENV](&ENV env, &ast_fold[ENV] fld,
                       &ast::obj_field f) -> ast::obj_field {
    auto ty = fold_ty(env, fld, f.ty);
    ret rec(ty=ty with f);
}


fn fold_method[ENV](&ENV env, &ast_fold[ENV] fld,
                    &@ast::method m) -> @ast::method {
    auto meth = fold_fn(env, fld, m.node.meth);
    ret @rec(node=rec(meth=meth with m.node) with *m);
}

fn fold_obj[ENV](&ENV env, &ast_fold[ENV] fld, &ast::_obj ob) -> ast::_obj {

    let vec[ast::obj_field] fields = [];
    let vec[@ast::method] meths = [];
    for (ast::obj_field f in ob.fields) {
        fields += [fold_obj_field(env, fld, f)];
    }
    let option::t[@ast::method] dtor = none[@ast::method];
    alt (ob.dtor) {
        case (none[@ast::method]) { }
        case (some[@ast::method](?m)) {
            dtor = some[@ast::method](fold_method[ENV](env, fld, m));
        }
    }
    let vec[ast::ty_param] tp = [];
    for (@ast::method m in ob.methods) {
        // Fake-up an ast::item for this method.
        // FIXME: this is kinda awful. Maybe we should reformulate
        // the way we store methods in the AST?
        let @ast::item i = @rec(node=ast::item_fn(m.node.ident,
                                                m.node.meth,
                                                tp,
                                                m.node.id,
                                                m.node.ann),
                               span=m.span);
        let ENV _env = fld.update_env_for_item(env, i);
        vec::push[@ast::method](meths, fold_method(_env, fld, m));
    }
    ret fld.fold_obj(env, fields, meths, dtor);
}

fn fold_anon_obj[ENV](&ENV env, &ast_fold[ENV] fld, &ast::anon_obj ob) 
    -> ast::anon_obj {

    // Fields
    let option::t[vec[ast::obj_field]] fields = none[vec[ast::obj_field]];
    alt (ob.fields) {
        case (none[vec[ast::obj_field]]) { }
        case (some[vec[ast::obj_field]](?v)) {
            let vec[ast::obj_field] fields = [];
            for (ast::obj_field f in v) {
                fields += [fold_obj_field(env, fld, f)];
            }
        }
    }

    // with_obj
    let option::t[@ast::expr] with_obj = none[@ast::expr];
    alt (ob.with_obj) {
        case (none[@ast::expr]) { }
        case (some[@ast::expr](?e)) {
            with_obj = some[@ast::expr](fold_expr(env, fld, e));
        }
    }

    // Methods
    let vec[@ast::method] meths = [];
    let vec[ast::ty_param] tp = [];
    for (@ast::method m in ob.methods) {
        // Fake-up an ast::item for this method.
        // FIXME: this is kinda awful. Maybe we should reformulate
        // the way we store methods in the AST?
        let @ast::item i = @rec(node=ast::item_fn(m.node.ident,
                                                m.node.meth,
                                                tp,
                                                m.node.id,
                                                m.node.ann),
                               span=m.span);
        let ENV _env = fld.update_env_for_item(env, i);
        vec::push[@ast::method](meths, fold_method(_env, fld, m));
    }
    ret fld.fold_anon_obj(env, fields, meths, with_obj);
}

fn fold_view_item[ENV](&ENV env, &ast_fold[ENV] fld, &@view_item vi)
    -> @view_item {

    let ENV env_ = fld.update_env_for_view_item(env, vi);

    if (!fld.keep_going(env_)) {
        ret vi;
    }

    alt (vi.node) {
        case (ast::view_item_use(?ident, ?meta_items, ?def_id, ?cnum)) {
            ret fld.fold_view_item_use(env_, vi.span, ident, meta_items,
                                       def_id, cnum);
        }
        case (ast::view_item_import(?def_ident, ?idents, ?def_id)) {
            ret fld.fold_view_item_import(env_, vi.span, def_ident, idents,
                                          def_id);
        }

        case (ast::view_item_export(?def_ident)) {
            ret fld.fold_view_item_export(env_, vi.span, def_ident);
        }
    }

    fail;
}

fn fold_item[ENV](&ENV env, &ast_fold[ENV] fld, &@item i) -> @item {

    let ENV env_ = fld.update_env_for_item(env, i);

    if (!fld.keep_going(env_)) {
        ret i;
    }

    alt (i.node) {

        case (ast::item_const(?ident, ?t, ?e, ?id, ?ann)) {
            let @ast::ty t_ = fold_ty[ENV](env_, fld, t);
            let @ast::expr e_ = fold_expr(env_, fld, e);
            ret fld.fold_item_const(env_, i.span, ident, t_, e_, id, ann);
        }

        case (ast::item_fn(?ident, ?ff, ?tps, ?id, ?ann)) {
            let ast::_fn ff_ = fold_fn[ENV](env_, fld, ff);
            ret fld.fold_item_fn(env_, i.span, ident, ff_, tps, id, ann);
        }

        case (ast::item_mod(?ident, ?mm, ?id)) {
            let ast::_mod mm_ = fold_mod[ENV](env_, fld, mm);
            ret fld.fold_item_mod(env_, i.span, ident, mm_, id);
        }

        case (ast::item_native_mod(?ident, ?mm, ?id)) {
            let ast::native_mod mm_ = fold_native_mod[ENV](env_, fld, mm);
            ret fld.fold_item_native_mod(env_, i.span, ident, mm_, id);
        }

        case (ast::item_ty(?ident, ?ty, ?params, ?id, ?ann)) {
            let @ast::ty ty_ = fold_ty[ENV](env_, fld, ty);
            ret fld.fold_item_ty(env_, i.span, ident, ty_, params, id, ann);
        }

        case (ast::item_tag(?ident, ?variants, ?ty_params, ?id, ?ann)) {
            let vec[ast::variant] new_variants = [];
            for (ast::variant v in variants) {
                let vec[ast::variant_arg] new_args = [];
                for (ast::variant_arg va in v.node.args) {
                    auto new_ty = fold_ty[ENV](env_, fld, va.ty);
                    new_args += [rec(ty=new_ty, id=va.id)];
                }
                auto new_v = rec(name=v.node.name, args=new_args,
                                 id=v.node.id, ann=v.node.ann);
                new_variants += [respan[ast::variant_](v.span, new_v)];
            }
            ret fld.fold_item_tag(env_, i.span, ident, new_variants,
                                  ty_params, id, ann);
        }

        case (ast::item_obj(?ident, ?ob, ?tps, ?odid, ?ann)) {
            let ast::_obj ob_ = fold_obj[ENV](env_, fld, ob);
            ret fld.fold_item_obj(env_, i.span, ident, ob_, tps, odid, ann);
        }

    }

    fail;
}

fn fold_mod[ENV](&ENV e, &ast_fold[ENV] fld, &ast::_mod m) -> ast::_mod {

    let vec[@view_item] view_items = [];
    let vec[@item] items = [];

    for (@view_item vi in m.view_items) {
        auto new_vi = fold_view_item[ENV](e, fld, vi);
        vec::push[@view_item](view_items, new_vi);
    }

    for (@item i in m.items) {
        auto new_item = fold_item[ENV](e, fld, i);
        vec::push[@item](items, new_item);
    }

    ret fld.fold_mod(e, rec(view_items=view_items, items=items));
}

fn fold_native_item[ENV](&ENV env, &ast_fold[ENV] fld,
                         &@native_item i) -> @native_item {
    let ENV env_ = fld.update_env_for_native_item(env, i);

    if (!fld.keep_going(env_)) {
        ret i;
    }
    alt (i.node) {
        case (ast::native_item_ty(?ident, ?id)) {
            ret fld.fold_native_item_ty(env_, i.span, ident, id);
        }
        case (ast::native_item_fn(?ident, ?lname, ?fn_decl,
                                 ?ty_params, ?id, ?ann)) {
            auto d = fold_fn_decl[ENV](env_, fld, fn_decl);
            ret fld.fold_native_item_fn(env_, i.span, ident, lname, d,
                                        ty_params, id, ann);
        }
    }
}

fn fold_native_mod[ENV](&ENV e, &ast_fold[ENV] fld,
                        &ast::native_mod m) -> ast::native_mod {
    let vec[@view_item] view_items = [];
    let vec[@native_item] items = [];

    for (@view_item vi in m.view_items) {
        auto new_vi = fold_view_item[ENV](e, fld, vi);
        vec::push[@view_item](view_items, new_vi);
    }

    for (@native_item i in m.items) {
        auto new_item = fold_native_item[ENV](e, fld, i);
        vec::push[@native_item](items, new_item);
    }

    ret fld.fold_native_mod(e, rec(native_name=m.native_name,
                                   abi=m.abi,
                                   view_items=view_items,
                                   items=items));
}

fn fold_crate[ENV](&ENV env, &ast_fold[ENV] fld,
                   &@ast::crate c) -> @ast::crate {
    // FIXME: possibly fold the directives so you process any expressions
    // within them? Not clear. After front/eval::rs, nothing else should look
    // at crate directives.
    let ENV env_ = fld.update_env_for_crate(env, c);
    let ast::_mod m = fold_mod[ENV](env_, fld, c.node.module);
    ret fld.fold_crate(env_, c.span, c.node.directives, m);
}

//// Identity folds.

fn respan[T](&span sp, &T t) -> spanned[T] {
    ret rec(node=t, span=sp);
}


// Path identity.

fn identity_fold_path[ENV](&ENV env, &span sp, &ast::path_ p) -> path {
    ret respan(sp, p);
}

// Type identities.

fn identity_fold_ty_nil[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_nil);
}

fn identity_fold_ty_bot[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_bot);
}

fn identity_fold_ty_bool[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_bool);
}

fn identity_fold_ty_int[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_int);
}

fn identity_fold_ty_uint[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_uint);
}

fn identity_fold_ty_float[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_float);
}

fn identity_fold_ty_machine[ENV](&ENV env, &span sp,
                                 ty_mach tm) -> @ty {
    ret @respan(sp, ast::ty_machine(tm));
}

fn identity_fold_ty_char[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_char);
}

fn identity_fold_ty_str[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast::ty_str);
}

fn identity_fold_ty_box[ENV](&ENV env, &span sp, &mt tm) -> @ty {
    ret @respan(sp, ast::ty_box(tm));
}

fn identity_fold_ty_vec[ENV](&ENV env, &span sp, &mt tm) -> @ty {
    ret @respan(sp, ast::ty_vec(tm));
}

fn identity_fold_ty_tup[ENV](&ENV env, &span sp,
                             &vec[mt] elts) -> @ty {
    ret @respan(sp, ast::ty_tup(elts));
}

fn identity_fold_ty_rec[ENV](&ENV env, &span sp,
                             &vec[ast::ty_field] elts) -> @ty {
    ret @respan(sp, ast::ty_rec(elts));
}

fn identity_fold_ty_obj[ENV](&ENV env, &span sp,
                             &vec[ast::ty_method] meths) -> @ty {
    ret @respan(sp, ast::ty_obj(meths));
}

fn identity_fold_ty_fn[ENV](&ENV env, &span sp,
                            ast::proto proto,
                            &vec[rec(ast::mode mode, @ty ty)] inputs,
                            &@ty output, &controlflow cf) -> @ty {
    ret @respan(sp, ast::ty_fn(proto, inputs, output, cf));
}

fn identity_fold_ty_path[ENV](&ENV env, &span sp, &ast::path p,
                              &ann a) -> @ty {
    ret @respan(sp, ast::ty_path(p, a));
}

fn identity_fold_ty_chan[ENV](&ENV env, &span sp, &@ty t) -> @ty {
    ret @respan(sp, ast::ty_chan(t));
}

fn identity_fold_ty_port[ENV](&ENV env, &span sp, &@ty t) -> @ty {
    ret @respan(sp, ast::ty_port(t));
}

// Expr identities.

fn identity_fold_expr_vec[ENV](&ENV env, &span sp, &vec[@expr] es,
                               ast::mutability mut, &ann a) -> @expr {
    ret @respan(sp, ast::expr_vec(es, mut, a));
}

fn identity_fold_expr_tup[ENV](&ENV env, &span sp,
                               &vec[ast::elt] es, &ann a) -> @expr {
    ret @respan(sp, ast::expr_tup(es, a));
}

fn identity_fold_expr_rec[ENV](&ENV env, &span sp,
                               &vec[ast::field] fields,
                               &option::t[@expr] base, &ann a) -> @expr {
    ret @respan(sp, ast::expr_rec(fields, base, a));
}

fn identity_fold_expr_call[ENV](&ENV env, &span sp, &@expr f,
                                &vec[@expr] args, &ann a) -> @expr {
    ret @respan(sp, ast::expr_call(f, args, a));
}

fn identity_fold_expr_self_method[ENV](&ENV env, &span sp, &ident id,
                                       &ann a) -> @expr {
    ret @respan(sp, ast::expr_self_method(id, a));
}

fn identity_fold_expr_bind[ENV](&ENV env, &span sp, &@expr f,
                                &vec[option::t[@expr]] args_opt, &ann a)
        -> @expr {
    ret @respan(sp, ast::expr_bind(f, args_opt, a));
}

fn identity_fold_expr_spawn[ENV](&ENV env, &span sp,
                                 ast::spawn_dom dom, &option::t[str] name,
                                 &@expr f, &vec[@expr] args,
                                 &ann a) -> @expr {
    ret @respan(sp, ast::expr_spawn(dom, name, f, args, a));
}

fn identity_fold_expr_binary[ENV](&ENV env, &span sp, ast::binop b,
                                  &@expr lhs, &@expr rhs,
                                  &ann a) -> @expr {
    ret @respan(sp, ast::expr_binary(b, lhs, rhs, a));
}

fn identity_fold_expr_unary[ENV](&ENV env, &span sp,
                                 ast::unop u, &@expr e, &ann a)
        -> @expr {
    ret @respan(sp, ast::expr_unary(u, e, a));
}

fn identity_fold_expr_lit[ENV](&ENV env, &span sp, &@ast::lit lit,
                               &ann a) -> @expr {
    ret @respan(sp, ast::expr_lit(lit, a));
}

fn identity_fold_expr_cast[ENV](&ENV env, &span sp, &@ast::expr e,
                                &@ast::ty t, &ann a) -> @expr {
    ret @respan(sp, ast::expr_cast(e, t, a));
}

fn identity_fold_expr_if[ENV](&ENV env, &span sp,
                              &@expr cond, &block thn,
                              &option::t[@expr] els, &ann a) -> @expr {
    ret @respan(sp, ast::expr_if(cond, thn, els, a));
}

fn identity_fold_expr_for[ENV](&ENV env, &span sp,
                               &@decl d, &@expr seq,
                               &block body, &ann a) -> @expr {
    ret @respan(sp, ast::expr_for(d, seq, body, a));
}

fn identity_fold_expr_for_each[ENV](&ENV env, &span sp,
                                    &@decl d, &@expr seq,
                                    &block body, &ann a) -> @expr {
    ret @respan(sp, ast::expr_for_each(d, seq, body, a));
}

fn identity_fold_expr_while[ENV](&ENV env, &span sp,
                                 &@expr cond, &block body, &ann a) -> @expr {
    ret @respan(sp, ast::expr_while(cond, body, a));
}

fn identity_fold_expr_do_while[ENV](&ENV env, &span sp,
                                    &block body, &@expr cond,
                                    &ann a) -> @expr {
    ret @respan(sp, ast::expr_do_while(body, cond, a));
}

fn identity_fold_expr_alt[ENV](&ENV env, &span sp,
                               &@expr e, &vec[arm] arms,
                               &ann a) -> @expr {
    ret @respan(sp, ast::expr_alt(e, arms, a));
}

fn identity_fold_expr_block[ENV](&ENV env, &span sp, &block blk,
                                 &ann a) -> @expr {
    ret @respan(sp, ast::expr_block(blk, a));
}

fn identity_fold_expr_assign[ENV](&ENV env, &span sp,
                                  &@expr lhs, &@expr rhs, &ann a)
        -> @expr {
    ret @respan(sp, ast::expr_assign(lhs, rhs, a));
}

fn identity_fold_expr_assign_op[ENV](&ENV env, &span sp, ast::binop op,
                                     &@expr lhs, &@expr rhs, &ann a)
        -> @expr {
    ret @respan(sp, ast::expr_assign_op(op, lhs, rhs, a));
}

fn identity_fold_expr_send[ENV](&ENV e, &span sp,
                                &@expr lhs, &@expr rhs, &ann a) -> @expr {
    ret @respan(sp, ast::expr_send(lhs, rhs, a));
}

fn identity_fold_expr_recv[ENV](&ENV e, &span sp,
                                &@expr lhs, &@expr rhs, &ann a) -> @expr {
    ret @respan(sp, ast::expr_recv(lhs, rhs, a));
}

fn identity_fold_expr_field[ENV](&ENV env, &span sp,
                                 &@expr e, &ident i, &ann a) -> @expr {
    ret @respan(sp, ast::expr_field(e, i, a));
}

fn identity_fold_expr_index[ENV](&ENV env, &span sp,
                                 &@expr e, &@expr ix, &ann a) -> @expr {
    ret @respan(sp, ast::expr_index(e, ix, a));
}

fn identity_fold_expr_path[ENV](&ENV env, &span sp,
                                &path p, &ann a) -> @expr {
    ret @respan(sp, ast::expr_path(p, a));
}

fn identity_fold_expr_ext[ENV](&ENV env, &span sp,
                               &path p, &vec[@expr] args,
                               &option::t[str] body,
                               &@expr expanded,
                               &ann a) -> @expr {
    ret @respan(sp, ast::expr_ext(p, args, body, expanded, a));
}

fn identity_fold_expr_fail[ENV](&ENV env, &span sp, &ann a) -> @expr {
    ret @respan(sp, ast::expr_fail(a));
}

fn identity_fold_expr_break[ENV](&ENV env, &span sp, &ann a) -> @expr {
    ret @respan(sp, ast::expr_break(a));
}

fn identity_fold_expr_cont[ENV](&ENV env, &span sp, &ann a) -> @expr {
    ret @respan(sp, ast::expr_cont(a));
}

fn identity_fold_expr_ret[ENV](&ENV env, &span sp,
                               &option::t[@expr] rv, &ann a) -> @expr {
    ret @respan(sp, ast::expr_ret(rv, a));
}

fn identity_fold_expr_put[ENV](&ENV env, &span sp,
                               &option::t[@expr] rv, &ann a) -> @expr {
    ret @respan(sp, ast::expr_put(rv, a));
}

fn identity_fold_expr_be[ENV](&ENV env, &span sp,
                              &@expr x, &ann a) -> @expr {
    ret @respan(sp, ast::expr_be(x, a));
}

fn identity_fold_expr_log[ENV](&ENV e, &span sp, int lvl, &@expr x,
                               &ann a) -> @expr {
    ret @respan(sp, ast::expr_log(lvl, x, a));
}

fn identity_fold_expr_check[ENV](&ENV e, &span sp, &@expr x, &ann a)
    -> @expr {
    ret @respan(sp, ast::expr_check(x, a));
}

fn identity_fold_expr_assert[ENV](&ENV e, &span sp, &@expr x, &ann a)
    -> @expr {
    ret @respan(sp, ast::expr_assert(x, a));
}

fn identity_fold_expr_port[ENV](&ENV e, &span sp, &ann a) -> @expr {
    ret @respan(sp, ast::expr_port(a));
}

fn identity_fold_expr_chan[ENV](&ENV e, &span sp, &@expr x,
                                &ann a) -> @expr {
    ret @respan(sp, ast::expr_chan(x, a));
}

fn identity_fold_expr_anon_obj[ENV](&ENV e, &span sp,
                                    &ast::anon_obj ob, 
                                    &vec[ast::ty_param] tps,
                                    &ast::obj_def_ids odid, 
                                    &ann a) -> @expr {
    ret @respan(sp, ast::expr_anon_obj(ob, tps, odid, a));
}

// Decl identities.

fn identity_fold_decl_local[ENV](&ENV e, &span sp,
                                 &@ast::local local) -> @decl {
    ret @respan(sp, ast::decl_local(local));
}

fn identity_fold_decl_item[ENV](&ENV e, &span sp, &@item i) -> @decl {
    ret @respan(sp, ast::decl_item(i));
}


// Pat identities.

fn identity_fold_pat_wild[ENV](&ENV e, &span sp, &ann a) -> @pat {
    ret @respan(sp, ast::pat_wild(a));
}

fn identity_fold_pat_lit[ENV](&ENV e, &span sp,
                              &@ast::lit lit, &ann a) -> @pat {
    ret @respan(sp, ast::pat_lit(lit, a));
}

fn identity_fold_pat_bind[ENV](&ENV e, &span sp, &ident i,
                               &def_id did, &ann a)
        -> @pat {
    ret @respan(sp, ast::pat_bind(i, did, a));
}

fn identity_fold_pat_tag[ENV](&ENV e, &span sp, &path p, &vec[@pat] args,
                              &ann a) -> @pat {
    ret @respan(sp, ast::pat_tag(p, args, a));
}


// Stmt identities.

fn identity_fold_stmt_decl[ENV](&ENV env, &span sp,
                                &@decl d, &ann a) -> @stmt {
    ret @respan(sp, ast::stmt_decl(d, a));
}

fn identity_fold_stmt_expr[ENV](&ENV e, &span sp,
                                &@expr x, &ann a) -> @stmt {
    ret @respan(sp, ast::stmt_expr(x, a));
}


// Item identities.

fn identity_fold_item_const[ENV](&ENV e, &span sp, &ident i,
                                  &@ty t, &@expr ex,
                                 &def_id id, &ann a) -> @item {
    ret @respan(sp, ast::item_const(i, t, ex, id, a));
}

fn identity_fold_item_fn[ENV](&ENV e, &span sp, &ident i,
                              &ast::_fn f, &vec[ast::ty_param] ty_params,
                              &def_id id, &ann a) -> @item {
    ret @respan(sp, ast::item_fn(i, f, ty_params, id, a));
}

fn identity_fold_native_item_fn[ENV](&ENV e, &span sp, &ident i,
                                     &option::t[str] link_name,
                                     &ast::fn_decl decl,
                                     &vec[ast::ty_param] ty_params,
                                     &def_id id, &ann a) -> @native_item {
    ret @respan(sp, ast::native_item_fn(i, link_name, decl, ty_params,
                                        id, a));
}

fn identity_fold_item_mod[ENV](&ENV e, &span sp, &ident i,
                               &ast::_mod m, &def_id id) -> @item {
    ret @respan(sp, ast::item_mod(i, m, id));
}

fn identity_fold_item_native_mod[ENV](&ENV e, &span sp, &ident i,
                                      &ast::native_mod m,
                                      &def_id id) -> @item {
    ret @respan(sp, ast::item_native_mod(i, m, id));
}

fn identity_fold_item_ty[ENV](&ENV e, &span sp, &ident i,
                              &@ty t, &vec[ast::ty_param] ty_params,
                              &def_id id, &ann a) -> @item {
    ret @respan(sp, ast::item_ty(i, t, ty_params, id, a));
}

fn identity_fold_native_item_ty[ENV](&ENV e, &span sp, &ident i,
                                     &def_id id) -> @native_item {
    ret @respan(sp, ast::native_item_ty(i, id));
}

fn identity_fold_item_tag[ENV](&ENV e, &span sp, &ident i,
                               &vec[ast::variant] variants,
                               &vec[ast::ty_param] ty_params,
                               &def_id id, &ann a) -> @item {
    ret @respan(sp, ast::item_tag(i, variants, ty_params, id, a));
}

fn identity_fold_item_obj[ENV](&ENV e, &span sp, &ident i,
                               &ast::_obj ob, &vec[ast::ty_param] ty_params,
                               &ast::obj_def_ids odid, &ann a) -> @item {
    ret @respan(sp, ast::item_obj(i, ob, ty_params, odid, a));
}

// View Item folds.

fn identity_fold_view_item_use[ENV](&ENV e, &span sp, &ident i,
                                    &vec[@meta_item] meta_items,
                                    &def_id id, &option::t[int] cnum)
    -> @view_item {
    ret @respan(sp, ast::view_item_use(i, meta_items, id, cnum));
}

fn identity_fold_view_item_import[ENV](&ENV e, &span sp, &ident i,
                                       &vec[ident] is, &def_id id)
    -> @view_item {
    ret @respan(sp, ast::view_item_import(i, is, id));
}

fn identity_fold_view_item_export[ENV](&ENV e, &span sp, &ident i)
    -> @view_item {
    ret @respan(sp, ast::view_item_export(i));
}

// Annotation folding.

fn identity_fold_ann[ENV](&ENV e, &ann a) -> ann {
    ret a;
}

// Additional identities.

fn identity_fold_block[ENV](&ENV e, &span sp, &ast::block_ blk) -> block {
    ret respan(sp, blk);
}

fn identity_fold_fn_decl[ENV](&ENV e,
                              &vec[arg] inputs,
                              &@ty output,
                              &purity p, &controlflow c) -> ast::fn_decl {
    ret rec(inputs=inputs, output=output, purity=p, cf=c);
}

fn identity_fold_fn[ENV](&ENV e,
                         &fn_decl decl,
                         ast::proto proto,
                         &block body) -> ast::_fn {
    ret rec(decl=decl, proto=proto, body=body);
}

fn identity_fold_mod[ENV](&ENV e, &ast::_mod m) -> ast::_mod {
    ret m;
}

fn identity_fold_native_mod[ENV](&ENV e,
                                 &ast::native_mod m) -> ast::native_mod {
    ret m;
}

fn identity_fold_crate[ENV](&ENV e, &span sp,
                            &vec[@ast::crate_directive] cdirs,
                            &ast::_mod m) -> @ast::crate {
    ret @respan(sp, rec(directives=cdirs, module=m));
}

fn identity_fold_obj[ENV](&ENV e,
                          &vec[ast::obj_field] fields,
                          &vec[@ast::method] methods,
                          &option::t[@ast::method] dtor) -> ast::_obj {
    ret rec(fields=fields, methods=methods, dtor=dtor);
}

fn identity_fold_anon_obj[ENV](&ENV e,
                               &option::t[vec[ast::obj_field]] fields,
                               &vec[@ast::method] methods,
                               &option::t[@ast::expr] with_obj) 
    -> ast::anon_obj {
    ret rec(fields=fields, methods=methods, with_obj=with_obj);
}

// Env update identities.

fn identity_update_env_for_crate[ENV](&ENV e, &@ast::crate c) -> ENV {
    ret e;
}

fn identity_update_env_for_item[ENV](&ENV e, &@item i) -> ENV {
    ret e;
}

fn identity_update_env_for_native_item[ENV](&ENV e, &@native_item i) -> ENV {
    ret e;
}

fn identity_update_env_for_view_item[ENV](&ENV e, &@view_item i) -> ENV {
    ret e;
}

fn identity_update_env_for_block[ENV](&ENV e, &block b) -> ENV {
    ret e;
}

fn identity_update_env_for_stmt[ENV](&ENV e, &@stmt s) -> ENV {
    ret e;
}

fn identity_update_env_for_decl[ENV](&ENV e, &@decl d) -> ENV {
    ret e;
}

fn identity_update_env_for_arm[ENV](&ENV e, &arm a) -> ENV {
    ret e;
}

fn identity_update_env_for_pat[ENV](&ENV e, &@pat p) -> ENV {
    ret e;
}

fn identity_update_env_for_expr[ENV](&ENV e, &@expr x) -> ENV {
    ret e;
}

fn identity_update_env_for_ty[ENV](&ENV e, &@ty t) -> ENV {
    ret e;
}


// Always-true traversal control fn.

fn always_keep_going[ENV](&ENV e) -> bool {
    ret true;
}


fn new_identity_fold[ENV]() -> ast_fold[ENV] {
    ret @rec
        (
         fold_path       = bind identity_fold_path[ENV](_,_,_),

         fold_ty_nil     = bind identity_fold_ty_nil[ENV](_,_),
         fold_ty_bot     = bind identity_fold_ty_bot[ENV](_,_),
         fold_ty_bool    = bind identity_fold_ty_bool[ENV](_,_),
         fold_ty_int     = bind identity_fold_ty_int[ENV](_,_),
         fold_ty_uint    = bind identity_fold_ty_uint[ENV](_,_),
         fold_ty_float   = bind identity_fold_ty_float[ENV](_,_),
         fold_ty_machine = bind identity_fold_ty_machine[ENV](_,_,_),
         fold_ty_char    = bind identity_fold_ty_char[ENV](_,_),
         fold_ty_str     = bind identity_fold_ty_str[ENV](_,_),
         fold_ty_box     = bind identity_fold_ty_box[ENV](_,_,_),
         fold_ty_vec     = bind identity_fold_ty_vec[ENV](_,_,_),
         fold_ty_tup     = bind identity_fold_ty_tup[ENV](_,_,_),
         fold_ty_rec     = bind identity_fold_ty_rec[ENV](_,_,_),
         fold_ty_obj     = bind identity_fold_ty_obj[ENV](_,_,_),
         fold_ty_fn      = bind identity_fold_ty_fn[ENV](_,_,_,_,_,_),
         fold_ty_path    = bind identity_fold_ty_path[ENV](_,_,_,_),
         fold_ty_chan    = bind identity_fold_ty_chan[ENV](_,_,_),
         fold_ty_port    = bind identity_fold_ty_port[ENV](_,_,_),

         fold_expr_vec    = bind identity_fold_expr_vec[ENV](_,_,_,_,_),
         fold_expr_tup    = bind identity_fold_expr_tup[ENV](_,_,_,_),
         fold_expr_rec    = bind identity_fold_expr_rec[ENV](_,_,_,_,_),
         fold_expr_call   = bind identity_fold_expr_call[ENV](_,_,_,_,_),
         fold_expr_self_method
                          = bind identity_fold_expr_self_method[ENV](_,_,_,_),
         fold_expr_bind   = bind identity_fold_expr_bind[ENV](_,_,_,_,_),
         fold_expr_spawn  = bind identity_fold_expr_spawn[ENV](_,_,_,_,_,_,_),
         fold_expr_binary = bind identity_fold_expr_binary[ENV](_,_,_,_,_,_),
         fold_expr_unary  = bind identity_fold_expr_unary[ENV](_,_,_,_,_),
         fold_expr_lit    = bind identity_fold_expr_lit[ENV](_,_,_,_),
         fold_expr_cast   = bind identity_fold_expr_cast[ENV](_,_,_,_,_),
         fold_expr_if     = bind identity_fold_expr_if[ENV](_,_,_,_,_,_),
         fold_expr_for    = bind identity_fold_expr_for[ENV](_,_,_,_,_,_),
         fold_expr_for_each
             = bind identity_fold_expr_for_each[ENV](_,_,_,_,_,_),
         fold_expr_while  = bind identity_fold_expr_while[ENV](_,_,_,_,_),
         fold_expr_do_while
                          = bind identity_fold_expr_do_while[ENV](_,_,_,_,_),
         fold_expr_alt    = bind identity_fold_expr_alt[ENV](_,_,_,_,_),
         fold_expr_block  = bind identity_fold_expr_block[ENV](_,_,_,_),
         fold_expr_assign = bind identity_fold_expr_assign[ENV](_,_,_,_,_),
         fold_expr_assign_op
                       = bind identity_fold_expr_assign_op[ENV](_,_,_,_,_,_),
         fold_expr_send   = bind identity_fold_expr_send[ENV](_,_,_,_,_),
         fold_expr_recv   = bind identity_fold_expr_recv[ENV](_,_,_,_,_),
         fold_expr_field  = bind identity_fold_expr_field[ENV](_,_,_,_,_),
         fold_expr_index  = bind identity_fold_expr_index[ENV](_,_,_,_,_),
         fold_expr_path   = bind identity_fold_expr_path[ENV](_,_,_,_),
         fold_expr_ext    = bind identity_fold_expr_ext[ENV](_,_,_,_,_,_,_),
         fold_expr_fail   = bind identity_fold_expr_fail[ENV](_,_,_),
         fold_expr_break  = bind identity_fold_expr_break[ENV](_,_,_),
         fold_expr_cont   = bind identity_fold_expr_cont[ENV](_,_,_),
         fold_expr_ret    = bind identity_fold_expr_ret[ENV](_,_,_,_),
         fold_expr_put    = bind identity_fold_expr_put[ENV](_,_,_,_),
         fold_expr_be     = bind identity_fold_expr_be[ENV](_,_,_,_),
         fold_expr_log    = bind identity_fold_expr_log[ENV](_,_,_,_,_),
         fold_expr_check
         = bind identity_fold_expr_check[ENV](_,_,_,_),
         fold_expr_assert
         = bind identity_fold_expr_assert[ENV](_,_,_,_),

         fold_expr_port   = bind identity_fold_expr_port[ENV](_,_,_),
         fold_expr_chan   = bind identity_fold_expr_chan[ENV](_,_,_,_),

         fold_expr_anon_obj   
                        = bind identity_fold_expr_anon_obj[ENV](_,_,_,_,_,_),

         fold_decl_local  = bind identity_fold_decl_local[ENV](_,_,_),
         fold_decl_item   = bind identity_fold_decl_item[ENV](_,_,_),

         fold_pat_wild    = bind identity_fold_pat_wild[ENV](_,_,_),
         fold_pat_lit     = bind identity_fold_pat_lit[ENV](_,_,_,_),
         fold_pat_bind    = bind identity_fold_pat_bind[ENV](_,_,_,_,_),
         fold_pat_tag     = bind identity_fold_pat_tag[ENV](_,_,_,_,_),

         fold_stmt_decl   = bind identity_fold_stmt_decl[ENV](_,_,_,_),
         fold_stmt_expr   = bind identity_fold_stmt_expr[ENV](_,_,_,_),

         fold_item_const= bind identity_fold_item_const[ENV](_,_,_,_,_,_,_),
         fold_item_fn   = bind identity_fold_item_fn[ENV](_,_,_,_,_,_,_),
         fold_native_item_fn =
             bind identity_fold_native_item_fn[ENV](_,_,_,_,_,_,_,_),
         fold_item_mod  = bind identity_fold_item_mod[ENV](_,_,_,_,_),
         fold_item_native_mod =
             bind identity_fold_item_native_mod[ENV](_,_,_,_,_),
         fold_item_ty   = bind identity_fold_item_ty[ENV](_,_,_,_,_,_,_),
         fold_native_item_ty =
             bind identity_fold_native_item_ty[ENV](_,_,_,_),
         fold_item_tag  = bind identity_fold_item_tag[ENV](_,_,_,_,_,_,_),
         fold_item_obj  = bind identity_fold_item_obj[ENV](_,_,_,_,_,_,_),

         fold_view_item_use =
             bind identity_fold_view_item_use[ENV](_,_,_,_,_,_),
         fold_view_item_import =
             bind identity_fold_view_item_import[ENV](_,_,_,_,_),
         fold_view_item_export =
             bind identity_fold_view_item_export[ENV](_,_,_),

         fold_ann = bind identity_fold_ann[ENV](_,_),

         fold_fn = bind identity_fold_fn[ENV](_,_,_,_),
         fold_fn_decl = bind identity_fold_fn_decl[ENV](_,_,_,_,_),
         fold_mod = bind identity_fold_mod[ENV](_,_),
         fold_native_mod = bind identity_fold_native_mod[ENV](_,_),
         fold_crate = bind identity_fold_crate[ENV](_,_,_,_),
         fold_obj = bind identity_fold_obj[ENV](_,_,_,_),
         fold_anon_obj = bind identity_fold_anon_obj[ENV](_,_,_,_),

         update_env_for_crate = bind identity_update_env_for_crate[ENV](_,_),
         update_env_for_item = bind identity_update_env_for_item[ENV](_,_),
         update_env_for_native_item =
             bind identity_update_env_for_native_item[ENV](_,_),
         update_env_for_view_item =
             bind identity_update_env_for_view_item[ENV](_,_),
         update_env_for_block = bind identity_update_env_for_block[ENV](_,_),
         update_env_for_stmt = bind identity_update_env_for_stmt[ENV](_,_),
         update_env_for_decl = bind identity_update_env_for_decl[ENV](_,_),
         update_env_for_pat = bind identity_update_env_for_pat[ENV](_,_),
         update_env_for_arm = bind identity_update_env_for_arm[ENV](_,_),
         update_env_for_expr = bind identity_update_env_for_expr[ENV](_,_),
         update_env_for_ty = bind identity_update_env_for_ty[ENV](_,_),

         keep_going = bind always_keep_going[ENV](_)
         );
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

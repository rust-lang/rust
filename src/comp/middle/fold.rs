import std.map.hashmap;
import std.option;
import std.option.some;
import std.option.none;

import util.common.new_str_hash;
import util.common.spanned;
import util.common.span;
import util.common.ty_mach;
import util.common.append;

import front.ast;
import front.ast.ident;
import front.ast.name;
import front.ast.path;
import front.ast.mutability;
import front.ast.ty;
import front.ast.expr;
import front.ast.stmt;
import front.ast.block;
import front.ast.item;
import front.ast.arg;
import front.ast.pat;
import front.ast.decl;
import front.ast.arm;
import front.ast.def;
import front.ast.def_id;
import front.ast.ann;

import std._vec;

type ast_fold[ENV] =
    @rec
    (
     // Name fold.
     (fn(&ENV e, &span sp, ast.name_ n) -> name)  fold_name,

     // Type folds.
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_nil,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_bool,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_int,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_uint,
     (fn(&ENV e, &span sp, ty_mach tm) -> @ty)    fold_ty_machine,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_char,
     (fn(&ENV e, &span sp) -> @ty)                fold_ty_str,
     (fn(&ENV e, &span sp, @ty t) -> @ty)         fold_ty_box,
     (fn(&ENV e, &span sp, @ty t) -> @ty)         fold_ty_vec,

     (fn(&ENV e, &span sp, vec[@ty] elts) -> @ty) fold_ty_tup,

     (fn(&ENV e, &span sp,
         vec[ast.ty_field] elts) -> @ty)          fold_ty_rec,

     (fn(&ENV e, &span sp,
         vec[ast.ty_method] meths) -> @ty)        fold_ty_obj,

     (fn(&ENV e, &span sp,
         vec[rec(ast.mode mode, @ty ty)] inputs,
         @ty output) -> @ty)                      fold_ty_fn,

     (fn(&ENV e, &span sp, ast.path p,
         &option.t[def] d) -> @ty)                fold_ty_path,

     (fn(&ENV e, &span sp, @ty t) -> @ty)         fold_ty_mutable,

     // Expr folds.
     (fn(&ENV e, &span sp,
         vec[@expr] es, ann a) -> @expr)          fold_expr_vec,

     (fn(&ENV e, &span sp,
         vec[ast.elt] es, ann a) -> @expr)        fold_expr_tup,

     (fn(&ENV e, &span sp,
         vec[ast.field] fields, ann a) -> @expr)  fold_expr_rec,

     (fn(&ENV e, &span sp,
         @expr f, vec[@expr] args,
         ann a) -> @expr)                         fold_expr_call,

     (fn(&ENV e, &span sp,
         @expr f, vec[option.t[@expr]] args,
         ann a) -> @expr)                         fold_expr_bind,

     (fn(&ENV e, &span sp,
         ast.binop,
         @expr lhs, @expr rhs,
         ann a) -> @expr)                         fold_expr_binary,

     (fn(&ENV e, &span sp,
         ast.unop, @expr e,
         ann a) -> @expr)                         fold_expr_unary,

     (fn(&ENV e, &span sp,
         @ast.lit, ann a) -> @expr)               fold_expr_lit,

     (fn(&ENV e, &span sp,
         @ast.expr e, @ast.ty ty,
         ann a) -> @expr)                         fold_expr_cast,

     (fn(&ENV e, &span sp,
         @expr cond, &block thn,
         &option.t[block] els,
         ann a) -> @expr)                         fold_expr_if,

     (fn(&ENV e, &span sp,
         @expr cond, &block body,
         ann a) -> @expr)                         fold_expr_while,

     (fn(&ENV e, &span sp,
         &block body, @expr cond,
         ann a) -> @expr)                         fold_expr_do_while,

     (fn(&ENV e, &span sp,
         @expr e, vec[arm] arms,
         ann a) -> @expr)                         fold_expr_alt,

     (fn(&ENV e, &span sp,
         &block blk, ann a) -> @expr)             fold_expr_block,

     (fn(&ENV e, &span sp,
         @expr lhs, @expr rhs,
         ann a) -> @expr)                         fold_expr_assign,

     (fn(&ENV e, &span sp,
         ast.binop,
         @expr lhs, @expr rhs,
         ann a) -> @expr)                         fold_expr_assign_op,

     (fn(&ENV e, &span sp,
         @expr e, ident i,
         ann a) -> @expr)                         fold_expr_field,

     (fn(&ENV e, &span sp,
         @expr e, @expr ix,
         ann a) -> @expr)                         fold_expr_index,

     (fn(&ENV e, &span sp,
         &name n,
         &option.t[def] d,
         ann a) -> @expr)                         fold_expr_name,

     // Decl folds.
     (fn(&ENV e, &span sp,
         @ast.local local) -> @decl)              fold_decl_local,

     (fn(&ENV e, &span sp,
         @item item) -> @decl)                    fold_decl_item,


     // Pat folds.
     (fn(&ENV e, &span sp,
         ann a) -> @pat)                          fold_pat_wild,

     (fn(&ENV e, &span sp,
         ident i, def_id did, ann a) -> @pat)     fold_pat_bind,

     (fn(&ENV e, &span sp,
         ident i, vec[@pat] args,
         option.t[ast.variant_def] d,
         ann a) -> @pat)                          fold_pat_tag,


     // Stmt folds.
     (fn(&ENV e, &span sp,
         @decl decl) -> @stmt)                    fold_stmt_decl,

     (fn(&ENV e, &span sp,
         &option.t[@expr] rv) -> @stmt)           fold_stmt_ret,

     (fn(&ENV e, &span sp,
         @expr e) -> @stmt)                       fold_stmt_log,

     (fn(&ENV e, &span sp,
         @expr e) -> @stmt)                       fold_stmt_check_expr,

     (fn(&ENV e, &span sp,
         @expr e) -> @stmt)                       fold_stmt_expr,

     // Item folds.
     (fn(&ENV e, &span sp, ident ident,
         @ty t, @expr e,
         def_id id, ann a) -> @item)              fold_item_const,

     (fn(&ENV e, &span sp, ident ident,
         &ast._fn f,
         vec[ast.ty_param] ty_params,
         def_id id, ann a) -> @item)              fold_item_fn,

     (fn(&ENV e, &span sp, ident ident,
         &ast._mod m, def_id id) -> @item)        fold_item_mod,

     (fn(&ENV e, &span sp, ident ident,
         @ty t, vec[ast.ty_param] ty_params,
         def_id id, ann a) -> @item)              fold_item_ty,

     (fn(&ENV e, &span sp, ident ident,
         vec[ast.variant] variants,
         vec[ast.ty_param] ty_params,
         def_id id) -> @item)                     fold_item_tag,

     (fn(&ENV e, &span sp, ident ident,
         &ast._obj ob,
         vec[ast.ty_param] ty_params,
         def_id id, ann a) -> @item)              fold_item_obj,

     // Additional nodes.
     (fn(&ENV e, &span sp,
         &ast.block_) -> block)                   fold_block,

     (fn(&ENV e, ast.effect effect,
         vec[arg] inputs,
         @ty output, &block body) -> ast._fn)     fold_fn,

     (fn(&ENV e, &ast._mod m) -> ast._mod)        fold_mod,

     (fn(&ENV e, &span sp,
         &ast._mod m) -> @ast.crate)              fold_crate,

     (fn(&ENV e,
         vec[ast.obj_field] fields,
         vec[@ast.method] methods) -> ast._obj)   fold_obj,

     // Env updates.
     (fn(&ENV e, @ast.crate c) -> ENV) update_env_for_crate,
     (fn(&ENV e, @item i) -> ENV) update_env_for_item,
     (fn(&ENV e, &block b) -> ENV) update_env_for_block,
     (fn(&ENV e, @stmt s) -> ENV) update_env_for_stmt,
     (fn(&ENV e, @decl i) -> ENV) update_env_for_decl,
     (fn(&ENV e, @pat p) -> ENV) update_env_for_pat,
     (fn(&ENV e, &arm a) -> ENV) update_env_for_arm,
     (fn(&ENV e, @expr x) -> ENV) update_env_for_expr,
     (fn(&ENV e, @ty t) -> ENV) update_env_for_ty,

     // Traversal control.
     (fn(&ENV v) -> bool) keep_going
     );


//// Fold drivers.

fn fold_name[ENV](&ENV env, ast_fold[ENV] fld, &name n) -> name {
    let vec[@ast.ty] tys_ = vec();
    for (@ast.ty t in n.node.types) {
        append[@ast.ty](tys_, fold_ty(env, fld, t));
    }
    let ast.name_ n_ = rec(ident=n.node.ident, types=tys_);
    ret fld.fold_name(env, n.span, n_);
}

fn fold_ty[ENV](&ENV env, ast_fold[ENV] fld, @ty t) -> @ty {
    let ENV env_ = fld.update_env_for_ty(env, t);

    if (!fld.keep_going(env_)) {
        ret t;
    }

    alt (t.node) {
        case (ast.ty_nil) { ret fld.fold_ty_nil(env_, t.span); }
        case (ast.ty_bool) { ret fld.fold_ty_bool(env_, t.span); }
        case (ast.ty_int) { ret fld.fold_ty_int(env_, t.span); }
        case (ast.ty_uint) { ret fld.fold_ty_uint(env_, t.span); }

        case (ast.ty_machine(?m)) {
            ret fld.fold_ty_machine(env_, t.span, m);
        }

        case (ast.ty_char) { ret fld.fold_ty_char(env_, t.span); }
        case (ast.ty_str) { ret fld.fold_ty_str(env_, t.span); }

        case (ast.ty_box(?ty)) {
            auto ty_ = fold_ty(env, fld, ty);
            ret fld.fold_ty_box(env_, t.span, ty_);
        }

        case (ast.ty_vec(?ty)) {
            auto ty_ = fold_ty(env, fld, ty);
            ret fld.fold_ty_vec(env_, t.span, ty_);
        }

        case (ast.ty_tup(?elts)) {
            let vec[@ty] elts_ = vec();
            for (@ty elt in elts) {
                append[@ty](elts_,fold_ty(env, fld, elt));
            }
            ret fld.fold_ty_tup(env_, t.span, elts_);
        }

        case (ast.ty_rec(?flds)) {
            let vec[ast.ty_field] flds_ = vec();
            for (ast.ty_field f in flds) {
                append[ast.ty_field]
                    (flds_, rec(ty=fold_ty(env, fld, f.ty) with f));
            }
            ret fld.fold_ty_rec(env_, t.span, flds_);
        }

        case (ast.ty_obj(?meths)) {
            let vec[ast.ty_method] meths_ = vec();
            for (ast.ty_method m in meths) {
                auto tfn = fld.fold_ty_fn(env_, t.span,
                                          m.inputs, m.output);
                alt (tfn.node) {
                    case (ast.ty_fn(?ins, ?out)) {
                        append[ast.ty_method]
                            (meths_, rec(inputs=ins, output=out with m));
                    }
                }
            }
            ret fld.fold_ty_obj(env_, t.span, meths_);
        }

        case (ast.ty_path(?pth, ?ref_opt)) {
            let vec[ast.name] path = vec();
            for (ast.name n in pth) {
                path += fold_name(env, fld, n);
            }
            ret fld.fold_ty_path(env_, t.span, path, ref_opt);
        }

        case (ast.ty_mutable(?ty)) {
            auto ty_ = fold_ty(env, fld, ty);
            ret fld.fold_ty_mutable(env_, t.span, ty_);
        }

        case (ast.ty_fn(?inputs, ?output)) {
            ret fld.fold_ty_fn(env_, t.span, inputs, output);
        }
    }
}

fn fold_decl[ENV](&ENV env, ast_fold[ENV] fld, @decl d) -> @decl {
    let ENV env_ = fld.update_env_for_decl(env, d);

    if (!fld.keep_going(env_)) {
        ret d;
    }

    alt (d.node) {
        case (ast.decl_local(?local)) {
            auto ty_ = none[@ast.ty];
            auto init_ = none[@ast.expr];
            alt (local.ty) {
                case (some[@ast.ty](?t)) {
                    ty_ = some[@ast.ty](fold_ty(env, fld, t));
                }
                case (_) { /* fall through */  }
            }
            alt (local.init) {
                case (some[@ast.expr](?e)) {
                    init_ = some[@ast.expr](fold_expr(env, fld, e));
                }
                case (_) { /* fall through */  }
            }
            let @ast.local local_ = @rec(ty=ty_, init=init_ with *local);
            ret fld.fold_decl_local(env_, d.span, local_);
        }

        case (ast.decl_item(?item)) {
            auto item_ = fold_item(env_, fld, item);
            ret fld.fold_decl_item(env_, d.span, item_);
        }
    }

    fail;
}

fn fold_pat[ENV](&ENV env, ast_fold[ENV] fld, @ast.pat p) -> @ast.pat {
    let ENV env_ = fld.update_env_for_pat(env, p);

    if (!fld.keep_going(env_)) {
        ret p;
    }

    alt (p.node) {
        case (ast.pat_wild(?t)) { ret fld.fold_pat_wild(env_, p.span, t); }
        case (ast.pat_bind(?id, ?did, ?t)) {
            ret fld.fold_pat_bind(env_, p.span, id, did, t);
        }
        case (ast.pat_tag(?id, ?pats, ?d, ?t)) {
            let vec[@ast.pat] ppats = vec();
            for (@ast.pat pat in pats) {
                ppats += vec(fold_pat(env_, fld, pat));
            }
            ret fld.fold_pat_tag(env_, p.span, id, ppats, d, t);
        }
    }
}

fn fold_exprs[ENV](&ENV env, ast_fold[ENV] fld, vec[@expr] es) -> vec[@expr] {
    let vec[@expr] exprs = vec();
    for (@expr e in es) {
        append[@expr](exprs, fold_expr(env, fld, e));
    }
    ret exprs;
}

fn fold_tup_elt[ENV](&ENV env, ast_fold[ENV] fld, &ast.elt e) -> ast.elt {
    ret rec(expr=fold_expr(env, fld, e.expr) with e);
}

fn fold_rec_field[ENV](&ENV env, ast_fold[ENV] fld, &ast.field f)
    -> ast.field {
    ret rec(expr=fold_expr(env, fld, f.expr) with f);
}

fn fold_expr[ENV](&ENV env, ast_fold[ENV] fld, &@expr e) -> @expr {

    let ENV env_ = fld.update_env_for_expr(env, e);

    if (!fld.keep_going(env_)) {
        ret e;
    }

    alt (e.node) {
        case (ast.expr_vec(?es, ?t)) {
            auto ees = fold_exprs(env_, fld, es);
            ret fld.fold_expr_vec(env_, e.span, ees, t);
        }

        case (ast.expr_tup(?es, ?t)) {
            let vec[ast.elt] elts = vec();
            for (ast.elt e in es) {
                elts += fold_tup_elt[ENV](env, fld, e);
            }
            ret fld.fold_expr_tup(env_, e.span, elts, t);
        }

        case (ast.expr_rec(?fs, ?t)) {
            let vec[ast.field] fields = vec();
            for (ast.field f in fs) {
                fields += fold_rec_field(env, fld, f);
            }
            ret fld.fold_expr_rec(env_, e.span, fields, t);
        }

        case (ast.expr_call(?f, ?args, ?t)) {
            auto ff = fold_expr(env_, fld, f);
            auto aargs = fold_exprs(env_, fld, args);
            ret fld.fold_expr_call(env_, e.span, ff, aargs, t);
        }

        case (ast.expr_bind(?f, ?args_opt, ?t)) {
            auto ff = fold_expr(env_, fld, f);
            let vec[option.t[@ast.expr]] aargs_opt = vec();
            for (option.t[@ast.expr] t_opt in args_opt) {
                alt (t_opt) {
                    case (some[@ast.expr](?e)) {
                        aargs_opt += vec(some(fold_expr(env_, fld, e)));
                    }
                    case (none[@ast.expr]) { /* empty */ }
                }
            }
            ret fld.fold_expr_bind(env_, e.span, ff, aargs_opt, t);
        }

        case (ast.expr_binary(?op, ?a, ?b, ?t)) {
            auto aa = fold_expr(env_, fld, a);
            auto bb = fold_expr(env_, fld, b);
            ret fld.fold_expr_binary(env_, e.span, op, aa, bb, t);
        }

        case (ast.expr_unary(?op, ?a, ?t)) {
            auto aa = fold_expr(env_, fld, a);
            ret fld.fold_expr_unary(env_, e.span, op, aa, t);
        }

        case (ast.expr_lit(?lit, ?t)) {
            ret fld.fold_expr_lit(env_, e.span, lit, t);
        }

        case (ast.expr_cast(?e, ?t, ?at)) {
            auto ee = fold_expr(env_, fld, e);
            auto tt = fold_ty(env, fld, t);
            ret fld.fold_expr_cast(env_, e.span, ee, tt, at);
        }

        case (ast.expr_if(?cnd, ?thn, ?els, ?t)) {
            auto ccnd = fold_expr(env_, fld, cnd);
            auto tthn = fold_block(env_, fld, thn);
            auto eels = none[block];
            alt (els) {
                case (some[block](?b)) {
                    eels = some(fold_block(env_, fld, b));
                }
                case (_) { /* fall through */  }
            }
            ret fld.fold_expr_if(env_, e.span, ccnd, tthn, eels, t);
        }

        case (ast.expr_while(?cnd, ?body, ?t)) {
            auto ccnd = fold_expr(env_, fld, cnd);
            auto bbody = fold_block(env_, fld, body);
            ret fld.fold_expr_while(env_, e.span, ccnd, bbody, t);
        }

        case (ast.expr_do_while(?body, ?cnd, ?t)) {
            auto bbody = fold_block(env_, fld, body);
            auto ccnd = fold_expr(env_, fld, cnd);
            ret fld.fold_expr_do_while(env_, e.span, bbody, ccnd, t);
        }

        case (ast.expr_alt(?expr, ?arms, ?t)) {
            auto eexpr = fold_expr(env_, fld, expr);
            let vec[ast.arm] aarms = vec();
            for (ast.arm a in arms) {
                aarms += vec(fold_arm(env_, fld, a));
            }
            ret fld.fold_expr_alt(env_, e.span, eexpr, aarms, t);
        }

        case (ast.expr_block(?b, ?t)) {
            auto bb = fold_block(env_, fld, b);
            ret fld.fold_expr_block(env_, e.span, bb, t);
        }

        case (ast.expr_assign(?lhs, ?rhs, ?t)) {
            auto llhs = fold_expr(env_, fld, lhs);
            auto rrhs = fold_expr(env_, fld, rhs);
            ret fld.fold_expr_assign(env_, e.span, llhs, rrhs, t);
        }

        case (ast.expr_assign_op(?op, ?lhs, ?rhs, ?t)) {
            auto llhs = fold_expr(env_, fld, lhs);
            auto rrhs = fold_expr(env_, fld, rhs);
            ret fld.fold_expr_assign_op(env_, e.span, op, llhs, rrhs, t);
        }

        case (ast.expr_field(?e, ?i, ?t)) {
            auto ee = fold_expr(env_, fld, e);
            ret fld.fold_expr_field(env_, e.span, ee, i, t);
        }

        case (ast.expr_index(?e, ?ix, ?t)) {
            auto ee = fold_expr(env_, fld, e);
            auto iix = fold_expr(env_, fld, ix);
            ret fld.fold_expr_index(env_, e.span, ee, iix, t);
        }

        case (ast.expr_name(?n, ?r, ?t)) {
            auto n_ = fold_name(env_, fld, n);
            ret fld.fold_expr_name(env_, e.span, n_, r, t);
        }
    }

    ret e;
}


fn fold_stmt[ENV](&ENV env, ast_fold[ENV] fld, &@stmt s) -> @stmt {

    let ENV env_ = fld.update_env_for_stmt(env, s);

    if (!fld.keep_going(env_)) {
        ret s;
    }

    alt (s.node) {
        case (ast.stmt_decl(?d)) {
            auto dd = fold_decl(env_, fld, d);
            ret fld.fold_stmt_decl(env_, s.span, dd);
        }

        case (ast.stmt_ret(?oe)) {
            auto oee = none[@expr];
            alt (oe) {
                case (some[@expr](?e)) {
                    oee = some(fold_expr(env_, fld, e));
                }
                case (_) { /* fall through */  }
            }
            ret fld.fold_stmt_ret(env_, s.span, oee);
        }

        case (ast.stmt_log(?e)) {
            auto ee = fold_expr(env_, fld, e);
            ret fld.fold_stmt_log(env_, s.span, ee);
        }

        case (ast.stmt_check_expr(?e)) {
            auto ee = fold_expr(env_, fld, e);
            ret fld.fold_stmt_check_expr(env_, s.span, ee);
        }

        case (ast.stmt_expr(?e)) {
            auto ee = fold_expr(env_, fld, e);
            ret fld.fold_stmt_expr(env_, s.span, ee);
        }
    }
    ret s;
}

fn fold_block[ENV](&ENV env, ast_fold[ENV] fld, &block blk) -> block {

    let ENV env_ = fld.update_env_for_block(env, blk);

    if (!fld.keep_going(env_)) {
        ret blk;
    }

    let vec[@ast.stmt] stmts = vec();
    for (@ast.stmt s in blk.node.stmts) {
        append[@ast.stmt](stmts, fold_stmt[ENV](env_, fld, s));
    }

    auto expr = none[@ast.expr];
    alt (blk.node.expr) {
        case (some[@ast.expr](?e)) {
            expr = some[@ast.expr](fold_expr[ENV](env_, fld, e));
        }
        case (none[@ast.expr]) {
            // empty
        }
    }

    // FIXME: should we reindex?
    ret respan(blk.span, rec(stmts=stmts, expr=expr, index=blk.node.index));
}

fn fold_arm[ENV](&ENV env, ast_fold[ENV] fld, &arm a) -> arm {
    let ENV env_ = fld.update_env_for_arm(env, a);
    auto ppat = fold_pat(env_, fld, a.pat);
    auto bblock = fold_block(env_, fld, a.block);
    ret rec(pat=ppat, block=bblock, index=a.index);
}

fn fold_arg[ENV](&ENV env, ast_fold[ENV] fld, &arg a) -> arg {
    auto ty = fold_ty(env, fld, a.ty);
    ret rec(ty=ty with a);
}


fn fold_fn[ENV](&ENV env, ast_fold[ENV] fld, &ast._fn f) -> ast._fn {

    let vec[ast.arg] inputs = vec();
    for (ast.arg a in f.inputs) {
        inputs += fold_arg(env, fld, a);
    }
    auto output = fold_ty[ENV](env, fld, f.output);
    auto body = fold_block[ENV](env, fld, f.body);

    ret fld.fold_fn(env, f.effect, inputs, output, body);
}


fn fold_obj_field[ENV](&ENV env, ast_fold[ENV] fld,
                       &ast.obj_field f) -> ast.obj_field {
    auto ty = fold_ty(env, fld, f.ty);
    ret rec(ty=ty with f);
}


fn fold_method[ENV](&ENV env, ast_fold[ENV] fld,
                    @ast.method m) -> @ast.method {
    auto meth = fold_fn(env, fld, m.node.meth);
    ret @rec(node=rec(meth=meth with m.node) with *m);
}


fn fold_obj[ENV](&ENV env, ast_fold[ENV] fld, &ast._obj ob) -> ast._obj {

    let vec[ast.obj_field] fields = vec();
    let vec[@ast.method] meths = vec();
    for (ast.obj_field f in ob.fields) {
        fields += fold_obj_field(env, fld, f);
    }
    let vec[ast.ty_param] tp = vec();
    for (@ast.method m in ob.methods) {
        // Fake-up an ast.item for this method.
        // FIXME: this is kinda awful. Maybe we should reformulate
        // the way we store methods in the AST?
        let @ast.item i = @rec(node=ast.item_fn(m.node.ident,
                                                m.node.meth,
                                                tp,
                                                m.node.id,
                                                m.node.ann),
                               span=m.span);
        let ENV _env = fld.update_env_for_item(env, i);
        append[@ast.method](meths, fold_method(_env, fld, m));
    }
    ret fld.fold_obj(env, fields, meths);
}

fn fold_item[ENV](&ENV env, ast_fold[ENV] fld, @item i) -> @item {

    let ENV env_ = fld.update_env_for_item(env, i);

    if (!fld.keep_going(env_)) {
        ret i;
    }

    alt (i.node) {

        case (ast.item_const(?ident, ?t, ?e, ?id, ?ann)) {
            let @ast.ty t_ = fold_ty[ENV](env_, fld, t);
            let @ast.expr e_ = fold_expr(env_, fld, e);
            ret fld.fold_item_const(env_, i.span, ident, t_, e_, id, ann);
        }

        case (ast.item_fn(?ident, ?ff, ?tps, ?id, ?ann)) {
            let ast._fn ff_ = fold_fn[ENV](env_, fld, ff);
            ret fld.fold_item_fn(env_, i.span, ident, ff_, tps, id, ann);
        }

        case (ast.item_mod(?ident, ?mm, ?id)) {
            let ast._mod mm_ = fold_mod[ENV](env_, fld, mm);
            ret fld.fold_item_mod(env_, i.span, ident, mm_, id);
        }

        case (ast.item_ty(?ident, ?ty, ?params, ?id, ?ann)) {
            let @ast.ty ty_ = fold_ty[ENV](env_, fld, ty);
            ret fld.fold_item_ty(env_, i.span, ident, ty_, params, id, ann);
        }

        case (ast.item_tag(?ident, ?variants, ?ty_params, ?id)) {
            let vec[ast.variant] new_variants = vec();
            for (ast.variant v in variants) {
                let vec[ast.variant_arg] new_args = vec();
                for (ast.variant_arg va in v.args) {
                    auto new_ty = fold_ty[ENV](env_, fld, va.ty);
                    new_args += vec(rec(ty=new_ty, id=va.id));
                }
                new_variants += rec(name=v.name, args=new_args, id=v.id,
                                    ann=v.ann);
            }
            ret fld.fold_item_tag(env_, i.span, ident, new_variants,
                                  ty_params, id);
        }

        case (ast.item_obj(?ident, ?ob, ?tps, ?id, ?ann)) {
            let ast._obj ob_ = fold_obj[ENV](env_, fld, ob);
            ret fld.fold_item_obj(env_, i.span, ident, ob_, tps, id, ann);
        }

    }

    fail;
}


fn fold_mod[ENV](&ENV e, ast_fold[ENV] fld, &ast._mod m) -> ast._mod {

    let vec[@item] items = vec();

    for (@item i in m.items) {
        append[@item](items, fold_item[ENV](e, fld, i));
    }

    ret fld.fold_mod(e, rec(items=items with m));
 }

fn fold_crate[ENV](&ENV env, ast_fold[ENV] fld, @ast.crate c) -> @ast.crate {
    let ENV env_ = fld.update_env_for_crate(env, c);
    let ast._mod m = fold_mod[ENV](env_, fld, c.node.module);
    ret fld.fold_crate(env_, c.span, m);
}

//// Identity folds.

fn respan[T](&span sp, &T t) -> spanned[T] {
    ret rec(node=t, span=sp);
}


// Name identity.

fn identity_fold_name[ENV](&ENV env, &span sp, ast.name_ n) -> name {
    ret respan(sp, n);
}

// Type identities.

fn identity_fold_ty_nil[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast.ty_nil);
}

fn identity_fold_ty_bool[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast.ty_bool);
}

fn identity_fold_ty_int[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast.ty_int);
}

fn identity_fold_ty_uint[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast.ty_uint);
}

fn identity_fold_ty_machine[ENV](&ENV env, &span sp,
                                 ty_mach tm) -> @ty {
    ret @respan(sp, ast.ty_machine(tm));
}

fn identity_fold_ty_char[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast.ty_char);
}

fn identity_fold_ty_str[ENV](&ENV env, &span sp) -> @ty {
    ret @respan(sp, ast.ty_str);
}

fn identity_fold_ty_box[ENV](&ENV env, &span sp, @ty t) -> @ty {
    ret @respan(sp, ast.ty_box(t));
}

fn identity_fold_ty_vec[ENV](&ENV env, &span sp, @ty t) -> @ty {
    ret @respan(sp, ast.ty_vec(t));
}

fn identity_fold_ty_tup[ENV](&ENV env, &span sp,
                             vec[@ty] elts) -> @ty {
    ret @respan(sp, ast.ty_tup(elts));
}

fn identity_fold_ty_rec[ENV](&ENV env, &span sp,
                             vec[ast.ty_field] elts) -> @ty {
    ret @respan(sp, ast.ty_rec(elts));
}

fn identity_fold_ty_obj[ENV](&ENV env, &span sp,
                             vec[ast.ty_method] meths) -> @ty {
    ret @respan(sp, ast.ty_obj(meths));
}

fn identity_fold_ty_fn[ENV](&ENV env, &span sp,
                            vec[rec(ast.mode mode, @ty ty)] inputs,
                            @ty output) -> @ty {
    ret @respan(sp, ast.ty_fn(inputs, output));
}

fn identity_fold_ty_path[ENV](&ENV env, &span sp, ast.path p,
                        &option.t[def] d) -> @ty {
    ret @respan(sp, ast.ty_path(p, d));
}

fn identity_fold_ty_mutable[ENV](&ENV env, &span sp, @ty t) -> @ty {
    ret @respan(sp, ast.ty_mutable(t));
}


// Expr identities.

fn identity_fold_expr_vec[ENV](&ENV env, &span sp, vec[@expr] es,
                               ann a) -> @expr {
    ret @respan(sp, ast.expr_vec(es, a));
}

fn identity_fold_expr_tup[ENV](&ENV env, &span sp,
                               vec[ast.elt] es, ann a) -> @expr {
    ret @respan(sp, ast.expr_tup(es, a));
}

fn identity_fold_expr_rec[ENV](&ENV env, &span sp,
                               vec[ast.field] fields, ann a) -> @expr {
    ret @respan(sp, ast.expr_rec(fields, a));
}

fn identity_fold_expr_call[ENV](&ENV env, &span sp, @expr f,
                                vec[@expr] args, ann a) -> @expr {
    ret @respan(sp, ast.expr_call(f, args, a));
}

fn identity_fold_expr_bind[ENV](&ENV env, &span sp, @expr f,
                                vec[option.t[@expr]] args_opt, ann a)
        -> @expr {
    ret @respan(sp, ast.expr_bind(f, args_opt, a));
}

fn identity_fold_expr_binary[ENV](&ENV env, &span sp, ast.binop b,
                                  @expr lhs, @expr rhs,
                                  ann a) -> @expr {
    ret @respan(sp, ast.expr_binary(b, lhs, rhs, a));
}

fn identity_fold_expr_unary[ENV](&ENV env, &span sp,
                                 ast.unop u, @expr e, ann a)
        -> @expr {
    ret @respan(sp, ast.expr_unary(u, e, a));
}

fn identity_fold_expr_lit[ENV](&ENV env, &span sp, @ast.lit lit,
                               ann a) -> @expr {
    ret @respan(sp, ast.expr_lit(lit, a));
}

fn identity_fold_expr_cast[ENV](&ENV env, &span sp, @ast.expr e,
                                @ast.ty t, ann a) -> @expr {
    ret @respan(sp, ast.expr_cast(e, t, a));
}

fn identity_fold_expr_if[ENV](&ENV env, &span sp,
                              @expr cond, &block thn,
                              &option.t[block] els, ann a) -> @expr {
    ret @respan(sp, ast.expr_if(cond, thn, els, a));
}

fn identity_fold_expr_while[ENV](&ENV env, &span sp,
                                 @expr cond, &block body, ann a) -> @expr {
    ret @respan(sp, ast.expr_while(cond, body, a));
}

fn identity_fold_expr_do_while[ENV](&ENV env, &span sp,
                                    &block body, @expr cond, ann a) -> @expr {
    ret @respan(sp, ast.expr_do_while(body, cond, a));
}

fn identity_fold_expr_alt[ENV](&ENV env, &span sp,
                               @expr e, vec[arm] arms, ann a) -> @expr {
    ret @respan(sp, ast.expr_alt(e, arms, a));
}

fn identity_fold_expr_block[ENV](&ENV env, &span sp, &block blk,
                                 ann a) -> @expr {
    ret @respan(sp, ast.expr_block(blk, a));
}

fn identity_fold_expr_assign[ENV](&ENV env, &span sp,
                                  @expr lhs, @expr rhs, ann a)
        -> @expr {
    ret @respan(sp, ast.expr_assign(lhs, rhs, a));
}

fn identity_fold_expr_assign_op[ENV](&ENV env, &span sp, ast.binop op,
                                     @expr lhs, @expr rhs, ann a)
        -> @expr {
    ret @respan(sp, ast.expr_assign_op(op, lhs, rhs, a));
}

fn identity_fold_expr_field[ENV](&ENV env, &span sp,
                                 @expr e, ident i, ann a) -> @expr {
    ret @respan(sp, ast.expr_field(e, i, a));
}

fn identity_fold_expr_index[ENV](&ENV env, &span sp,
                                 @expr e, @expr ix, ann a) -> @expr {
    ret @respan(sp, ast.expr_index(e, ix, a));
}

fn identity_fold_expr_name[ENV](&ENV env, &span sp,
                                &name n, &option.t[def] d,
                                ann a) -> @expr {
    ret @respan(sp, ast.expr_name(n, d, a));
}


// Decl identities.

fn identity_fold_decl_local[ENV](&ENV e, &span sp,
                                 @ast.local local) -> @decl {
    ret @respan(sp, ast.decl_local(local));
}

fn identity_fold_decl_item[ENV](&ENV e, &span sp, @item i) -> @decl {
    ret @respan(sp, ast.decl_item(i));
}


// Pat identities.

fn identity_fold_pat_wild[ENV](&ENV e, &span sp, ann a) -> @pat {
    ret @respan(sp, ast.pat_wild(a));
}

fn identity_fold_pat_bind[ENV](&ENV e, &span sp, ident i, def_id did, ann a)
        -> @pat {
    ret @respan(sp, ast.pat_bind(i, did, a));
}

fn identity_fold_pat_tag[ENV](&ENV e, &span sp, ident i, vec[@pat] args,
                              option.t[ast.variant_def] d, ann a) -> @pat {
    ret @respan(sp, ast.pat_tag(i, args, d, a));
}


// Stmt identities.

fn identity_fold_stmt_decl[ENV](&ENV env, &span sp, @decl d) -> @stmt {
    ret @respan(sp, ast.stmt_decl(d));
}

fn identity_fold_stmt_ret[ENV](&ENV env, &span sp,
                               &option.t[@expr] rv) -> @stmt {
    ret @respan(sp, ast.stmt_ret(rv));
}

fn identity_fold_stmt_log[ENV](&ENV e, &span sp, @expr x) -> @stmt {
    ret @respan(sp, ast.stmt_log(x));
}

fn identity_fold_stmt_check_expr[ENV](&ENV e, &span sp, @expr x) -> @stmt {
    ret @respan(sp, ast.stmt_check_expr(x));
}

fn identity_fold_stmt_expr[ENV](&ENV e, &span sp, @expr x) -> @stmt {
    ret @respan(sp, ast.stmt_expr(x));
}


// Item identities.

fn identity_fold_item_const[ENV](&ENV e, &span sp, ident i,
                                 @ty t, @expr ex,
                                 def_id id, ann a) -> @item {
    ret @respan(sp, ast.item_const(i, t, ex, id, a));
}

fn identity_fold_item_fn[ENV](&ENV e, &span sp, ident i,
                              &ast._fn f, vec[ast.ty_param] ty_params,
                              def_id id, ann a) -> @item {
    ret @respan(sp, ast.item_fn(i, f, ty_params, id, a));
}

fn identity_fold_item_mod[ENV](&ENV e, &span sp, ident i,
                               &ast._mod m, def_id id) -> @item {
    ret @respan(sp, ast.item_mod(i, m, id));
}

fn identity_fold_item_ty[ENV](&ENV e, &span sp, ident i,
                              @ty t, vec[ast.ty_param] ty_params,
                              def_id id, ann a) -> @item {
    ret @respan(sp, ast.item_ty(i, t, ty_params, id, a));
}

fn identity_fold_item_tag[ENV](&ENV e, &span sp, ident i,
                               vec[ast.variant] variants,
                               vec[ast.ty_param] ty_params,
                               def_id id) -> @item {
    ret @respan(sp, ast.item_tag(i, variants, ty_params, id));
}

fn identity_fold_item_obj[ENV](&ENV e, &span sp, ident i,
                               &ast._obj ob, vec[ast.ty_param] ty_params,
                               def_id id, ann a) -> @item {
    ret @respan(sp, ast.item_obj(i, ob, ty_params, id, a));
}


// Additional identities.

fn identity_fold_block[ENV](&ENV e, &span sp, &ast.block_ blk) -> block {
    ret respan(sp, blk);
}

fn identity_fold_fn[ENV](&ENV e,
                         ast.effect effect,
                         vec[arg] inputs,
                         @ast.ty output,
                         &block body) -> ast._fn {
    ret rec(effect=effect, inputs=inputs, output=output, body=body);
}

fn identity_fold_mod[ENV](&ENV e, &ast._mod m) -> ast._mod {
    ret m;
}

fn identity_fold_crate[ENV](&ENV e, &span sp, &ast._mod m) -> @ast.crate {
    ret @respan(sp, rec(module=m));
}

fn identity_fold_obj[ENV](&ENV e,
                          vec[ast.obj_field] fields,
                          vec[@ast.method] methods) -> ast._obj {
    ret rec(fields=fields, methods=methods);
}


// Env update identities.

fn identity_update_env_for_crate[ENV](&ENV e, @ast.crate c) -> ENV {
    ret e;
}

fn identity_update_env_for_item[ENV](&ENV e, @item i) -> ENV {
    ret e;
}

fn identity_update_env_for_block[ENV](&ENV e, &block b) -> ENV {
    ret e;
}

fn identity_update_env_for_stmt[ENV](&ENV e, @stmt s) -> ENV {
    ret e;
}

fn identity_update_env_for_decl[ENV](&ENV e, @decl d) -> ENV {
    ret e;
}

fn identity_update_env_for_arm[ENV](&ENV e, &arm a) -> ENV {
    ret e;
}

fn identity_update_env_for_pat[ENV](&ENV e, @pat p) -> ENV {
    ret e;
}

fn identity_update_env_for_expr[ENV](&ENV e, @expr x) -> ENV {
    ret e;
}

fn identity_update_env_for_ty[ENV](&ENV e, @ty t) -> ENV {
    ret e;
}


// Always-true traversal control fn.

fn always_keep_going[ENV](&ENV e) -> bool {
    ret true;
}


fn new_identity_fold[ENV]() -> ast_fold[ENV] {
    ret @rec
        (
         fold_name       = bind identity_fold_name[ENV](_,_,_),

         fold_ty_nil     = bind identity_fold_ty_nil[ENV](_,_),
         fold_ty_bool    = bind identity_fold_ty_bool[ENV](_,_),
         fold_ty_int     = bind identity_fold_ty_int[ENV](_,_),
         fold_ty_uint    = bind identity_fold_ty_uint[ENV](_,_),
         fold_ty_machine = bind identity_fold_ty_machine[ENV](_,_,_),
         fold_ty_char    = bind identity_fold_ty_char[ENV](_,_),
         fold_ty_str     = bind identity_fold_ty_str[ENV](_,_),
         fold_ty_box     = bind identity_fold_ty_box[ENV](_,_,_),
         fold_ty_vec     = bind identity_fold_ty_vec[ENV](_,_,_),
         fold_ty_tup     = bind identity_fold_ty_tup[ENV](_,_,_),
         fold_ty_rec     = bind identity_fold_ty_rec[ENV](_,_,_),
         fold_ty_obj     = bind identity_fold_ty_obj[ENV](_,_,_),
         fold_ty_fn      = bind identity_fold_ty_fn[ENV](_,_,_,_),
         fold_ty_path    = bind identity_fold_ty_path[ENV](_,_,_,_),
         fold_ty_mutable = bind identity_fold_ty_mutable[ENV](_,_,_),

         fold_expr_vec    = bind identity_fold_expr_vec[ENV](_,_,_,_),
         fold_expr_tup    = bind identity_fold_expr_tup[ENV](_,_,_,_),
         fold_expr_rec    = bind identity_fold_expr_rec[ENV](_,_,_,_),
         fold_expr_call   = bind identity_fold_expr_call[ENV](_,_,_,_,_),
         fold_expr_bind   = bind identity_fold_expr_bind[ENV](_,_,_,_,_),
         fold_expr_binary = bind identity_fold_expr_binary[ENV](_,_,_,_,_,_),
         fold_expr_unary  = bind identity_fold_expr_unary[ENV](_,_,_,_,_),
         fold_expr_lit    = bind identity_fold_expr_lit[ENV](_,_,_,_),
         fold_expr_cast   = bind identity_fold_expr_cast[ENV](_,_,_,_,_),
         fold_expr_if     = bind identity_fold_expr_if[ENV](_,_,_,_,_,_),
         fold_expr_while  = bind identity_fold_expr_while[ENV](_,_,_,_,_),
         fold_expr_do_while
                          = bind identity_fold_expr_do_while[ENV](_,_,_,_,_),
         fold_expr_alt    = bind identity_fold_expr_alt[ENV](_,_,_,_,_),
         fold_expr_block  = bind identity_fold_expr_block[ENV](_,_,_,_),
         fold_expr_assign = bind identity_fold_expr_assign[ENV](_,_,_,_,_),
         fold_expr_assign_op
                       = bind identity_fold_expr_assign_op[ENV](_,_,_,_,_,_),
         fold_expr_field  = bind identity_fold_expr_field[ENV](_,_,_,_,_),
         fold_expr_index  = bind identity_fold_expr_index[ENV](_,_,_,_,_),
         fold_expr_name   = bind identity_fold_expr_name[ENV](_,_,_,_,_),

         fold_decl_local  = bind identity_fold_decl_local[ENV](_,_,_),
         fold_decl_item   = bind identity_fold_decl_item[ENV](_,_,_),

         fold_pat_wild    = bind identity_fold_pat_wild[ENV](_,_,_),
         fold_pat_bind    = bind identity_fold_pat_bind[ENV](_,_,_,_,_),
         fold_pat_tag     = bind identity_fold_pat_tag[ENV](_,_,_,_,_,_),

         fold_stmt_decl   = bind identity_fold_stmt_decl[ENV](_,_,_),
         fold_stmt_ret    = bind identity_fold_stmt_ret[ENV](_,_,_),
         fold_stmt_log    = bind identity_fold_stmt_log[ENV](_,_,_),
         fold_stmt_check_expr
                          = bind identity_fold_stmt_check_expr[ENV](_,_,_),
         fold_stmt_expr   = bind identity_fold_stmt_expr[ENV](_,_,_),

         fold_item_const= bind identity_fold_item_const[ENV](_,_,_,_,_,_,_),
         fold_item_fn   = bind identity_fold_item_fn[ENV](_,_,_,_,_,_,_),
         fold_item_mod  = bind identity_fold_item_mod[ENV](_,_,_,_,_),
         fold_item_ty   = bind identity_fold_item_ty[ENV](_,_,_,_,_,_,_),
         fold_item_tag  = bind identity_fold_item_tag[ENV](_,_,_,_,_,_),
         fold_item_obj  = bind identity_fold_item_obj[ENV](_,_,_,_,_,_,_),

         fold_block = bind identity_fold_block[ENV](_,_,_),
         fold_fn = bind identity_fold_fn[ENV](_,_,_,_,_),
         fold_mod = bind identity_fold_mod[ENV](_,_),
         fold_crate = bind identity_fold_crate[ENV](_,_,_),
         fold_obj = bind identity_fold_obj[ENV](_,_,_),

         update_env_for_crate = bind identity_update_env_for_crate[ENV](_,_),
         update_env_for_item = bind identity_update_env_for_item[ENV](_,_),
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
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

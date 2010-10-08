import std.map.hashmap;
import util.common.new_str_hash;
import util.common.spanned;
import util.common.span;
import util.common.option;
import util.common.some;
import util.common.none;
import util.common.ty_mach;


import front.ast;
import front.ast.ident;
import front.ast.name;
import front.ast.ty;
import front.ast.expr;
import front.ast.stmt;
import front.ast.block;
import front.ast.item;
import front.ast.slot;
import front.ast.decl;
import front.ast.referent;

import std._vec;

import std.util.operator;

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
     (fn(&ENV e, &span sp, ast.path p,
         &option[referent] r) -> @ty)             fold_ty_path,

     // Expr folds.
     (fn(&ENV e, &span sp,
         vec[@expr] es) -> @expr)                 fold_expr_vec,

     (fn(&ENV e, &span sp,
         vec[@expr] es) -> @expr)                 fold_expr_tup,

     (fn(&ENV e, &span sp,
         vec[tup(ident,@expr)] fields) -> @expr)  fold_expr_rec,

     (fn(&ENV e, &span sp,
         @expr f, vec[@expr] args) -> @expr)      fold_expr_call,

     (fn(&ENV e, &span sp,
         ast.binop,
         @expr lhs, @expr rhs) -> @expr)          fold_expr_binary,

     (fn(&ENV e, &span sp,
         ast.unop, @expr e) -> @expr)             fold_expr_unary,

     (fn(&ENV e, &span sp,
         @ast.lit) -> @expr)                      fold_expr_lit,

     (fn(&ENV e, &span sp,
         &name n,
         &option[referent] r) -> @expr)           fold_expr_name,

     (fn(&ENV e, &span sp,
         @expr e, ident i) -> @expr)              fold_expr_field,

     (fn(&ENV e, &span sp,
         @expr e, @expr ix) -> @expr)             fold_expr_index,

     (fn(&ENV e, &span sp,
         @expr cond, block thn,
         &option[block] els) -> @expr)            fold_expr_if,

     (fn(&ENV e, &span sp,
         block blk) -> @expr)                     fold_expr_block,


     // Decl folds.
     (fn(&ENV e, &span sp,
         ident ident, bool infer,
         &option[@ty] ty) -> @decl)               fold_decl_local,

     (fn(&ENV e, &span sp,
         &name name, @item item) -> @decl)        fold_decl_item,


     // Stmt folds.
     (fn(&ENV e, &span sp,
         @decl decl) -> @stmt)                    fold_stmt_decl,

     (fn(&ENV e, &span sp,
         &option[@expr] rv) -> @stmt)             fold_stmt_ret,

     (fn(&ENV e, &span sp,
         @expr e) -> @stmt)                       fold_stmt_log,

     (fn(&ENV e, &span sp,
         @expr e) -> @stmt)                       fold_stmt_expr,

     // Item folds.
     (fn(&ENV e, &span sp,
         &ast._fn f, ast.item_id id) -> @item)    fold_item_fn,

     (fn(&ENV e, &span sp,
         &ast._mod m) -> @item)                   fold_item_mod,

     (fn(&ENV e, &span sp,
         @ty t, ast.item_id id) -> @item)         fold_item_ty,

     // Additional nodes.
     (fn(&ENV e, &span sp,
         vec[@stmt] stmts) -> block)              fold_block,

     (fn(&ENV e, vec[ast.input] inputs,
         &slot output, block body) -> ast._fn)    fold_fn,

     (fn(&ENV e, &ast._mod m) -> ast._mod)        fold_mod,

     (fn(&ENV e, &span sp,
         &ast._mod m) -> @ast.crate)              fold_crate,

     // Env updates.
     (fn(&ENV e, @ast.crate c) -> ENV) update_env_for_crate,
     (fn(&ENV e, @item i) -> ENV) update_env_for_item,
     (fn(&ENV e, @stmt s) -> ENV) update_env_for_stmt,
     (fn(&ENV e, @expr x) -> ENV) update_env_for_expr,
     (fn(&ENV e, @ty t) -> ENV) update_env_for_ty,

     // Traversal control.
     (fn(&ENV v) -> bool) keep_going
     );


//// Fold drivers.

// FIXME: Finish these.

fn fold_expr_name[ENV](&ENV env, ast_fold[ENV] fld, &name n,
                  &option[referent] r) -> tup(name,option[referent]) {
    ret tup(n,r);
}

fn fold_ty[ENV](&ENV env, ast_fold[ENV] fld, @ty t) -> @ty {
    ret t;
}

fn fold_decl[ENV](&ENV env, ast_fold[ENV] fld, @decl d) -> @decl {
    ret d;
}

fn fold_exprs[ENV](&ENV env, ast_fold[ENV] fld, vec[@expr] e) -> vec[@expr] {
    let operator[@expr, @expr] fe = bind fold_expr[ENV](env, fld, _);
    ret _vec.map[@expr, @expr](fe, e);
}

fn fold_rec_entry[ENV](&ENV env, ast_fold[ENV] fld, &tup(ident,@expr) e)
    -> tup(ident,@expr) {
    ret tup(e._0, fold_expr(env, fld, e._1));
}

fn fold_expr[ENV](&ENV env, ast_fold[ENV] fld, &@expr e) -> @expr {

    let ENV env_ = fld.update_env_for_expr(env, e);

    if (!fld.keep_going(env_)) {
        ret e;
    }

    alt (e.node) {
        case (ast.expr_vec(?es)) {
            auto ees = fold_exprs(env_, fld, es);
            ret fld.fold_expr_vec(env_, e.span, ees);
        }

        case (ast.expr_tup(?es)) {
            auto ees = fold_exprs(env_, fld, es);
            ret fld.fold_expr_vec(env_, e.span, ees);
        }

        case (ast.expr_rec(?es)) {
            let operator[tup(ident,@expr), tup(ident,@expr)] fe =
                bind fold_rec_entry[ENV](env, fld, _);
            auto ees = _vec.map[tup(ident,@expr), tup(ident,@expr)](fe, es);
            ret fld.fold_expr_rec(env_, e.span, ees);
        }

        case (ast.expr_call(?f, ?args)) {
            auto ff = fold_expr(env_, fld, f);
            auto aargs = fold_exprs(env_, fld, args);
            ret fld.fold_expr_call(env_, e.span, ff, aargs);
        }

        case (ast.expr_binary(?op, ?a, ?b)) {
            auto aa = fold_expr(env_, fld, a);
            auto bb = fold_expr(env_, fld, b);
            ret fld.fold_expr_binary(env_, e.span, op, aa, bb);
        }

        case (ast.expr_unary(?op, ?a)) {
            auto aa = fold_expr(env_, fld, a);
            ret fld.fold_expr_unary(env_, e.span, op, a);
        }

        case (ast.expr_lit(?lit)) {
            ret fld.fold_expr_lit(env_, e.span, lit);
        }

        case (ast.expr_name(?n, ?r)) {
            auto nn = fold_expr_name(env_, fld, n, r);
            ret fld.fold_expr_name(env_, e.span, nn._0, nn._1);
        }

        case (ast.expr_field(?e, ?i)) {
            auto ee = fold_expr(env_, fld, e);
            ret fld.fold_expr_field(env_, e.span, ee, i);
        }

        case (ast.expr_index(?e, ?i)) {
            auto ee = fold_expr(env_, fld, e);
            auto ii = fold_expr(env_, fld, i);
            ret fld.fold_expr_index(env_, e.span, ee, ii);
        }

        case (ast.expr_if(?cnd, ?thn, ?els)) {
            auto ccnd = fold_expr(env_, fld, cnd);
            auto tthn = fold_block(env_, fld, thn);
            auto eels = none[block];
            alt (els) {
                case (some[block](?b)) {
                    eels = some(fold_block(env_, fld, b));
                }
            }
            ret fld.fold_expr_if(env_, e.span, ccnd, tthn, eels);
        }

        case (ast.expr_block(?b)) {
            auto bb = fold_block(env_, fld, b);
            ret fld.fold_expr_block(env_, e.span, bb);
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
            ret fld.fold_stmt_decl(env_, s.span, d);
        }

        case (ast.stmt_ret(?oe)) {
            auto oee = none[@expr];
            alt (oe) {
                case (some[@expr](?e)) {
                    oee = some(fold_expr(env_, fld, e));
                }
            }
            ret fld.fold_stmt_ret(env_, s.span, oee);
        }

        case (ast.stmt_log(?e)) {
            auto ee = fold_expr(env_, fld, e);
            ret fld.fold_stmt_log(env_, s.span, e);
        }

        case (ast.stmt_expr(?e)) {
            auto ee = fold_expr(env_, fld, e);
            ret fld.fold_stmt_expr(env_, s.span, e);
        }
    }
    ret s;
}

fn fold_block[ENV](&ENV env, ast_fold[ENV] fld, &block blk) -> block {
    let operator[@stmt, @stmt] fs = bind fold_stmt[ENV](env, fld, _);
    auto stmts = _vec.map[@stmt, @stmt](fs, blk.node);
    ret respan(blk.span, stmts);
}

fn fold_slot[ENV](&ENV env, ast_fold[ENV] fld, &slot s) -> slot {
    auto ty = fold_ty[ENV](env, fld, s.ty);
    ret rec(ty=ty, mode=s.mode, id=s.id);
}


fn fold_fn[ENV](&ENV env, ast_fold[ENV] fld, &ast._fn f) -> ast._fn {

    fn fold_input[ENV](&ENV env, ast_fold[ENV] fld, &ast.input i)
        -> ast.input {
        ret rec(slot=fold_slot[ENV](env, fld, i.slot),
                ident=i.ident);
    }

    let operator[ast.input,ast.input] fi = bind fold_input[ENV](env, fld, _);
    auto inputs = _vec.map[ast.input, ast.input](fi, f.inputs);
    auto output = fold_slot[ENV](env, fld, f.output);
    auto body = fold_block[ENV](env, fld, f.body);

    ret fld.fold_fn(env, inputs, output, body);
}

fn fold_item[ENV](&ENV env, ast_fold[ENV] fld, @item i) -> @item {

    let ENV env_ = fld.update_env_for_item(env, i);

    if (!fld.keep_going(env_)) {
        ret i;
    }

    alt (i.node) {

        case (ast.item_fn(?ff, ?id)) {
            let ast._fn ff_ = fold_fn[ENV](env_, fld, ff);
            ret fld.fold_item_fn(env_, i.span, ff_, id);
        }

        case (ast.item_mod(?mm)) {
            let ast._mod mm_ = fold_mod[ENV](env_, fld, mm);
            ret fld.fold_item_mod(env_, i.span, mm_);
        }

        case (ast.item_ty(?ty, ?id)) {
            let @ast.ty ty_ = fold_ty[ENV](env_, fld, ty);
            ret fld.fold_item_ty(env_, i.span, ty_, id);
        }
    }

    fail;
}


fn fold_mod[ENV](&ENV e, ast_fold[ENV] fld, &ast._mod m_in) -> ast._mod {

    auto m_out = new_str_hash[@item]();

    for each (tup(ident, @item) pairs in m_in.items()) {
        auto i = fold_item[ENV](e, fld, pairs._1);
        m_out.insert(pairs._0, i);
    }

    ret fld.fold_mod(e, m_out);
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

fn identity_fold_ty_path[ENV](&ENV env, &span sp, ast.path p,
                        &option[referent] r) -> @ty {
    ret @respan(sp, ast.ty_path(p, r));
}


// Expr identities.

fn identity_fold_expr_vec[ENV](&ENV env, &span sp, vec[@expr] es) -> @expr {
    ret @respan(sp, ast.expr_vec(es));
}

fn identity_fold_expr_tup[ENV](&ENV env, &span sp, vec[@expr] es) -> @expr {
    ret @respan(sp, ast.expr_tup(es));
}

fn identity_fold_expr_rec[ENV](&ENV env, &span sp,
                               vec[tup(ident,@expr)] fields)
    -> @expr {
    ret @respan(sp, ast.expr_rec(fields));
}

fn identity_fold_expr_call[ENV](&ENV env, &span sp, @expr f,
                                vec[@expr] args) -> @expr {
    ret @respan(sp, ast.expr_call(f, args));
}

fn identity_fold_expr_binary[ENV](&ENV env, &span sp, ast.binop b,
                                  @expr lhs, @expr rhs) -> @expr {
    ret @respan(sp, ast.expr_binary(b, lhs, rhs));
}

fn identity_fold_expr_unary[ENV](&ENV env, &span sp,
                                 ast.unop u, @expr e) -> @expr {
    ret @respan(sp, ast.expr_unary(u, e));
}

fn identity_fold_expr_lit[ENV](&ENV env, &span sp, @ast.lit lit) -> @expr {
    ret @respan(sp, ast.expr_lit(lit));
}

fn identity_fold_expr_name[ENV](&ENV env, &span sp, &name n,
                          &option[referent] r) -> @expr {
    ret @respan(sp, ast.expr_name(n, r));
}

fn identity_fold_expr_field[ENV](&ENV env, &span sp,
                                 @expr e, ident i) -> @expr {
    ret @respan(sp, ast.expr_field(e, i));
}

fn identity_fold_expr_index[ENV](&ENV env, &span sp,
                                 @expr e, @expr ix) -> @expr {
    ret @respan(sp, ast.expr_index(e, ix));
}

fn identity_fold_expr_if[ENV](&ENV env, &span sp,
                              @expr cond, block thn,
                              &option[block] els) -> @expr {
    ret @respan(sp, ast.expr_if(cond, thn, els));
}

fn identity_fold_expr_block[ENV](&ENV env, &span sp, block blk) -> @expr {
    ret @respan(sp, ast.expr_block(blk));
}


// Decl identities.

fn identity_fold_decl_local[ENV](&ENV e, &span sp,
                                 ident i, bool infer,
                                 &option[@ty] t) -> @decl {
    ret @respan(sp, ast.decl_local(i, infer, t));
}

fn identity_fold_decl_item[ENV](&ENV e, &span sp,
                                &name n, @item i) -> @decl {
    ret @respan(sp, ast.decl_item(n, i));
}


// Stmt identities.

fn identity_fold_stmt_decl[ENV](&ENV env, &span sp, @decl d) -> @stmt {
    ret @respan(sp, ast.stmt_decl(d));
}

fn identity_fold_stmt_ret[ENV](&ENV env, &span sp,
                               &option[@expr] rv) -> @stmt {
    ret @respan(sp, ast.stmt_ret(rv));
}

fn identity_fold_stmt_log[ENV](&ENV e, &span sp, @expr x) -> @stmt {
    ret @respan(sp, ast.stmt_log(x));
}

fn identity_fold_stmt_expr[ENV](&ENV e, &span sp, @expr x) -> @stmt {
    ret @respan(sp, ast.stmt_expr(x));
}


// Item identities.

fn identity_fold_item_fn[ENV](&ENV e, &span sp, &ast._fn f,
                              ast.item_id id) -> @item {
    ret @respan(sp, ast.item_fn(f, id));
}

fn identity_fold_item_mod[ENV](&ENV e, &span sp, &ast._mod m) -> @item {
    ret @respan(sp, ast.item_mod(m));
}

fn identity_fold_item_ty[ENV](&ENV e, &span sp, @ty t,
                              ast.item_id id) -> @item {
    ret @respan(sp, ast.item_ty(t, id));
}


// Additional identities.

fn identity_fold_block[ENV](&ENV e, &span sp, vec[@stmt] stmts) -> block {
    ret respan(sp, stmts);
}

fn identity_fold_fn[ENV](&ENV e,
                         vec[ast.input] inputs,
                         &slot output,
                         block body) -> ast._fn {
    ret rec(inputs=inputs, output=output, body=body);
}

fn identity_fold_mod[ENV](&ENV e, &ast._mod m) -> ast._mod {
    ret m;
}

fn identity_fold_crate[ENV](&ENV e, &span sp, &ast._mod m) -> @ast.crate {
    ret @respan(sp, rec(module=m));
}


// Env update identities.

fn identity_update_env_for_crate[ENV](&ENV e, @ast.crate c) -> ENV {
    ret e;
}

fn identity_update_env_for_item[ENV](&ENV e, @item i) -> ENV {
    ret e;
}

fn identity_update_env_for_stmt[ENV](&ENV e, @stmt s) -> ENV {
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
         fold_ty_path    = bind identity_fold_ty_path[ENV](_,_,_,_),

         fold_expr_vec    = bind identity_fold_expr_vec[ENV](_,_,_),
         fold_expr_tup    = bind identity_fold_expr_tup[ENV](_,_,_),
         fold_expr_rec    = bind identity_fold_expr_rec[ENV](_,_,_),
         fold_expr_call   = bind identity_fold_expr_call[ENV](_,_,_,_),
         fold_expr_binary = bind identity_fold_expr_binary[ENV](_,_,_,_,_),
         fold_expr_unary  = bind identity_fold_expr_unary[ENV](_,_,_,_),
         fold_expr_lit    = bind identity_fold_expr_lit[ENV](_,_,_),
         fold_expr_name   = bind identity_fold_expr_name[ENV](_,_,_,_),
         fold_expr_field  = bind identity_fold_expr_field[ENV](_,_,_,_),
         fold_expr_index  = bind identity_fold_expr_index[ENV](_,_,_,_),
         fold_expr_if     = bind identity_fold_expr_if[ENV](_,_,_,_,_),
         fold_expr_block  = bind identity_fold_expr_block[ENV](_,_,_),

         fold_decl_local  = bind identity_fold_decl_local[ENV](_,_,_,_,_),
         fold_decl_item   = bind identity_fold_decl_item[ENV](_,_,_,_),

         fold_stmt_decl   = bind identity_fold_stmt_decl[ENV](_,_,_),
         fold_stmt_ret    = bind identity_fold_stmt_ret[ENV](_,_,_),
         fold_stmt_log    = bind identity_fold_stmt_log[ENV](_,_,_),
         fold_stmt_expr   = bind identity_fold_stmt_expr[ENV](_,_,_),

         fold_item_fn   = bind identity_fold_item_fn[ENV](_,_,_,_),
         fold_item_mod  = bind identity_fold_item_mod[ENV](_,_,_),
         fold_item_ty   = bind identity_fold_item_ty[ENV](_,_,_,_),

         fold_block = bind identity_fold_block[ENV](_,_,_),
         fold_fn = bind identity_fold_fn[ENV](_,_,_,_),
         fold_mod = bind identity_fold_mod[ENV](_,_),
         fold_crate = bind identity_fold_crate[ENV](_,_,_),

         update_env_for_crate = bind identity_update_env_for_crate[ENV](_,_),
         update_env_for_item = bind identity_update_env_for_item[ENV](_,_),
         update_env_for_stmt = bind identity_update_env_for_stmt[ENV](_,_),
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

import std.map.hashmap;
import front.ast;
import util.common.new_str_hash;
import util.common.spanned;
import util.common.span;
import util.common.option;
import util.common.some;
import util.common.none;
import util.common.ty_mach;
import std._vec;
import std.util.operator;

type slot[TY] = rec(TY ty, ast.mode mode, option[ast.slot_id] id);
type input[T] = rec(slot[T] slot, ast.ident ident);
type name[TY] = rec(ast.ident ident, vec[TY] types);

type ast_fold[ENV,
              NAME,TY,EXPR,STMT,BLOCK,
              FN,MOD,DECL,ITEM,CRATE] =
    @rec
    (
     // Name fold.
     (fn(&ENV e, &span sp, &name[TY] name) -> NAME) fold_name,

     // Type folds.
     (fn(&ENV e, &span sp) -> TY)               fold_ty_nil,
     (fn(&ENV e, &span sp) -> TY)               fold_ty_bool,
     (fn(&ENV e, &span sp) -> TY)               fold_ty_int,
     (fn(&ENV e, &span sp) -> TY)               fold_ty_uint,
     (fn(&ENV e, &span sp, ty_mach tm) -> TY)   fold_ty_machine,
     (fn(&ENV e, &span sp) -> TY)               fold_ty_char,
     (fn(&ENV e, &span sp) -> TY)               fold_ty_str,
     (fn(&ENV e, &span sp, &TY t) -> TY)        fold_ty_box,
     (fn(&ENV e, &span sp, &ast.path p,
         &option[ast.referent] r) -> TY)        fold_ty_path,

     // Expr folds.
     (fn(&ENV e, &span sp,
         &vec[EXPR] es) -> EXPR)                      fold_expr_vec,

     (fn(&ENV e, &span sp,
         &vec[EXPR] es) -> EXPR)                      fold_expr_tup,

     (fn(&ENV e, &span sp,
         &vec[tup(ast.ident,EXPR)] fields) -> EXPR)   fold_expr_rec,

     (fn(&ENV e, &span sp,
         &EXPR f, &vec[EXPR] args) -> EXPR)           fold_expr_call,

     (fn(&ENV e, &span sp,
         ast.binop,
         &EXPR lhs, &EXPR rhs) -> EXPR)               fold_expr_binary,

     (fn(&ENV e, &span sp,
         ast.unop, &EXPR e) -> EXPR)                  fold_expr_unary,

     (fn(&ENV e, &span sp,
         @ast.lit) -> EXPR)                           fold_expr_lit,

     (fn(&ENV e, &span sp,
         &NAME name,
         &option[ast.referent] r) -> EXPR)            fold_expr_name,

     (fn(&ENV e, &span sp,
         &EXPR e, &ast.ident i) -> EXPR)              fold_expr_field,

     (fn(&ENV e, &span sp,
         &EXPR e, &EXPR ix) -> EXPR)                  fold_expr_index,

     (fn(&ENV e, &span sp,
         &EXPR cond, &BLOCK thn,
         &option[BLOCK] els) -> EXPR)                 fold_expr_if,

     (fn(&ENV e, &span sp,
         &BLOCK blk) -> EXPR)                         fold_expr_block,


     // Decl folds.
     (fn(&ENV e, &span sp,
         &ast.ident ident, bool infer,
         &option[TY] ty) -> DECL)                 fold_decl_local,

     (fn(&ENV e, &span sp,
         &NAME name, ITEM item) -> DECL)          fold_decl_item,


     // Stmt folds.
     (fn(&ENV e, &span sp, &DECL decl) -> STMT) fold_stmt_decl,
     (fn(&ENV e, &span sp, &option[EXPR] rv) -> STMT) fold_stmt_ret,
     (fn(&ENV e, &span sp, &EXPR e) -> STMT) fold_stmt_log,
     (fn(&ENV e, &span sp, &EXPR e) -> STMT) fold_stmt_expr,

     // Item folds.
     (fn(&ENV e, &span sp, &FN f, ast.item_id id) -> ITEM) fold_item_fn,
     (fn(&ENV e, &span sp, &MOD m) -> ITEM) fold_item_mod,
     (fn(&ENV e, &span sp, &TY t, ast.item_id id) -> ITEM) fold_item_ty,

     // Additional nodes.
     (fn(&ENV e, &span sp, &vec[STMT] stmts) -> BLOCK) fold_block,
     (fn(&ENV e, &vec[rec(slot[TY] slot, ast.ident ident)] inputs,
         &slot[TY] output, &BLOCK body) -> FN) fold_fn,
     (fn(&ENV e, hashmap[ast.ident,ITEM] m) -> MOD) fold_mod,
     (fn(&ENV e, &span sp, &MOD m) -> CRATE) fold_crate,

     // Env updates.
     (fn(&ENV e, &ast.crate c) -> ENV) update_env_for_crate,
     (fn(&ENV e, &ast.item i) -> ENV) update_env_for_item,
     (fn(&ENV e, &ast.stmt s) -> ENV) update_env_for_stmt,
     (fn(&ENV e, &ast.expr x) -> ENV) update_env_for_expr,
     (fn(&ENV e, &ast.ty t) -> ENV) update_env_for_ty,

     // Traversal control.
     (fn(&ENV v) -> bool) keep_going
      );


//// Fold drivers.

// FIXME: Finish these.

// FIXME: Also, little more type-inference love would help here.

fn fold_ty[E,N,T,X,S,B,F,M,D,I,C]
(&E env, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld, @ast.ty ty) -> T {
    fail;
}

fn fold_block[E,N,T,X,S,B,F,M,D,I,C]
(&E env, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld, &ast.block blk) -> B {
    fail;
}

fn fold_slot[E,N,T,X,S,B,F,M,D,I,C]
(&E env, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld, &ast.slot s) -> slot[T] {
    auto ty = fold_ty[E,N,T,X,S,B,F,M,D,I,C](env, fld, s.ty);
    ret rec(ty=ty, mode=s.mode, id=s.id);
}


fn fold_fn[E,N,T,X,S,B,F,M,D,I,C]
(&E env, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld, &ast._fn f) -> F {

    fn fold_input[E,N,T,X,S,B,F,M,D,I,C]
        (&E env, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld,
         &rec(ast.slot slot, ast.ident ident) i)
        -> input[T] {
        ret rec(slot=fold_slot[E,N,T,X,S,B,F,M,D,I,C](env, fld, i.slot),
                ident=i.ident);
    }

    let operator[ast.input,input[T]] fi =
        bind fold_input[E,N,T,X,S,B,F,M,D,I,C](env, fld, _);
    auto inputs = _vec.map[ast.input, input[T]](fi, f.inputs);
    auto output = fold_slot[E,N,T,X,S,B,F,M,D,I,C](env, fld, f.output);
    auto body = fold_block[E,N,T,X,S,B,F,M,D,I,C](env, fld, f.body);

    ret fld.fold_fn(env, inputs, output, body);
}

fn fold_item[E,N,T,X,S,B,F,M,D,I,C]
(&E env, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld, @ast.item item) -> I {

    let E env_ = fld.update_env_for_item(env, *item);

    alt (item.node) {

        case (ast.item_fn(?ff, ?id)) {
            let F ff_ = fold_fn[E,N,T,X,S,B,F,M,D,I,C](env_, fld, ff);
            ret fld.fold_item_fn(env_, item.span, ff_, id);
        }

        case (ast.item_mod(?mm)) {
            let M mm_ = fold_mod[E,N,T,X,S,B,F,M,D,I,C](env_, fld, mm);
            ret fld.fold_item_mod(env_, item.span, mm_);
        }

        case (ast.item_ty(?ty, ?id)) {
            let T ty_ = fold_ty[E,N,T,X,S,B,F,M,D,I,C](env_, fld, ty);
            ret fld.fold_item_ty(env_, item.span, ty_, id);
        }
    }

    fail;
}


fn fold_mod[E,N,T,X,S,B,F,M,D,I,C]
(&E e, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld, &ast._mod m_in) -> M {

    auto m_out = new_str_hash[I]();

    for each (tup(ast.ident, @ast.item) pairs in m_in.items()) {
        auto i = fold_item[E,N,T,X,S,B,F,M,D,I,C](e, fld, pairs._1);
        m_out.insert(pairs._0, i);
    }

    ret fld.fold_mod(e, m_out);
 }

fn fold_crate[E,N,T,X,S,B,F,M,D,I,C]
(&E env, ast_fold[E,N,T,X,S,B,F,M,D,I,C] fld, @ast.crate c) -> C {
    let E env_ = fld.update_env_for_crate(env, *c);
    let M m = fold_mod[E,N,T,X,S,B,F,M,D,I,C](env_, fld, c.node.module);
    ret fld.fold_crate(env_, c.span, m);
}

//// Identity folds.

fn respan[T](&span sp, &T t) -> spanned[T] {
    ret rec(node=t, span=sp);
}


// Name identity.

fn identity_fold_name[ENV](&ENV env, &span sp,
                           &ast.name_ n) -> ast.name {
    ret respan(sp, n);
}


// Type identities.

fn identity_fold_ty_nil[ENV](&ENV env, &span sp) -> @ast.ty {
    ret @respan(sp, ast.ty_nil);
}

fn identity_fold_ty_bool[ENV](&ENV env, &span sp) -> @ast.ty {
    ret @respan(sp, ast.ty_bool);
}

fn identity_fold_ty_int[ENV](&ENV env, &span sp) -> @ast.ty {
    ret @respan(sp, ast.ty_int);
}

fn identity_fold_ty_uint[ENV](&ENV env, &span sp) -> @ast.ty {
    ret @respan(sp, ast.ty_uint);
}

fn identity_fold_ty_machine[ENV](&ENV env, &span sp,
                                 ty_mach tm) -> @ast.ty {
    ret @respan(sp, ast.ty_machine(tm));
}

fn identity_fold_ty_char[ENV](&ENV env, &span sp) -> @ast.ty {
    ret @respan(sp, ast.ty_char);
}

fn identity_fold_ty_str[ENV](&ENV env, &span sp) -> @ast.ty {
    ret @respan(sp, ast.ty_str);
}

fn identity_fold_ty_box[ENV](&ENV env, &span sp, &@ast.ty t) -> @ast.ty {
    ret @respan(sp, ast.ty_box(t));
}

fn identity_fold_ty_path[ENV](&ENV env, &span sp, &ast.path p,
                        &option[ast.referent] r) -> @ast.ty {
    ret @respan(sp, ast.ty_path(p, r));
}


// Expr identities.

fn identity_fold_expr_vec[ENV](&ENV env, &span sp,
                               &vec[@ast.expr] es) -> @ast.expr {
    ret @respan(sp, ast.expr_vec(es));
}

fn identity_fold_expr_tup[ENV](&ENV env, &span sp,
                               &vec[@ast.expr] es) -> @ast.expr {
    ret @respan(sp, ast.expr_tup(es));
}

fn identity_fold_expr_rec[ENV](&ENV env, &span sp,
                               &vec[tup(ast.ident,@ast.expr)] fields)
    -> @ast.expr {
    ret @respan(sp, ast.expr_rec(fields));
}

fn identity_fold_expr_call[ENV](&ENV env, &span sp, &@ast.expr f,
                                &vec[@ast.expr] args) -> @ast.expr {
    ret @respan(sp, ast.expr_call(f, args));
}

fn identity_fold_expr_binary[ENV](&ENV env, &span sp, ast.binop b,
                                  &@ast.expr lhs,
                                  &@ast.expr rhs) -> @ast.expr {
    ret @respan(sp, ast.expr_binary(b, lhs, rhs));
}

fn identity_fold_expr_unary[ENV](&ENV env, &span sp,
                                 ast.unop u, &@ast.expr e) -> @ast.expr {
    ret @respan(sp, ast.expr_unary(u, e));
}

fn identity_fold_expr_lit[ENV](&ENV env, &span sp,
                               @ast.lit lit) -> @ast.expr {
    ret @respan(sp, ast.expr_lit(lit));
}

fn identity_fold_expr_name[ENV](&ENV env, &span sp, &ast.name name,
                          &option[ast.referent] r) -> @ast.expr {
    ret @respan(sp, ast.expr_name(name, r));
}

fn identity_fold_expr_field[ENV](&ENV env, &span sp,
                                 &@ast.expr e, &ast.ident i)
    -> @ast.expr {
    ret @respan(sp, ast.expr_field(e, i));
}

fn identity_fold_expr_index[ENV](&ENV env, &span sp,
                                 &@ast.expr e, &@ast.expr ix)
    -> @ast.expr {
    ret @respan(sp, ast.expr_index(e, ix));
}

fn identity_fold_expr_if[ENV](&ENV env, &span sp,
                              &@ast.expr cond, &ast.block thn,
                              &option[ast.block] els) -> @ast.expr {
    ret @respan(sp, ast.expr_if(cond, thn, els));
}

fn identity_fold_expr_block[ENV](&ENV env, &span sp,
                                 &ast.block blk) -> @ast.expr {
    ret @respan(sp, ast.expr_block(blk));
}


// Decl identities.

fn identity_fold_decl_local[ENV](&ENV e, &span sp,
                                 &ast.ident ident, bool infer,
                                 &option[@ast.ty] ty) -> @ast.decl {
    ret @respan(sp, ast.decl_local(ident, infer, ty));
}

fn identity_fold_decl_item[ENV](&ENV e, &span sp,
                                &ast.name name,
                                @ast.item item) -> @ast.decl {
    ret @respan(sp, ast.decl_item(name, item));
}


// Stmt identities.

fn identity_fold_stmt_decl[ENV](&ENV env, &span sp,
                                &@ast.decl decl) -> @ast.stmt {
    ret @respan(sp, ast.stmt_decl(decl));
}

fn identity_fold_stmt_ret[ENV](&ENV env, &span sp,
                               &option[@ast.expr] rv) -> @ast.stmt {
    ret @respan(sp, ast.stmt_ret(rv));
}

fn identity_fold_stmt_log[ENV](&ENV e, &span sp, &@ast.expr x) -> @ast.stmt {
    ret @respan(sp, ast.stmt_log(x));
}

fn identity_fold_stmt_expr[ENV](&ENV e, &span sp, &@ast.expr x) -> @ast.stmt {
    ret @respan(sp, ast.stmt_expr(x));
}


// Item identities.

fn identity_fold_item_fn[ENV](&ENV e, &span sp, &ast._fn f,
                              ast.item_id id) -> @ast.item {
    ret @respan(sp, ast.item_fn(f, id));
}

fn identity_fold_item_mod[ENV](&ENV e, &span sp, &ast._mod m) -> @ast.item {
    ret @respan(sp, ast.item_mod(m));
}

fn identity_fold_item_ty[ENV](&ENV e, &span sp, &@ast.ty t,
                              ast.item_id id) -> @ast.item {
    ret @respan(sp, ast.item_ty(t, id));
}


// Additional identities.

fn identity_fold_block[ENV](&ENV e, &span sp,
                            &vec[@ast.stmt] stmts) -> ast.block {
    ret respan(sp, stmts);
}

fn identity_fold_fn[ENV](&ENV e,
                         &vec[rec(ast.slot slot, ast.ident ident)] inputs,
                         &ast.slot output,
                         &ast.block body) -> ast._fn {
    ret rec(inputs=inputs, output=output, body=body);
}

fn identity_fold_mod[ENV](&ENV e,
                          hashmap[ast.ident, @ast.item] m) -> ast._mod {
    ret m;
}

fn identity_fold_crate[ENV](&ENV e, &span sp,
                            &hashmap[ast.ident, @ast.item] m) -> @ast.crate {
    ret @respan(sp, rec(module=m));
}


// Env update identities.

fn identity_update_env_for_crate[ENV](&ENV e, &ast.crate c) -> ENV {
    ret e;
}

fn identity_update_env_for_item[ENV](&ENV e, &ast.item i) -> ENV {
    ret e;
}

fn identity_update_env_for_stmt[ENV](&ENV e, &ast.stmt s) -> ENV {
    ret e;
}

fn identity_update_env_for_expr[ENV](&ENV e, &ast.expr x) -> ENV {
    ret e;
}

fn identity_update_env_for_ty[ENV](&ENV e, &ast.ty t) -> ENV {
    ret e;
}


// Always-true traversal control fn.

fn always_keep_going[ENV](&ENV e) -> bool {
    ret true;
}


type identity_fold[ENV] = ast_fold[ENV,
                                   ast.name, @ast.ty, @ast.expr,
                                   @ast.stmt, ast.block, ast._fn,
                                   ast._mod, @ast.decl, @ast.item,
                                   @ast.crate];

type query_fold[ENV,T] = ast_fold[ENV,
                                  T,T,T,T,T,
                                  T,T,T,T,T];

fn new_identity_fold[ENV]() -> identity_fold[ENV] {
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

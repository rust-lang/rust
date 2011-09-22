import syntax::ast;
import lib::llvm::llvm::ValueRef;
import trans_common::*;
import trans_build::*;
import trans::{
    trans_shared_malloc,
    type_of_or_i8,
    size_of,
    move_val_if_temp,
    node_id_type,
    trans_lval,
    INIT,
    trans_shared_free,
    drop_ty,
    new_sub_block_ctxt
};

export trans_uniq, make_free_glue;

fn trans_uniq(cx: @block_ctxt, contents: @ast::expr,
              node_id: ast::node_id) -> result {
    let bcx = cx;

    let lv = trans_lval(bcx, contents);
    bcx = lv.bcx;

    let uniq_ty = node_id_type(bcx_ccx(cx), node_id);
    assert ty::type_is_unique_box(bcx_tcx(cx), uniq_ty);
    let {bcx, val: llptr} = alloc_uniq(bcx, uniq_ty);

    bcx = move_val_if_temp(bcx, INIT, llptr, lv,
                           content_ty(bcx, uniq_ty));

    ret rslt(bcx, llptr);
}

fn alloc_uniq(cx: @block_ctxt, uniq_ty: ty::t) -> result {
    assert ty::type_is_unique_box(bcx_tcx(cx), uniq_ty);

    let bcx = cx;
    let contents_ty = content_ty(bcx, uniq_ty);
    let r = size_of(bcx, contents_ty);
    bcx = r.bcx;
    let llsz = r.val;

    let llptrty = T_ptr(type_of_or_i8(bcx, contents_ty));

    r = trans_shared_malloc(bcx, llptrty, llsz);
    bcx = r.bcx;
    let llptr = r.val;

    add_clean_temp(bcx, llptr, uniq_ty);

    ret rslt(bcx, llptr);
}

fn make_free_glue(bcx: @block_ctxt, v: ValueRef, t: ty::t) -> @block_ctxt {
    assert ty::type_is_unique_box(bcx_tcx(bcx), t);

    let free_cx = new_sub_block_ctxt(bcx, "uniq_free");
    let next_cx = new_sub_block_ctxt(bcx, "uniq_free_next");
    let vptr = Load(bcx, v);
    let null_test = IsNull(bcx, vptr);
    CondBr(bcx, null_test, next_cx.llbb, free_cx.llbb);

    let bcx = free_cx;
    let bcx = drop_ty(bcx, vptr, content_ty(bcx, t));
    let bcx = trans_shared_free(bcx, vptr);
    Store(bcx, C_null(val_ty(vptr)), v);
    Br(bcx, next_cx.llbb);

    next_cx
}

fn content_ty(bcx: @block_ctxt, t: ty::t) -> ty::t {
    assert ty::type_is_unique_box(bcx_tcx(bcx), t);

    alt ty::struct(bcx_tcx(bcx), t) {
      ty::ty_uniq({ty: ct, _}) { ct }
    }
}
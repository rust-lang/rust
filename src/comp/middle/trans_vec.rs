import std::vec;
import std::option::none;
import syntax::ast;
import lib::llvm::llvm::{ValueRef, TypeRef};
import back::abi;
import trans::{call_memmove, trans_shared_malloc, llsize_of,
               type_of_or_i8, incr_ptr, INIT, copy_val, load_if_immediate,
               alloca, size_of, llderivedtydescs_block_ctxt,
               lazily_emit_tydesc_glue, get_tydesc, load_inbounds,
               move_val_if_temp, trans_lval, node_id_type,
               new_sub_block_ctxt, tps_normal, do_spill};
import trans_build::*;
import trans_common::*;

fn get_fill(bcx: &@block_ctxt, vptr: ValueRef) -> ValueRef {
    Load(bcx, InBoundsGEP(bcx, vptr, [C_int(0), C_uint(abi::vec_elt_fill)]))
}
fn get_alloc(bcx: &@block_ctxt, vptr: ValueRef) -> ValueRef {
    Load(bcx, InBoundsGEP(bcx, vptr, [C_int(0), C_uint(abi::vec_elt_alloc)]))
}
fn get_dataptr(bcx: &@block_ctxt, vpt: ValueRef,
               unit_ty: TypeRef) -> ValueRef {
    let ptr = InBoundsGEP(bcx, vpt, [C_int(0), C_uint(abi::vec_elt_elems)]);
    PointerCast(bcx, ptr, T_ptr(unit_ty))
}

fn pointer_add(bcx: &@block_ctxt, ptr: ValueRef, bytes: ValueRef)
    -> ValueRef {
    let old_ty = val_ty(ptr);
    let bptr = PointerCast(bcx, ptr, T_ptr(T_i8()));
    ret PointerCast(bcx, InBoundsGEP(bcx, bptr, [bytes]), old_ty);
}

fn alloc_raw(bcx: &@block_ctxt, fill: ValueRef, alloc: ValueRef) -> result {
    let llvecty = T_opaque_vec();
    let vecsize = Add(bcx, alloc, llsize_of(llvecty));
    let {bcx, val: vecptr} =
        trans_shared_malloc(bcx, T_ptr(llvecty), vecsize);
    Store(bcx, fill, InBoundsGEP
          (bcx, vecptr, [C_int(0), C_uint(abi::vec_elt_fill)]));
    Store(bcx, alloc, InBoundsGEP
          (bcx, vecptr, [C_int(0), C_uint(abi::vec_elt_alloc)]));
    ret {bcx: bcx, val: vecptr};
}

type alloc_result = {bcx: @block_ctxt,
                     val: ValueRef,
                     unit_ty: ty::t,
                     llunitsz: ValueRef,
                     llunitty: TypeRef};

fn alloc(bcx: &@block_ctxt, vec_ty: &ty::t, elts: uint) -> alloc_result {
    let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let llunitty = type_of_or_i8(bcx, unit_ty);
    let llvecty = T_vec(llunitty);
    let {bcx, val: unit_sz} = size_of(bcx, unit_ty);

    let fill = Mul(bcx, C_uint(elts), unit_sz);
    let alloc = if elts < 4u { Mul(bcx, C_int(4), unit_sz) } else { fill };
    let {bcx, val: vptr} = alloc_raw(bcx, fill, alloc);
    let vptr = PointerCast(bcx, vptr, T_ptr(llvecty));
    add_clean_temp(bcx, vptr, vec_ty);
    ret {bcx: bcx, val: vptr, unit_ty: unit_ty,
         llunitsz: unit_sz, llunitty: llunitty};
}

fn duplicate(bcx: &@block_ctxt, vptrptr: ValueRef) -> @block_ctxt {
    let vptr = Load(bcx, vptrptr);
    let fill = get_fill(bcx, vptr);
    let size = Add(bcx, fill, llsize_of(T_opaque_vec()));
    let {bcx, val: newptr} = trans_shared_malloc(bcx, val_ty(vptr), size);
    let bcx = call_memmove(bcx, newptr, vptr, size).bcx;
    Store(bcx, fill,
          InBoundsGEP(bcx, newptr, [C_int(0), C_uint(abi::vec_elt_alloc)]));
    Store(bcx, newptr, vptrptr);
    ret bcx;
}
fn make_drop_glue(bcx: &@block_ctxt, vptrptr: ValueRef, vec_ty: ty::t)
    -> @block_ctxt {
    let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let vptr = Load(bcx, vptrptr);
    let drop_cx = new_sub_block_ctxt(bcx, ~"drop");
    let next_cx = new_sub_block_ctxt(bcx, ~"next");
    let null_test = IsNull(bcx, vptr);
    CondBr(bcx, null_test, next_cx.llbb, drop_cx.llbb);
    if ty::type_needs_drop(bcx_tcx(bcx), unit_ty) {
        drop_cx = iter_vec(drop_cx, vptrptr, vec_ty, trans::drop_ty);
    }
    drop_cx = trans::trans_shared_free(drop_cx, vptr);
    Br(drop_cx, next_cx.llbb);
    ret next_cx;
}

fn trans_vec(bcx: &@block_ctxt, args: &[@ast::expr],
              id: ast::node_id) -> result {
    let vec_ty = node_id_type(bcx_ccx(bcx), id);
    let {bcx, val: vptr, llunitsz, unit_ty, llunitty} =
        alloc(bcx, vec_ty, vec::len(args));

    // Store the individual elements.
    let dataptr = get_dataptr(bcx, vptr, llunitty);
    let i = 0u;
    for e in args {
        let lv = trans_lval(bcx, e);
        bcx = lv.res.bcx;
        let lleltptr = if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
            InBoundsGEP(bcx, dataptr, [Mul(bcx, C_uint(i), llunitsz)])
        } else {
            InBoundsGEP(bcx, dataptr, [C_uint(i)])
        };
        bcx = move_val_if_temp(bcx, INIT, lleltptr, lv, unit_ty);
        i += 1u;
    }
    ret rslt(bcx, vptr);
}
fn trans_istr(bcx: &@block_ctxt, s: istr) -> result {
    let veclen = std::str::byte_len(s) + 1u; // +1 for \0
    let {bcx, val: sptr, _} =
        alloc(bcx, ty::mk_istr(bcx_tcx(bcx)), veclen);

    let llcstr = C_cstr(bcx_ccx(bcx), s);
    let bcx = call_memmove(bcx, get_dataptr(bcx, sptr, T_i8()),
                           llcstr, C_uint(veclen)).bcx;

    ret rslt(bcx, sptr);
}

fn trans_append(cx: &@block_ctxt, vec_ty: ty::t, lhsptr: ValueRef,
                rhs: ValueRef) -> result {
    // Cast to opaque interior vector types if necessary.
    let unit_ty = ty::sequence_element_type(bcx_tcx(cx), vec_ty);
    let dynamic = ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty);
    if dynamic {
        lhsptr = PointerCast(cx, lhsptr, T_ptr(T_ptr(T_opaque_vec())));
        rhs = PointerCast(cx, rhs, T_ptr(T_opaque_vec()));
    }
    let strings = alt ty::struct(bcx_tcx(cx), vec_ty) {
      ty::ty_istr. { true }
      ty::ty_vec(_) { false }
    };

    let {bcx, val: unit_sz} = size_of(cx, unit_ty);
    let llunitty = type_of_or_i8(cx, unit_ty);

    let lhs = Load(bcx, lhsptr);
    let self_append = ICmp(bcx, lib::llvm::LLVMIntEQ, lhs, rhs);
    let lfill = get_fill(bcx, lhs);
    let rfill = get_fill(bcx, rhs);
    let new_fill = Add(bcx, lfill, rfill);
    if strings { new_fill = Sub(bcx, new_fill, C_int(1)); }
    let opaque_lhs = PointerCast(bcx, lhsptr, T_ptr(T_ptr(T_opaque_vec())));
    Call(bcx, bcx_ccx(cx).upcalls.vec_grow,
         [cx.fcx.lltaskptr, opaque_lhs, new_fill]);
    // Was overwritten if we resized
    let lhs = Load(bcx, lhsptr);
    let rhs = Select(bcx, self_append, lhs, rhs);

    let lhs_data = get_dataptr(bcx, lhs, llunitty);
    let lhs_off = lfill;
    if strings { lhs_off = Sub(bcx, lhs_off, C_int(1)); }
    let write_ptr = pointer_add(bcx, lhs_data, lhs_off);
    let write_ptr_ptr = do_spill(bcx, write_ptr);
    let bcx = iter_vec_raw(bcx, rhs, vec_ty, rfill, { | &bcx, addr, _ty |
        let write_ptr = Load(bcx, write_ptr_ptr);
        let bcx = copy_val(bcx, INIT, write_ptr,
                           load_if_immediate(bcx, addr, unit_ty), unit_ty);
        if dynamic {
            // We have to increment by the dynamically-computed size.
            incr_ptr(bcx, write_ptr, unit_sz, write_ptr_ptr);
        } else {
            incr_ptr(bcx, write_ptr, C_int(1), write_ptr_ptr);
        }
        ret bcx;
    });
    ret rslt(bcx, C_nil());
}

fn trans_append_literal(bcx: &@block_ctxt, vptrptr: ValueRef, vec_ty: ty::t,
                        vals: &[@ast::expr]) -> @block_ctxt {
    let elt_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let ti = none;
    let {bcx, val: td} =
        get_tydesc(bcx, elt_ty, false, tps_normal, ti).result;
    trans::lazily_emit_tydesc_glue(bcx, abi::tydesc_field_take_glue, ti);
    let opaque_v = PointerCast(bcx, vptrptr, T_ptr(T_ptr(T_opaque_vec())));
    for val in vals {
        let {bcx: e_bcx, val: elt} = trans::trans_expr(bcx, val);
        bcx = e_bcx;
        let spilled = trans::spill_if_immediate(bcx, elt, elt_ty);
        Call(bcx, bcx_ccx(bcx).upcalls.vec_push,
             [bcx.fcx.lltaskptr, opaque_v, td,
              PointerCast(bcx, spilled, T_ptr(T_i8()))]);
    }
    ret bcx;
}

fn trans_add(bcx: &@block_ctxt, vec_ty: ty::t, lhs: ValueRef,
             rhs: ValueRef) -> result {
    let strings = alt ty::struct(bcx_tcx(bcx), vec_ty) {
      ty::ty_istr. { true }
      ty::ty_vec(_) { false }
    };
    let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let llunitty = type_of_or_i8(bcx, unit_ty);
    let {bcx, val: llunitsz} = size_of(bcx, unit_ty);

    let lhs_fill = get_fill(bcx, lhs);
    if strings { lhs_fill = Sub(bcx, lhs_fill, C_int(1)); }
    let rhs_fill = get_fill(bcx, rhs);
    let new_fill = Add(bcx, lhs_fill, rhs_fill);
    let {bcx, val: new_vec} = alloc_raw(bcx, new_fill, new_fill);
    let new_vec = PointerCast(bcx, new_vec, T_ptr(T_vec(llunitty)));
    add_clean_temp(bcx, new_vec, vec_ty);

    let write_ptr_ptr = do_spill(bcx, get_dataptr(bcx, new_vec, llunitty));
    let copy_fn = bind fn(bcx: &@block_ctxt, addr: ValueRef, _ty: ty::t,
                          write_ptr_ptr: ValueRef, unit_ty: ty::t,
                          llunitsz: ValueRef) -> @block_ctxt {
        let write_ptr = Load(bcx, write_ptr_ptr);
        let bcx = copy_val(bcx, INIT, write_ptr,
                           load_if_immediate(bcx, addr, unit_ty), unit_ty);
        if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
            // We have to increment by the dynamically-computed size.
            incr_ptr(bcx, write_ptr, llunitsz, write_ptr_ptr);
        } else {
            incr_ptr(bcx, write_ptr, C_int(1), write_ptr_ptr);
        }
        ret bcx;
    } (_, _, _, write_ptr_ptr, unit_ty, llunitsz);

    let bcx = iter_vec_raw(bcx, lhs, vec_ty, lhs_fill, copy_fn);
    let bcx = iter_vec_raw(bcx, rhs, vec_ty, rhs_fill, copy_fn);
    ret rslt(bcx, new_vec);
}

type val_and_ty_fn = fn(&@block_ctxt, ValueRef, ty::t) -> result;

type iter_vec_block = block(&@block_ctxt, ValueRef, ty::t) -> @block_ctxt;

fn iter_vec_raw(bcx: &@block_ctxt, vptr: ValueRef, vec_ty: ty::t,
                 fill: ValueRef, f: &iter_vec_block) -> @block_ctxt {
    let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let llunitty = type_of_or_i8(bcx, unit_ty);
    let {bcx, val: unit_sz} = size_of(bcx, unit_ty);
    let vptr = PointerCast(bcx, vptr, T_ptr(T_vec(llunitty)));
    let data_ptr = get_dataptr(bcx, vptr, llunitty);

    // Calculate the last pointer address we want to handle.
    // TODO: Optimize this when the size of the unit type is statically
    // known to not use pointer casts, which tend to confuse LLVM.
    let data_end_ptr = pointer_add(bcx, data_ptr, fill);
    let data_ptr_ptr = do_spill(bcx, data_ptr);

    // Now perform the iteration.
    let header_cx = new_sub_block_ctxt(bcx, ~"iter_vec_loop_header");
    Br(bcx, header_cx.llbb);
    let data_ptr = Load(header_cx, data_ptr_ptr);
    let not_yet_at_end = ICmp(header_cx, lib::llvm::LLVMIntULT,
                              data_ptr, data_end_ptr);
    let body_cx = new_sub_block_ctxt(bcx, ~"iter_vec_loop_body");
    let next_cx = new_sub_block_ctxt(bcx, ~"iter_vec_next");
    CondBr(header_cx, not_yet_at_end, body_cx.llbb, next_cx.llbb);
    body_cx = f(body_cx, data_ptr, unit_ty);
    let increment = if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
        unit_sz
    } else { C_int(1) };
    incr_ptr(body_cx, data_ptr, increment, data_ptr_ptr);
    Br(body_cx, header_cx.llbb);

    ret next_cx;
}

fn iter_vec(bcx: &@block_ctxt, vptrptr: ValueRef, vec_ty: ty::t,
             f: &iter_vec_block) -> @block_ctxt {
    let vptr = Load(bcx, PointerCast(bcx, vptrptr,
                                     T_ptr(T_ptr(T_opaque_vec()))));
    ret iter_vec_raw(bcx, vptr, vec_ty, get_fill(bcx, vptr), f);
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

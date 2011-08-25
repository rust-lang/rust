import std::vec;
import std::option::none;
import syntax::ast;
import lib::llvm::llvm::{ValueRef, TypeRef};
import back::abi;
import trans::{call_memmove, trans_shared_malloc, llsize_of,
               type_of_or_i8, incr_ptr, INIT, copy_val, load_if_immediate,
               alloca, array_alloca, size_of, llderivedtydescs_block_ctxt,
               lazily_emit_tydesc_glue, get_tydesc, load_inbounds,
               move_val_if_temp, trans_lval, node_id_type,
               new_sub_block_ctxt, tps_normal, do_spill};
import trans_build::*;
import trans_common::*;

fn get_fill(bcx: &@block_ctxt, vptr: ValueRef) -> ValueRef {
    Load(bcx, InBoundsGEP(bcx, vptr, [C_int(0), C_uint(abi::ivec_elt_fill)]))
}
fn get_alloc(bcx: &@block_ctxt, vptr: ValueRef) -> ValueRef {
    Load(bcx, InBoundsGEP(bcx, vptr, [C_int(0), C_uint(abi::ivec_elt_alloc)]))
}
fn get_dataptr(bcx: &@block_ctxt, vpt: ValueRef,
               unit_ty: TypeRef) -> ValueRef {
    let ptr = InBoundsGEP(bcx, vpt, [C_int(0), C_uint(abi::ivec_elt_elems)]);
    PointerCast(bcx, ptr, T_ptr(unit_ty))
}

fn pointer_add(bcx: &@block_ctxt, ptr: ValueRef, bytes: ValueRef)
    -> ValueRef {
    let old_ty = val_ty(ptr);
    let bptr = PointerCast(bcx, ptr, T_ptr(T_i8()));
    ret PointerCast(bcx, InBoundsGEP(bcx, bptr, [bytes]), old_ty);
}

// FIXME factor out a scaling version wrapping a non-scaling version
fn alloc(bcx: &@block_ctxt, vec_ty: &ty::t, vecsz: ValueRef, is_scaled: bool)
    -> {bcx: @block_ctxt,
        val: ValueRef,
        unit_ty: ty::t,
        llunitsz: ValueRef,
        llunitty: TypeRef} {

    let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let llunitty = type_of_or_i8(bcx, unit_ty);
    let llvecty = T_ivec(llunitty);
    let {bcx, val: unit_sz} = size_of(bcx, unit_ty);

    let fill = if is_scaled { vecsz }
               else { Mul(bcx, vecsz, unit_sz) };
    let vecsize = Add(bcx, fill, llsize_of(llvecty));
    let {bcx, val: vecptr} =
        trans_shared_malloc(bcx, T_ptr(llvecty), vecsize);
    add_clean_temp(bcx, vecptr, vec_ty);

    Store(bcx, fill, InBoundsGEP
          (bcx, vecptr, [C_int(0), C_uint(abi::ivec_elt_fill)]));
    Store(bcx, fill, InBoundsGEP
          (bcx, vecptr, [C_int(0), C_uint(abi::ivec_elt_alloc)]));
    ret {bcx: bcx, val: vecptr,
         unit_ty: unit_ty, llunitsz: unit_sz, llunitty: llunitty};
}
fn duplicate(bcx: &@block_ctxt, vptrptr: ValueRef) -> @block_ctxt {
    let vptr = Load(bcx, vptrptr);
    let fill = get_fill(bcx, vptr);
    let size = Add(bcx, fill, llsize_of(T_opaque_ivec()));
    let {bcx, val: newptr} = trans_shared_malloc(bcx, val_ty(vptr), size);
    let bcx = call_memmove(bcx, newptr, vptr, size).bcx;
    Store(bcx, fill,
          InBoundsGEP(bcx, newptr, [C_int(0), C_uint(abi::ivec_elt_alloc)]));
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
        drop_cx = iter_ivec(drop_cx, vptrptr, vec_ty, trans::drop_ty).bcx;
    }
    drop_cx = trans::trans_shared_free(drop_cx, vptr).bcx;
    Br(drop_cx, next_cx.llbb);
    ret next_cx;
}

fn trans_ivec(bcx: &@block_ctxt, args: &[@ast::expr],
              id: ast::node_id) -> result {
    let vec_ty = node_id_type(bcx_ccx(bcx), id);
    let {bcx, val: vptr, llunitsz, unit_ty, llunitty} =
        alloc(bcx, vec_ty, C_uint(vec::len(args)), false);

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
    let veclen = std::istr::byte_len(s) + 1u; // +1 for \0
    let {bcx, val: sptr, _} =
        alloc(bcx, ty::mk_istr(bcx_tcx(bcx)), C_uint(veclen), false);

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
        lhsptr = PointerCast(cx, lhsptr, T_ptr(T_ptr(T_opaque_ivec())));
        rhs = PointerCast(cx, rhs, T_ptr(T_opaque_ivec()));
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
    let opaque_lhs = PointerCast(bcx, lhsptr, T_ptr(T_ptr(T_opaque_ivec())));
    Call(bcx, bcx_ccx(cx).upcalls.ivec_grow,
         [cx.fcx.lltaskptr, opaque_lhs, new_fill]);
    // Was overwritten if we resized
    let lhs = Load(bcx, lhsptr);
    let rhs = Select(bcx, self_append, lhs, rhs);

    let lhs_data = get_dataptr(bcx, lhs, llunitty);
    let lhs_off = lfill;
    if strings { lhs_off = Sub(bcx, lfill, C_int(1)); }
    let write_ptr = pointer_add(bcx, lhs_data, lhs_off);
    let write_ptr_ptr = do_spill(bcx, write_ptr);
    let end_ptr = pointer_add(bcx, write_ptr, rfill);
    let read_ptr_ptr = do_spill(bcx, get_dataptr(bcx, rhs, llunitty));

    let header_cx = new_sub_block_ctxt(bcx, ~"copy_loop_header");
    Br(bcx, header_cx.llbb);
    let write_ptr = Load(header_cx, write_ptr_ptr);
    let not_yet_at_end = ICmp(header_cx, lib::llvm::LLVMIntNE,
                              write_ptr, end_ptr);
    let body_cx = new_sub_block_ctxt(bcx, ~"copy_loop_body");
    let next_cx = new_sub_block_ctxt(bcx, ~"next");
    CondBr(header_cx, not_yet_at_end,
           body_cx.llbb, next_cx.llbb);

    let read_ptr = Load(body_cx, read_ptr_ptr);
    let body_cx = copy_val(body_cx, INIT, write_ptr,
                           load_if_immediate(body_cx, read_ptr, unit_ty),
                           unit_ty);
    // Increment both pointers.
    if dynamic {
        // We have to increment by the dynamically-computed size.
        incr_ptr(body_cx, write_ptr, unit_sz, write_ptr_ptr);
        incr_ptr(body_cx, read_ptr, unit_sz, read_ptr_ptr);
    } else {
        incr_ptr(body_cx, write_ptr, C_int(1), write_ptr_ptr);
        incr_ptr(body_cx, read_ptr, C_int(1), read_ptr_ptr);
    }
    Br(body_cx, header_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn trans_append_literal(bcx: &@block_ctxt, vptrptr: ValueRef, vec_ty: ty::t,
                        vals: &[@ast::expr]) -> @block_ctxt {
    let elt_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let ti = none;
    let {bcx, val: td} =
        get_tydesc(bcx, elt_ty, false, tps_normal, ti).result;
    trans::lazily_emit_tydesc_glue(bcx, abi::tydesc_field_take_glue, ti);
    let opaque_v = PointerCast(bcx, vptrptr, T_ptr(T_ptr(T_opaque_ivec())));
    for val in vals {
        let {bcx: e_bcx, val: elt} = trans::trans_expr(bcx, val);
        bcx = e_bcx;
        let spilled = trans::spill_if_immediate(bcx, elt, elt_ty);
        Call(bcx, bcx_ccx(bcx).upcalls.ivec_push,
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
    let lhs_fill = get_fill(bcx, lhs);
    if strings { lhs_fill = Sub(bcx, lhs_fill, C_int(1)); }
    let rhs_fill = get_fill(bcx, rhs);
    let new_fill = Add(bcx, lhs_fill, rhs_fill);
    let {bcx, val: new_vec, unit_ty, llunitsz, llunitty} =
        alloc(bcx, vec_ty, new_fill, true);

    // Emit the copy loop
    let write_ptr_ptr = do_spill(bcx, get_dataptr(bcx, new_vec, llunitty));
    let lhs_ptr = get_dataptr(bcx, lhs, llunitty);
    let lhs_ptr_ptr = do_spill(bcx, lhs_ptr);
    let lhs_end_ptr = pointer_add(bcx, lhs_ptr, lhs_fill);
    let rhs_ptr = get_dataptr(bcx, rhs, llunitty);
    let rhs_ptr_ptr = do_spill(bcx, rhs_ptr);
    let rhs_end_ptr = pointer_add(bcx, rhs_ptr, rhs_fill);

    // Copy in elements from the LHS.
    let lhs_cx = new_sub_block_ctxt(bcx, ~"lhs_copy_header");
    Br(bcx, lhs_cx.llbb);
    let lhs_ptr = Load(lhs_cx, lhs_ptr_ptr);
    let not_at_end_lhs =
        ICmp(lhs_cx, lib::llvm::LLVMIntNE, lhs_ptr, lhs_end_ptr);
    let lhs_copy_cx = new_sub_block_ctxt(bcx, ~"lhs_copy_body");
    let rhs_cx = new_sub_block_ctxt(bcx, ~"rhs_copy_header");
    CondBr(lhs_cx, not_at_end_lhs, lhs_copy_cx.llbb, rhs_cx.llbb);
    let write_ptr = Load(lhs_copy_cx, write_ptr_ptr);
    lhs_copy_cx =
        copy_val(lhs_copy_cx, INIT, write_ptr,
                 load_if_immediate(lhs_copy_cx, lhs_ptr, unit_ty), unit_ty);
    // Increment both pointers.
    if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
        // We have to increment by the dynamically-computed size.
        incr_ptr(lhs_copy_cx, write_ptr, llunitsz, write_ptr_ptr);
        incr_ptr(lhs_copy_cx, lhs_ptr, llunitsz, lhs_ptr_ptr);
    } else {
        incr_ptr(lhs_copy_cx, write_ptr, C_int(1), write_ptr_ptr);
        incr_ptr(lhs_copy_cx, lhs_ptr, C_int(1), lhs_ptr_ptr);
    }
    Br(lhs_copy_cx, lhs_cx.llbb);

    // Copy in elements from the RHS.
    let rhs_ptr = Load(rhs_cx, rhs_ptr_ptr);
    let not_at_end_rhs =
        ICmp(rhs_cx, lib::llvm::LLVMIntNE, rhs_ptr, rhs_end_ptr);
    let rhs_copy_cx = new_sub_block_ctxt(bcx, ~"rhs_copy_body");
    let next_cx = new_sub_block_ctxt(bcx, ~"next");
    CondBr(rhs_cx, not_at_end_rhs, rhs_copy_cx.llbb, next_cx.llbb);
    let write_ptr = Load(rhs_copy_cx, write_ptr_ptr);
    rhs_copy_cx =
        copy_val(rhs_copy_cx, INIT, write_ptr,
                 load_if_immediate(rhs_copy_cx, rhs_ptr, unit_ty), unit_ty);
    // Increment both pointers.
    if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
        // We have to increment by the dynamically-computed size.
        incr_ptr(rhs_copy_cx, write_ptr, llunitsz, write_ptr_ptr);
        incr_ptr(rhs_copy_cx, rhs_ptr, llunitsz, rhs_ptr_ptr);
    } else {
        incr_ptr(rhs_copy_cx, write_ptr, C_int(1), write_ptr_ptr);
        incr_ptr(rhs_copy_cx, rhs_ptr, C_int(1), rhs_ptr_ptr);
    }
    Br(rhs_copy_cx, rhs_cx.llbb);

    ret rslt(next_cx, new_vec);
}

// FIXME factor out a utility that can be used to create the loops built
// above
fn iter_ivec(bcx: &@block_ctxt, vptrptr: ValueRef, vec_ty: ty::t,
             f: &trans::val_and_ty_fn) -> result {
    let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let llunitty = type_of_or_i8(bcx, unit_ty);
    let {bcx, val: unit_sz} = size_of(bcx, unit_ty);

    let vptr = Load(bcx, PointerCast(bcx, vptrptr,
                                     T_ptr(T_ptr(T_ivec(llunitty)))));
    let fill = get_fill(bcx, vptr);
    let data_ptr = get_dataptr(bcx, vptr, llunitty);

    // Calculate the last pointer address we want to handle.
    // TODO: Optimize this when the size of the unit type is statically
    // known to not use pointer casts, which tend to confuse LLVM.
    let data_end_ptr = pointer_add(bcx, data_ptr, fill);
    let data_ptr_ptr = do_spill(bcx, data_ptr);

    // Now perform the iteration.
    let header_cx = new_sub_block_ctxt(bcx, ~"iter_ivec_loop_header");
    Br(bcx, header_cx.llbb);
    let data_ptr = Load(header_cx, data_ptr_ptr);
    let not_yet_at_end = ICmp(header_cx, lib::llvm::LLVMIntULT,
                              data_ptr, data_end_ptr);
    let body_cx = new_sub_block_ctxt(bcx, ~"iter_ivec_loop_body");
    let next_cx = new_sub_block_ctxt(bcx, ~"iter_ivec_next");
    CondBr(header_cx, not_yet_at_end, body_cx.llbb, next_cx.llbb);
    body_cx = f(body_cx, data_ptr, unit_ty).bcx;
    let increment = if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
        unit_sz
    } else { C_int(1) };
    incr_ptr(body_cx, data_ptr, increment, data_ptr_ptr);
    Br(body_cx, header_cx.llbb);

    ret rslt(next_cx, C_nil());
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

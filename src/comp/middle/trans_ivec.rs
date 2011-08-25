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
               new_sub_block_ctxt, tps_normal};
import bld = trans_build;
import trans_common::*;

fn alloc_with_heap(bcx: @block_ctxt, typ: &ty::t, vecsz: uint) ->
    {bcx: @block_ctxt,
     unit_ty: ty::t,
     llunitsz: ValueRef,
     llptr: ValueRef,
     llfirsteltptr: ValueRef} {

    let unit_ty;
    alt ty::struct(bcx_tcx(bcx), typ) {
      ty::ty_vec(mt) { unit_ty = mt.ty; }
      _ { bcx_ccx(bcx).sess.bug("non-ivec type in trans_ivec"); }
    }
    let llunitty = type_of_or_i8(bcx, unit_ty);

    let ares = alloc(bcx, unit_ty);
    bcx = ares.bcx;
    let llvecptr = ares.llptr;
    let unit_sz = ares.llunitsz;
    let llalen = ares.llalen;

    add_clean_temp(bcx, llvecptr, typ);

    let lllen = bld::Mul(bcx, C_uint(vecsz), unit_sz);
    // Allocate the vector pieces and store length and allocated length.

    let llfirsteltptr;
    if vecsz > 0u && vecsz <= abi::ivec_default_length {
        // Interior case.

        bld::Store(bcx, lllen,
                        bld::InBoundsGEP(bcx, llvecptr,
                                              [C_int(0),
                                               C_uint(abi::ivec_elt_len)]));
        bld::Store(bcx, llalen,
                        bld::InBoundsGEP(bcx, llvecptr,
                                              [C_int(0),
                                               C_uint(abi::ivec_elt_alen)]));
        llfirsteltptr =
            bld::InBoundsGEP(bcx, llvecptr,
                                  [C_int(0), C_uint(abi::ivec_elt_elems),
                                   C_int(0)]);
    } else {
        // Heap case.

        let stub_z = [C_int(0), C_uint(abi::ivec_heap_stub_elt_zero)];
        let stub_a = [C_int(0), C_uint(abi::ivec_heap_stub_elt_alen)];
        let stub_p = [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];
        let llstubty = T_ivec_heap(llunitty);
        let llstubptr = bld::PointerCast(bcx, llvecptr, T_ptr(llstubty));
        bld::Store(bcx, C_int(0), bld::InBoundsGEP(bcx, llstubptr, stub_z));
        let llheapty = T_ivec_heap_part(llunitty);
        if vecsz == 0u {
            // Null heap pointer indicates a zero-length vector.

            bld::Store(bcx, llalen, bld::InBoundsGEP(bcx, llstubptr, stub_a));
            bld::Store(bcx, C_null(T_ptr(llheapty)),
                            bld::InBoundsGEP(bcx, llstubptr, stub_p));
            llfirsteltptr = C_null(T_ptr(llunitty));
        } else {
            bld::Store(bcx, lllen, bld::InBoundsGEP(bcx, llstubptr, stub_a));

            let llheapsz = bld::Add(bcx, llsize_of(llheapty), lllen);
            let rslt = trans_shared_malloc(bcx, T_ptr(llheapty), llheapsz);
            bcx = rslt.bcx;
            let llheapptr = rslt.val;
            bld::Store(bcx, llheapptr,
                            bld::InBoundsGEP(bcx, llstubptr, stub_p));
            let heap_l = [C_int(0), C_uint(abi::ivec_heap_elt_len)];
            bld::Store(bcx, lllen, bld::InBoundsGEP(bcx, llheapptr, heap_l));
            llfirsteltptr =
                bld::InBoundsGEP(bcx, llheapptr,
                                      [C_int(0),
                                       C_uint(abi::ivec_heap_elt_elems),
                                       C_int(0)]);
        }
    }
    ret {
        bcx: bcx,
        unit_ty: unit_ty,
        llunitsz: unit_sz,
        llptr: llvecptr,
        llfirsteltptr: llfirsteltptr};
}

fn trans_ivec(bcx: @block_ctxt, args: &[@ast::expr],
              id: ast::node_id) -> result {

    let typ = node_id_type(bcx_ccx(bcx), id);
    let alloc_res = alloc_with_heap(bcx, typ, vec::len(args));

    let bcx = alloc_res.bcx;
    let unit_ty = alloc_res.unit_ty;
    let llunitsz = alloc_res.llunitsz;
    let llvecptr = alloc_res.llptr;
    let llfirsteltptr = alloc_res.llfirsteltptr;

    // Store the individual elements.
    let i = 0u;
    for e: @ast::expr in args {
        let lv = trans_lval(bcx, e);
        bcx = lv.res.bcx;
        let lleltptr;
        if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
            lleltptr =
                bld::InBoundsGEP(bcx, llfirsteltptr,
                                      [bld::Mul(bcx, C_uint(i), llunitsz)]);
        } else {
            lleltptr = bld::InBoundsGEP(bcx, llfirsteltptr, [C_uint(i)]);
        }
        bcx = move_val_if_temp(bcx, INIT, lleltptr, lv, unit_ty);
        i += 1u;
    }
    ret rslt(bcx, llvecptr);
}

// Returns the length of an interior vector and a pointer to its first
// element, in that order.
fn get_len_and_data(bcx: &@block_ctxt, orig_v: ValueRef, unit_ty: ty::t)
    -> {len: ValueRef, data: ValueRef, bcx: @block_ctxt} {
    // If this interior vector has dynamic size, we can't assume anything
    // about the LLVM type of the value passed in, so we cast it to an
    // opaque vector type.
    let v;
    if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
        v = bld::PointerCast(bcx, orig_v, T_ptr(T_opaque_ivec()));
    } else { v = orig_v; }

    let llunitty = type_of_or_i8(bcx, unit_ty);
    let stack_len =
        load_inbounds(bcx, v, [C_int(0), C_uint(abi::ivec_elt_len)]);
    let stack_elem =
        bld::InBoundsGEP(bcx, v,
                              [C_int(0), C_uint(abi::ivec_elt_elems),
                               C_int(0)]);
    let on_heap =
        bld::ICmp(bcx, lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    let on_heap_cx = new_sub_block_ctxt(bcx, "on_heap");
    let next_cx = new_sub_block_ctxt(bcx, "next");
    bld::CondBr(bcx, on_heap, on_heap_cx.llbb, next_cx.llbb);
    let heap_stub =
        bld::PointerCast(on_heap_cx, v, T_ptr(T_ivec_heap(llunitty)));
    let heap_ptr =
        load_inbounds(on_heap_cx, heap_stub,
                      [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)]);

    // Check whether the heap pointer is null. If it is, the vector length
    // is truly zero.

    let llstubty = T_ivec_heap(llunitty);
    let llheapptrty = struct_elt(llstubty, abi::ivec_heap_stub_elt_ptr);
    let heap_ptr_is_null =
        bld::ICmp(on_heap_cx, lib::llvm::LLVMIntEQ, heap_ptr,
                              C_null(T_ptr(llheapptrty)));
    let zero_len_cx = new_sub_block_ctxt(bcx, "zero_len");
    let nonzero_len_cx = new_sub_block_ctxt(bcx, "nonzero_len");
    bld::CondBr(on_heap_cx, heap_ptr_is_null, zero_len_cx.llbb,
                            nonzero_len_cx.llbb);
    // Technically this context is unnecessary, but it makes this function
    // clearer.

    let zero_len = C_int(0);
    let zero_elem = C_null(T_ptr(llunitty));
    bld::Br(zero_len_cx, next_cx.llbb);
    // If we're here, then we actually have a heapified vector.

    let heap_len =
        load_inbounds(nonzero_len_cx, heap_ptr,
                      [C_int(0), C_uint(abi::ivec_heap_elt_len)]);
    let heap_elem =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems), C_int(0)];
        bld::InBoundsGEP(nonzero_len_cx, heap_ptr, v)
    };

    bld::Br(nonzero_len_cx, next_cx.llbb);
    // Now we can figure out the length of `v` and get a pointer to its
    // first element.

    let len =
        bld::Phi(next_cx, T_int(), [stack_len, zero_len, heap_len],
                          [bcx.llbb, zero_len_cx.llbb,
                           nonzero_len_cx.llbb]);
    let elem =
        bld::Phi(next_cx, T_ptr(llunitty),
                          [stack_elem, zero_elem, heap_elem],
                          [bcx.llbb, zero_len_cx.llbb,
                           nonzero_len_cx.llbb]);
    ret {len: len, data: elem, bcx: next_cx};
}

// Returns a tuple consisting of a pointer to the newly-reserved space and
// a block context. Updates the length appropriately.
fn reserve_space(cx: &@block_ctxt, llunitty: TypeRef, v: ValueRef,
                 len_needed: ValueRef) -> result {
    let stack_len_ptr =
        bld::InBoundsGEP(cx, v, [C_int(0), C_uint(abi::ivec_elt_len)]);
    let stack_len = bld::Load(cx, stack_len_ptr);
    let alen =
        load_inbounds(cx, v, [C_int(0), C_uint(abi::ivec_elt_alen)]);
    // There are four cases we have to consider:
    // (1) On heap, no resize necessary.
    // (2) On heap, need to resize.
    // (3) On stack, no resize necessary.
    // (4) On stack, need to spill to heap.

    let maybe_on_heap =
        bld::ICmp(cx, lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    let maybe_on_heap_cx = new_sub_block_ctxt(cx, "maybe_on_heap");
    let on_stack_cx = new_sub_block_ctxt(cx, "on_stack");
    bld::CondBr(cx, maybe_on_heap, maybe_on_heap_cx.llbb,
                    on_stack_cx.llbb);
    let next_cx = new_sub_block_ctxt(cx, "next");
    // We're possibly on the heap, unless the vector is zero-length.

    let stub_p = [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];
    let stub_ptr =
        bld::PointerCast(maybe_on_heap_cx, v,
                                           T_ptr(T_ivec_heap(llunitty)));
    let heap_ptr = load_inbounds(maybe_on_heap_cx, stub_ptr, stub_p);
    let on_heap =
        bld::ICmp(maybe_on_heap_cx, lib::llvm::LLVMIntNE, heap_ptr,
                                    C_null(val_ty(heap_ptr)));
    let on_heap_cx = new_sub_block_ctxt(cx, "on_heap");
    bld::CondBr(maybe_on_heap_cx, on_heap, on_heap_cx.llbb,
                                  on_stack_cx.llbb);
    // We're definitely on the heap. Check whether we need to resize.

    let heap_len_ptr =
        bld::InBoundsGEP(on_heap_cx, heap_ptr,
                                     [C_int(0),
                                      C_uint(abi::ivec_heap_elt_len)]);
    let heap_len = bld::Load(on_heap_cx, heap_len_ptr);
    let new_heap_len = bld::Add(on_heap_cx, heap_len, len_needed);
    let heap_len_unscaled =
        bld::UDiv(on_heap_cx, heap_len, llsize_of(llunitty));
    let heap_no_resize_needed =
        bld::ICmp(on_heap_cx, lib::llvm::LLVMIntULE, new_heap_len, alen);
    let heap_no_resize_cx = new_sub_block_ctxt(cx, "heap_no_resize");
    let heap_resize_cx = new_sub_block_ctxt(cx, "heap_resize");
    bld::CondBr(on_heap_cx, heap_no_resize_needed, heap_no_resize_cx.llbb,
                            heap_resize_cx.llbb);
    // Case (1): We're on the heap and don't need to resize.

    let heap_data_no_resize =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems),
             heap_len_unscaled];
        bld::InBoundsGEP(heap_no_resize_cx, heap_ptr, v)
    };
    bld::Store(heap_no_resize_cx, new_heap_len, heap_len_ptr);
    bld::Br(heap_no_resize_cx, next_cx.llbb);
    // Case (2): We're on the heap and need to resize. This path is rare,
    // so we delegate to cold glue.

    {
        let p =
            bld::PointerCast(heap_resize_cx, v, T_ptr(T_opaque_ivec()));
        let upcall = bcx_ccx(cx).upcalls.ivec_resize_shared;
        bld::Call(heap_resize_cx, upcall,
                                  [cx.fcx.lltaskptr, p, new_heap_len]);
    }
    let heap_ptr_resize = load_inbounds(heap_resize_cx, stub_ptr, stub_p);

    let heap_data_resize =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems),
             heap_len_unscaled];
        bld::InBoundsGEP(heap_resize_cx, heap_ptr_resize, v)
    };
    bld::Br(heap_resize_cx, next_cx.llbb);
    // We're on the stack. Check whether we need to spill to the heap.

    let new_stack_len = bld::Add(on_stack_cx, stack_len, len_needed);
    let stack_no_spill_needed =
        bld::ICmp(on_stack_cx, lib::llvm::LLVMIntULE, new_stack_len,
                               alen);
    let stack_len_unscaled =
        bld::UDiv(on_stack_cx, stack_len, llsize_of(llunitty));
    let stack_no_spill_cx = new_sub_block_ctxt(cx, "stack_no_spill");
    let stack_spill_cx = new_sub_block_ctxt(cx, "stack_spill");
    bld::CondBr(on_stack_cx, stack_no_spill_needed,
                             stack_no_spill_cx.llbb, stack_spill_cx.llbb);
    // Case (3): We're on the stack and don't need to spill.

    let stack_data_no_spill =
        bld::InBoundsGEP(stack_no_spill_cx, v,
                                            [C_int(0),
                                             C_uint(abi::ivec_elt_elems),
                                             stack_len_unscaled]);
    bld::Store(stack_no_spill_cx, new_stack_len, stack_len_ptr);
    bld::Br(stack_no_spill_cx, next_cx.llbb);
    // Case (4): We're on the stack and need to spill. Like case (2), this
    // path is rare, so we delegate to cold glue.

    {
        let p =
            bld::PointerCast(stack_spill_cx, v, T_ptr(T_opaque_ivec()));
        let upcall = bcx_ccx(cx).upcalls.ivec_spill_shared;
        bld::Call(stack_spill_cx, upcall,
                                  [cx.fcx.lltaskptr, p, new_stack_len]);
    }
    let spill_stub =
        bld::PointerCast(stack_spill_cx, v, T_ptr(T_ivec_heap(llunitty)));

    let heap_ptr_spill =
        load_inbounds(stack_spill_cx, spill_stub, stub_p);

    let heap_data_spill =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems),
             stack_len_unscaled];
        bld::InBoundsGEP(stack_spill_cx, heap_ptr_spill, v)
    };
    bld::Br(stack_spill_cx, next_cx.llbb);
    // Phi together the different data pointers to get the result.

    let data_ptr =
        bld::Phi(next_cx, T_ptr(llunitty),
                          [heap_data_no_resize, heap_data_resize,
                           stack_data_no_spill, heap_data_spill],
                          [heap_no_resize_cx.llbb, heap_resize_cx.llbb,
                           stack_no_spill_cx.llbb, stack_spill_cx.llbb]);
    ret rslt(next_cx, data_ptr);
}
fn trans_append(cx: &@block_ctxt, t: ty::t, lhs: ValueRef,
                rhs: ValueRef) -> result {
    // Cast to opaque interior vector types if necessary.
    if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        lhs = bld::PointerCast(cx, lhs, T_ptr(T_opaque_ivec()));
        rhs = bld::PointerCast(cx, rhs, T_ptr(T_opaque_ivec()));
    }

    let unit_ty = ty::sequence_element_type(bcx_tcx(cx), t);
    let llunitty = type_of_or_i8(cx, unit_ty);

    let rs = size_of(cx, unit_ty);
    let bcx = rs.bcx;
    let unit_sz = rs.val;

    // Gather the various type descriptors we'll need.

    // FIXME (issue #511): This is needed to prevent a leak.
    let no_tydesc_info = none;

    rs = get_tydesc(bcx, t, false, tps_normal, no_tydesc_info).result;
    bcx = rs.bcx;
    rs = get_tydesc(bcx, unit_ty, false, tps_normal, no_tydesc_info).result;
    bcx = rs.bcx;
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_take_glue, none);
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, none);
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, none);
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_copy_glue, none);
    let rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
    let rhs_len = rhs_len_and_data.len;
    let rhs_data = rhs_len_and_data.data;
    bcx = rhs_len_and_data.bcx;

    let have_istrs = alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_istr. { true }
      ty::ty_vec(_) { false }
      _ { bcx_tcx(cx).sess.bug("non-istr/ivec in trans_append"); }
    };

    let extra_len = if have_istrs {
        // Only need one of the nulls
        bld::Sub(bcx, rhs_len, C_uint(1u))
    } else { rhs_len };

    rs = reserve_space(bcx, llunitty, lhs, extra_len);
    bcx = rs.bcx;

    let lhs_data = if have_istrs {
        let lhs_data = rs.val;
        let lhs_data_without_null_ptr = alloca(bcx, T_ptr(llunitty));
        incr_ptr(bcx, lhs_data, C_int(-1),
                 lhs_data_without_null_ptr);
        bld::Load(bcx, lhs_data_without_null_ptr)
    } else {
        rs.val
    };

    // If rhs is lhs then our rhs pointer may have changed
    rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
    rhs_data = rhs_len_and_data.data;
    bcx = rhs_len_and_data.bcx;

    // Work out the end pointer.

    let lhs_unscaled_idx = bld::UDiv(bcx, rhs_len, llsize_of(llunitty));
    let lhs_end = bld::InBoundsGEP(bcx, lhs_data, [lhs_unscaled_idx]);
    // Now emit the copy loop.

    let dest_ptr = alloca(bcx, T_ptr(llunitty));
    bld::Store(bcx, lhs_data, dest_ptr);
    let src_ptr = alloca(bcx, T_ptr(llunitty));
    bld::Store(bcx, rhs_data, src_ptr);
    let copy_loop_header_cx = new_sub_block_ctxt(bcx, "copy_loop_header");
    bld::Br(bcx, copy_loop_header_cx.llbb);
    let copy_dest_ptr = bld::Load(copy_loop_header_cx, dest_ptr);
    let not_yet_at_end =
        bld::ICmp(copy_loop_header_cx, lib::llvm::LLVMIntNE,
                                       copy_dest_ptr, lhs_end);
    let copy_loop_body_cx = new_sub_block_ctxt(bcx, "copy_loop_body");
    let next_cx = new_sub_block_ctxt(bcx, "next");
    bld::CondBr(copy_loop_header_cx, not_yet_at_end,
                                     copy_loop_body_cx.llbb,
                                     next_cx.llbb);

    let copy_src_ptr = bld::Load(copy_loop_body_cx, src_ptr);
    let copy_src =
        load_if_immediate(copy_loop_body_cx, copy_src_ptr, unit_ty);

    let post_copy_cx = copy_val
        (copy_loop_body_cx, INIT, copy_dest_ptr, copy_src, unit_ty);
    // Increment both pointers.
    if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        // We have to increment by the dynamically-computed size.
        incr_ptr(post_copy_cx, copy_dest_ptr, unit_sz, dest_ptr);
        incr_ptr(post_copy_cx, copy_src_ptr, unit_sz, src_ptr);
    } else {
        incr_ptr(post_copy_cx, copy_dest_ptr, C_int(1), dest_ptr);
        incr_ptr(post_copy_cx, copy_src_ptr, C_int(1), src_ptr);
    }

    bld::Br(post_copy_cx, copy_loop_header_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn trans_append_literal(bcx: &@block_ctxt, v: ValueRef, vec_ty: ty::t,
                        vals: &[@ast::expr]) -> @block_ctxt {
    let elt_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);
    let ti = none;
    let {bcx, val: td} =
        get_tydesc(bcx, elt_ty, false, tps_normal, ti).result;
    trans::lazily_emit_all_tydesc_glue(bcx, ti);
    let opaque_v = bld::PointerCast(bcx, v, T_ptr(T_opaque_ivec()));
    for val in vals {
        let {bcx: e_bcx, val: elt} = trans::trans_expr(bcx, val);
        bcx = e_bcx;
        let spilled = trans::spill_if_immediate(bcx, elt, elt_ty);
        bld::Call(bcx, bcx_ccx(bcx).upcalls.ivec_push,
                       [bcx.fcx.lltaskptr, opaque_v, td,
                        bld::PointerCast(bcx, spilled, T_ptr(T_i8()))]);
    }
    ret bcx;
}

type alloc_result =
    {bcx: @block_ctxt,
     llptr: ValueRef,
     llunitsz: ValueRef,
     llalen: ValueRef};

fn alloc(cx: &@block_ctxt, unit_ty: ty::t) -> alloc_result {
    let dynamic = ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty);

    let bcx;
    if dynamic {
        bcx = llderivedtydescs_block_ctxt(cx.fcx);
    } else { bcx = cx; }

    let llunitsz;
    let rslt = size_of(bcx, unit_ty);
    bcx = rslt.bcx;
    llunitsz = rslt.val;

    if dynamic { cx.fcx.llderivedtydescs = bcx.llbb; }

    let llalen =
        bld::Mul(bcx, llunitsz, C_uint(abi::ivec_default_length));

    let llptr;
    let llunitty = type_of_or_i8(bcx, unit_ty);
    let bcx_result;
    if dynamic {
        let llarraysz = bld::Add(bcx, llsize_of(T_opaque_ivec()), llalen);
        let llvecptr = array_alloca(bcx, T_i8(), llarraysz);

        bcx_result = cx;
        llptr =
            bld::PointerCast(bcx_result, llvecptr,
                                         T_ptr(T_opaque_ivec()));
    } else { llptr = alloca(bcx, T_ivec(llunitty)); bcx_result = bcx; }

    ret {bcx: bcx_result,
         llptr: llptr,
         llunitsz: llunitsz,
         llalen: llalen};
}

fn trans_add(cx: &@block_ctxt, vec_ty: ty::t, lhs: ValueRef,
             rhs: ValueRef) -> result {
    let bcx = cx;
    let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);

    let ares = alloc(bcx, unit_ty);
    bcx = ares.bcx;
    let llvecptr = ares.llptr;
    let unit_sz = ares.llunitsz;
    let llalen = ares.llalen;

    add_clean_temp(bcx, llvecptr, vec_ty);

    let llunitty = type_of_or_i8(bcx, unit_ty);
    let llheappartty = T_ivec_heap_part(llunitty);
    let lhs_len_and_data = get_len_and_data(bcx, lhs, unit_ty);
    let lhs_len = lhs_len_and_data.len;
    let lhs_data = lhs_len_and_data.data;
    bcx = lhs_len_and_data.bcx;

    lhs_len = alt ty::struct(bcx_tcx(bcx), vec_ty) {
      ty::ty_istr. {
        // Forget about the trailing null on the left side
        bld::Sub(bcx, lhs_len, C_uint(1u))
      }
      ty::ty_vec(_) { lhs_len }
      _ { bcx_tcx(bcx).sess.bug("non-istr/ivec in trans_add") }
    };

    let rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
    let rhs_len = rhs_len_and_data.len;
    let rhs_data = rhs_len_and_data.data;
    bcx = rhs_len_and_data.bcx;
    let lllen = bld::Add(bcx, lhs_len, rhs_len);
    // We have three cases to handle here:
    // (1) Length is zero ([] + []).
    // (2) Copy onto stack.
    // (3) Allocate on heap and copy there.

    let len_is_zero =
        bld::ICmp(bcx, lib::llvm::LLVMIntEQ, lllen, C_int(0));
    let zero_len_cx = new_sub_block_ctxt(bcx, "zero_len");
    let nonzero_len_cx = new_sub_block_ctxt(bcx, "nonzero_len");
    bld::CondBr(bcx, len_is_zero, zero_len_cx.llbb, nonzero_len_cx.llbb);
    // Case (1): Length is zero.

    let stub_z = [C_int(0), C_uint(abi::ivec_heap_stub_elt_zero)];
    let stub_a = [C_int(0), C_uint(abi::ivec_heap_stub_elt_alen)];
    let stub_p = [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];

    let vec_l = [C_int(0), C_uint(abi::ivec_elt_len)];
    let vec_a = [C_int(0), C_uint(abi::ivec_elt_alen)];

    let stub_ptr_zero =
        bld::PointerCast(zero_len_cx, llvecptr,
                                      T_ptr(T_ivec_heap(llunitty)));
    bld::Store(zero_len_cx, C_int(0),
                            bld::InBoundsGEP(zero_len_cx, stub_ptr_zero,
                                                          stub_z));
    bld::Store(zero_len_cx, llalen,
                            bld::InBoundsGEP(zero_len_cx, stub_ptr_zero,
                                                          stub_a));
    bld::Store(zero_len_cx, C_null(T_ptr(llheappartty)),
                            bld::InBoundsGEP(zero_len_cx, stub_ptr_zero,
                                                          stub_p));
    let next_cx = new_sub_block_ctxt(bcx, "next");
    bld::Br(zero_len_cx, next_cx.llbb);
    // Determine whether we need to spill to the heap.

    let on_stack =
        bld::ICmp(nonzero_len_cx, lib::llvm::LLVMIntULE, lllen, llalen);
    let stack_cx = new_sub_block_ctxt(bcx, "stack");
    let heap_cx = new_sub_block_ctxt(bcx, "heap");
    bld::CondBr(nonzero_len_cx, on_stack, stack_cx.llbb, heap_cx.llbb);
    // Case (2): Copy onto stack.

    bld::Store(stack_cx, lllen,
                         bld::InBoundsGEP(stack_cx, llvecptr, vec_l));
    bld::Store(stack_cx, llalen,
                         bld::InBoundsGEP(stack_cx, llvecptr, vec_a));
    let dest_ptr_stack =
        bld::InBoundsGEP(stack_cx, llvecptr,
                                   [C_int(0), C_uint(abi::ivec_elt_elems),
                                    C_int(0)]);
    let copy_cx = new_sub_block_ctxt(bcx, "copy");
    bld::Br(stack_cx, copy_cx.llbb);
    // Case (3): Allocate on heap and copy there.

    let stub_ptr_heap =
        bld::PointerCast(heap_cx, llvecptr, T_ptr(T_ivec_heap(llunitty)));
    bld::Store(heap_cx, C_int(0),
                        bld::InBoundsGEP(heap_cx, stub_ptr_heap, stub_z));
    bld::Store(heap_cx, lllen,
                        bld::InBoundsGEP(heap_cx, stub_ptr_heap, stub_a));
    let heap_sz = bld::Add(heap_cx, llsize_of(llheappartty), lllen);
    let rs = trans_shared_malloc(heap_cx, T_ptr(llheappartty), heap_sz);
    let heap_part = rs.val;
    heap_cx = rs.bcx;
    bld::Store(heap_cx, heap_part,
                        bld::InBoundsGEP(heap_cx, stub_ptr_heap, stub_p));
    {
        let v = [C_int(0), C_uint(abi::ivec_heap_elt_len)];
        bld::Store(heap_cx, lllen,
                            bld::InBoundsGEP(heap_cx, heap_part, v));
    }
    let dest_ptr_heap =
        bld::InBoundsGEP(heap_cx, heap_part,
                                  [C_int(0),
                                   C_uint(abi::ivec_heap_elt_elems),
                                   C_int(0)]);
    bld::Br(heap_cx, copy_cx.llbb);
    // Emit the copy loop.

    let first_dest_ptr =
        bld::Phi(copy_cx, T_ptr(llunitty),
                          [dest_ptr_stack, dest_ptr_heap],
                          [stack_cx.llbb, heap_cx.llbb]);

    let lhs_end_ptr;
    let rhs_end_ptr;
    if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
        lhs_end_ptr = bld::InBoundsGEP(copy_cx, lhs_data, [lhs_len]);
        rhs_end_ptr = bld::InBoundsGEP(copy_cx, rhs_data, [rhs_len]);
    } else {
        let lhs_len_unscaled = bld::UDiv(copy_cx, lhs_len, unit_sz);
        lhs_end_ptr =
            bld::InBoundsGEP(copy_cx, lhs_data, [lhs_len_unscaled]);
        let rhs_len_unscaled = bld::UDiv(copy_cx, rhs_len, unit_sz);
        rhs_end_ptr =
            bld::InBoundsGEP(copy_cx, rhs_data, [rhs_len_unscaled]);
    }

    let dest_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
    bld::Store(copy_cx, first_dest_ptr, dest_ptr_ptr);
    let lhs_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
    bld::Store(copy_cx, lhs_data, lhs_ptr_ptr);
    let rhs_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
    bld::Store(copy_cx, rhs_data, rhs_ptr_ptr);
    let lhs_copy_cx = new_sub_block_ctxt(bcx, "lhs_copy");
    bld::Br(copy_cx, lhs_copy_cx.llbb);
    // Copy in elements from the LHS.

    let lhs_ptr = bld::Load(lhs_copy_cx, lhs_ptr_ptr);
    let not_at_end_lhs =
        bld::ICmp(lhs_copy_cx, lib::llvm::LLVMIntNE, lhs_ptr,
                               lhs_end_ptr);
    let lhs_do_copy_cx = new_sub_block_ctxt(bcx, "lhs_do_copy");
    let rhs_copy_cx = new_sub_block_ctxt(bcx, "rhs_copy");
    bld::CondBr(lhs_copy_cx, not_at_end_lhs, lhs_do_copy_cx.llbb,
                             rhs_copy_cx.llbb);
    let dest_ptr_lhs_copy = bld::Load(lhs_do_copy_cx, dest_ptr_ptr);
    let lhs_val = load_if_immediate(lhs_do_copy_cx, lhs_ptr, unit_ty);
    lhs_do_copy_cx = copy_val(lhs_do_copy_cx, INIT, dest_ptr_lhs_copy,
                              lhs_val, unit_ty);

    // Increment both pointers.
    if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
        // We have to increment by the dynamically-computed size.
        incr_ptr(lhs_do_copy_cx, dest_ptr_lhs_copy, unit_sz,
                 dest_ptr_ptr);
        incr_ptr(lhs_do_copy_cx, lhs_ptr, unit_sz, lhs_ptr_ptr);
    } else {
        incr_ptr(lhs_do_copy_cx, dest_ptr_lhs_copy, C_int(1),
                 dest_ptr_ptr);
        incr_ptr(lhs_do_copy_cx, lhs_ptr, C_int(1), lhs_ptr_ptr);
    }

    bld::Br(lhs_do_copy_cx, lhs_copy_cx.llbb);
    // Copy in elements from the RHS.

    let rhs_ptr = bld::Load(rhs_copy_cx, rhs_ptr_ptr);
    let not_at_end_rhs =
        bld::ICmp(rhs_copy_cx, lib::llvm::LLVMIntNE, rhs_ptr,
                               rhs_end_ptr);
    let rhs_do_copy_cx = new_sub_block_ctxt(bcx, "rhs_do_copy");
    bld::CondBr(rhs_copy_cx, not_at_end_rhs, rhs_do_copy_cx.llbb,
                             next_cx.llbb);
    let dest_ptr_rhs_copy = bld::Load(rhs_do_copy_cx, dest_ptr_ptr);
    let rhs_val = load_if_immediate(rhs_do_copy_cx, rhs_ptr, unit_ty);
    rhs_do_copy_cx = copy_val(rhs_do_copy_cx, INIT, dest_ptr_rhs_copy,
                              rhs_val, unit_ty);

    // Increment both pointers.
    if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
        // We have to increment by the dynamically-computed size.
        incr_ptr(rhs_do_copy_cx, dest_ptr_rhs_copy, unit_sz,
                 dest_ptr_ptr);
        incr_ptr(rhs_do_copy_cx, rhs_ptr, unit_sz, rhs_ptr_ptr);
    } else {
        incr_ptr(rhs_do_copy_cx, dest_ptr_rhs_copy, C_int(1),
                 dest_ptr_ptr);
        incr_ptr(rhs_do_copy_cx, rhs_ptr, C_int(1), rhs_ptr_ptr);
    }

    bld::Br(rhs_do_copy_cx, rhs_copy_cx.llbb);
    // Finally done!

    ret rslt(next_cx, llvecptr);
}

// NB: This does *not* adjust reference counts. The caller must have done
// this via take_ty() beforehand.
fn duplicate_heap_part(cx: &@block_ctxt, orig_vptr: ValueRef,
                       unit_ty: ty::t) -> result {
    // Cast to an opaque interior vector if we can't trust the pointer
    // type.
    let vptr;
    if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
        vptr = bld::PointerCast(cx, orig_vptr, T_ptr(T_opaque_ivec()));
    } else { vptr = orig_vptr; }

    let llunitty = type_of_or_i8(cx, unit_ty);
    let llheappartty = T_ivec_heap_part(llunitty);

    // Check to see if the vector is heapified.
    let stack_len_ptr =
        bld::InBoundsGEP(cx, vptr, [C_int(0), C_uint(abi::ivec_elt_len)]);
    let stack_len = bld::Load(cx, stack_len_ptr);
    let stack_len_is_zero =
        bld::ICmp(cx, lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    let maybe_on_heap_cx = new_sub_block_ctxt(cx, "maybe_on_heap");
    let next_cx = new_sub_block_ctxt(cx, "next");
    bld::CondBr(cx, stack_len_is_zero, maybe_on_heap_cx.llbb,
                    next_cx.llbb);

    let stub_ptr =
        bld::PointerCast(maybe_on_heap_cx, vptr,
                                           T_ptr(T_ivec_heap(llunitty)));
    let heap_ptr_ptr =
        bld::InBoundsGEP(maybe_on_heap_cx,
            stub_ptr,
            [C_int(0),
             C_uint(abi::ivec_heap_stub_elt_ptr)]);
    let heap_ptr = bld::Load(maybe_on_heap_cx, heap_ptr_ptr);
    let heap_ptr_is_nonnull =
        bld::ICmp(maybe_on_heap_cx, lib::llvm::LLVMIntNE, heap_ptr,
                                    C_null(T_ptr(llheappartty)));
    let on_heap_cx = new_sub_block_ctxt(cx, "on_heap");
    bld::CondBr(maybe_on_heap_cx, heap_ptr_is_nonnull, on_heap_cx.llbb,
                                  next_cx.llbb);

    // Ok, the vector is on the heap. Copy the heap part.
    let alen_ptr =
        bld::InBoundsGEP(on_heap_cx, stub_ptr,
            [C_int(0),
             C_uint(abi::ivec_heap_stub_elt_alen)]);
    let alen = bld::Load(on_heap_cx, alen_ptr);

    let heap_part_sz =
        bld::Add(on_heap_cx, alen, llsize_of(T_opaque_ivec_heap_part()));
    let rs =
        trans_shared_malloc(on_heap_cx, T_ptr(llheappartty),
                            heap_part_sz);
    on_heap_cx = rs.bcx;
    let new_heap_ptr = rs.val;

    rs = call_memmove(on_heap_cx, new_heap_ptr, heap_ptr, heap_part_sz);
    on_heap_cx = rs.bcx;

    bld::Store(on_heap_cx, new_heap_ptr, heap_ptr_ptr);
    bld::Br(on_heap_cx, next_cx.llbb);

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

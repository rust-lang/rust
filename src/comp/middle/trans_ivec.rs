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
               new_sub_block_ctxt};
import trans_common::*;

export trans_ivec, get_len_and_data, duplicate_heap_part, trans_add,
trans_append, alloc_with_heap;

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

    let lllen = bcx.build.Mul(C_uint(vecsz), unit_sz);
    // Allocate the vector pieces and store length and allocated length.

    let llfirsteltptr;
    if vecsz > 0u && vecsz <= abi::ivec_default_length {
        // Interior case.

        bcx.build.Store(lllen,
                        bcx.build.InBoundsGEP(llvecptr,
                                              [C_int(0),
                                               C_uint(abi::ivec_elt_len)]));
        bcx.build.Store(llalen,
                        bcx.build.InBoundsGEP(llvecptr,
                                              [C_int(0),
                                               C_uint(abi::ivec_elt_alen)]));
        llfirsteltptr =
            bcx.build.InBoundsGEP(llvecptr,
                                  [C_int(0), C_uint(abi::ivec_elt_elems),
                                   C_int(0)]);
    } else {
        // Heap case.

        let stub_z = [C_int(0), C_uint(abi::ivec_heap_stub_elt_zero)];
        let stub_a = [C_int(0), C_uint(abi::ivec_heap_stub_elt_alen)];
        let stub_p = [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];
        let llstubty = T_ivec_heap(llunitty);
        let llstubptr = bcx.build.PointerCast(llvecptr, T_ptr(llstubty));
        bcx.build.Store(C_int(0), bcx.build.InBoundsGEP(llstubptr, stub_z));
        let llheapty = T_ivec_heap_part(llunitty);
        if vecsz == 0u {
            // Null heap pointer indicates a zero-length vector.

            bcx.build.Store(llalen, bcx.build.InBoundsGEP(llstubptr, stub_a));
            bcx.build.Store(C_null(T_ptr(llheapty)),
                            bcx.build.InBoundsGEP(llstubptr, stub_p));
            llfirsteltptr = C_null(T_ptr(llunitty));
        } else {
            bcx.build.Store(lllen, bcx.build.InBoundsGEP(llstubptr, stub_a));

            let llheapsz = bcx.build.Add(llsize_of(llheapty), lllen);
            let rslt = trans_shared_malloc(bcx, T_ptr(llheapty), llheapsz);
            bcx = rslt.bcx;
            let llheapptr = rslt.val;
            bcx.build.Store(llheapptr,
                            bcx.build.InBoundsGEP(llstubptr, stub_p));
            let heap_l = [C_int(0), C_uint(abi::ivec_heap_elt_len)];
            bcx.build.Store(lllen, bcx.build.InBoundsGEP(llheapptr, heap_l));
            llfirsteltptr =
                bcx.build.InBoundsGEP(llheapptr,
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
                bcx.build.InBoundsGEP(llfirsteltptr,
                                      [bcx.build.Mul(C_uint(i), llunitsz)]);
        } else {
            lleltptr = bcx.build.InBoundsGEP(llfirsteltptr, [C_uint(i)]);
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
        v = bcx.build.PointerCast(orig_v, T_ptr(T_opaque_ivec()));
    } else { v = orig_v; }

    let llunitty = type_of_or_i8(bcx, unit_ty);
    let stack_len =
        load_inbounds(bcx, v, [C_int(0), C_uint(abi::ivec_elt_len)]);
    let stack_elem =
        bcx.build.InBoundsGEP(v,
                              [C_int(0), C_uint(abi::ivec_elt_elems),
                               C_int(0)]);
    let on_heap =
        bcx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    let on_heap_cx = new_sub_block_ctxt(bcx, "on_heap");
    let next_cx = new_sub_block_ctxt(bcx, "next");
    bcx.build.CondBr(on_heap, on_heap_cx.llbb, next_cx.llbb);
    let heap_stub =
        on_heap_cx.build.PointerCast(v, T_ptr(T_ivec_heap(llunitty)));
    let heap_ptr =
        load_inbounds(on_heap_cx, heap_stub,
                      [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)]);

    // Check whether the heap pointer is null. If it is, the vector length
    // is truly zero.

    let llstubty = T_ivec_heap(llunitty);
    let llheapptrty = struct_elt(llstubty, abi::ivec_heap_stub_elt_ptr);
    let heap_ptr_is_null =
        on_heap_cx.build.ICmp(lib::llvm::LLVMIntEQ, heap_ptr,
                              C_null(T_ptr(llheapptrty)));
    let zero_len_cx = new_sub_block_ctxt(bcx, "zero_len");
    let nonzero_len_cx = new_sub_block_ctxt(bcx, "nonzero_len");
    on_heap_cx.build.CondBr(heap_ptr_is_null, zero_len_cx.llbb,
                            nonzero_len_cx.llbb);
    // Technically this context is unnecessary, but it makes this function
    // clearer.

    let zero_len = C_int(0);
    let zero_elem = C_null(T_ptr(llunitty));
    zero_len_cx.build.Br(next_cx.llbb);
    // If we're here, then we actually have a heapified vector.

    let heap_len =
        load_inbounds(nonzero_len_cx, heap_ptr,
                      [C_int(0), C_uint(abi::ivec_heap_elt_len)]);
    let heap_elem =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems), C_int(0)];
        nonzero_len_cx.build.InBoundsGEP(heap_ptr, v)
    };

    nonzero_len_cx.build.Br(next_cx.llbb);
    // Now we can figure out the length of `v` and get a pointer to its
    // first element.

    let len =
        next_cx.build.Phi(T_int(), [stack_len, zero_len, heap_len],
                          [bcx.llbb, zero_len_cx.llbb,
                           nonzero_len_cx.llbb]);
    let elem =
        next_cx.build.Phi(T_ptr(llunitty),
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
        cx.build.InBoundsGEP(v, [C_int(0), C_uint(abi::ivec_elt_len)]);
    let stack_len = cx.build.Load(stack_len_ptr);
    let alen =
        load_inbounds(cx, v, [C_int(0), C_uint(abi::ivec_elt_alen)]);
    // There are four cases we have to consider:
    // (1) On heap, no resize necessary.
    // (2) On heap, need to resize.
    // (3) On stack, no resize necessary.
    // (4) On stack, need to spill to heap.

    let maybe_on_heap =
        cx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    let maybe_on_heap_cx = new_sub_block_ctxt(cx, "maybe_on_heap");
    let on_stack_cx = new_sub_block_ctxt(cx, "on_stack");
    cx.build.CondBr(maybe_on_heap, maybe_on_heap_cx.llbb,
                    on_stack_cx.llbb);
    let next_cx = new_sub_block_ctxt(cx, "next");
    // We're possibly on the heap, unless the vector is zero-length.

    let stub_p = [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];
    let stub_ptr =
        maybe_on_heap_cx.build.PointerCast(v,
                                           T_ptr(T_ivec_heap(llunitty)));
    let heap_ptr = load_inbounds(maybe_on_heap_cx, stub_ptr, stub_p);
    let on_heap =
        maybe_on_heap_cx.build.ICmp(lib::llvm::LLVMIntNE, heap_ptr,
                                    C_null(val_ty(heap_ptr)));
    let on_heap_cx = new_sub_block_ctxt(cx, "on_heap");
    maybe_on_heap_cx.build.CondBr(on_heap, on_heap_cx.llbb,
                                  on_stack_cx.llbb);
    // We're definitely on the heap. Check whether we need to resize.

    let heap_len_ptr =
        on_heap_cx.build.InBoundsGEP(heap_ptr,
                                     [C_int(0),
                                      C_uint(abi::ivec_heap_elt_len)]);
    let heap_len = on_heap_cx.build.Load(heap_len_ptr);
    let new_heap_len = on_heap_cx.build.Add(heap_len, len_needed);
    let heap_len_unscaled =
        on_heap_cx.build.UDiv(heap_len, llsize_of(llunitty));
    let heap_no_resize_needed =
        on_heap_cx.build.ICmp(lib::llvm::LLVMIntULE, new_heap_len, alen);
    let heap_no_resize_cx = new_sub_block_ctxt(cx, "heap_no_resize");
    let heap_resize_cx = new_sub_block_ctxt(cx, "heap_resize");
    on_heap_cx.build.CondBr(heap_no_resize_needed, heap_no_resize_cx.llbb,
                            heap_resize_cx.llbb);
    // Case (1): We're on the heap and don't need to resize.

    let heap_data_no_resize =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems),
             heap_len_unscaled];
        heap_no_resize_cx.build.InBoundsGEP(heap_ptr, v)
    };
    heap_no_resize_cx.build.Store(new_heap_len, heap_len_ptr);
    heap_no_resize_cx.build.Br(next_cx.llbb);
    // Case (2): We're on the heap and need to resize. This path is rare,
    // so we delegate to cold glue.

    {
        let p =
            heap_resize_cx.build.PointerCast(v, T_ptr(T_opaque_ivec()));
        let upcall = bcx_ccx(cx).upcalls.ivec_resize_shared;
        heap_resize_cx.build.Call(upcall,
                                  [cx.fcx.lltaskptr, p, new_heap_len]);
    }
    let heap_ptr_resize = load_inbounds(heap_resize_cx, stub_ptr, stub_p);

    let heap_data_resize =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems),
             heap_len_unscaled];
        heap_resize_cx.build.InBoundsGEP(heap_ptr_resize, v)
    };
    heap_resize_cx.build.Br(next_cx.llbb);
    // We're on the stack. Check whether we need to spill to the heap.

    let new_stack_len = on_stack_cx.build.Add(stack_len, len_needed);
    let stack_no_spill_needed =
        on_stack_cx.build.ICmp(lib::llvm::LLVMIntULE, new_stack_len,
                               alen);
    let stack_len_unscaled =
        on_stack_cx.build.UDiv(stack_len, llsize_of(llunitty));
    let stack_no_spill_cx = new_sub_block_ctxt(cx, "stack_no_spill");
    let stack_spill_cx = new_sub_block_ctxt(cx, "stack_spill");
    on_stack_cx.build.CondBr(stack_no_spill_needed,
                             stack_no_spill_cx.llbb, stack_spill_cx.llbb);
    // Case (3): We're on the stack and don't need to spill.

    let stack_data_no_spill =
        stack_no_spill_cx.build.InBoundsGEP(v,
                                            [C_int(0),
                                             C_uint(abi::ivec_elt_elems),
                                             stack_len_unscaled]);
    stack_no_spill_cx.build.Store(new_stack_len, stack_len_ptr);
    stack_no_spill_cx.build.Br(next_cx.llbb);
    // Case (4): We're on the stack and need to spill. Like case (2), this
    // path is rare, so we delegate to cold glue.

    {
        let p =
            stack_spill_cx.build.PointerCast(v, T_ptr(T_opaque_ivec()));
        let upcall = bcx_ccx(cx).upcalls.ivec_spill_shared;
        stack_spill_cx.build.Call(upcall,
                                  [cx.fcx.lltaskptr, p, new_stack_len]);
    }
    let spill_stub =
        stack_spill_cx.build.PointerCast(v, T_ptr(T_ivec_heap(llunitty)));

    let heap_ptr_spill =
        load_inbounds(stack_spill_cx, spill_stub, stub_p);

    let heap_data_spill =
        {
        let v =
            [C_int(0), C_uint(abi::ivec_heap_elt_elems),
             stack_len_unscaled];
        stack_spill_cx.build.InBoundsGEP(heap_ptr_spill, v)
    };
    stack_spill_cx.build.Br(next_cx.llbb);
    // Phi together the different data pointers to get the result.

    let data_ptr =
        next_cx.build.Phi(T_ptr(llunitty),
                          [heap_data_no_resize, heap_data_resize,
                           stack_data_no_spill, heap_data_spill],
                          [heap_no_resize_cx.llbb, heap_resize_cx.llbb,
                           stack_no_spill_cx.llbb, stack_spill_cx.llbb]);
    ret rslt(next_cx, data_ptr);
}
fn trans_append(cx: &@block_ctxt, t: ty::t, orig_lhs: ValueRef,
                orig_rhs: ValueRef) -> result {
    // Cast to opaque interior vector types if necessary.
    let lhs;
    let rhs;
    if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        lhs = cx.build.PointerCast(orig_lhs, T_ptr(T_opaque_ivec()));
        rhs = cx.build.PointerCast(orig_rhs, T_ptr(T_opaque_ivec()));
    } else { lhs = orig_lhs; rhs = orig_rhs; }

    let unit_ty = ty::sequence_element_type(bcx_tcx(cx), t);
    let llunitty = type_of_or_i8(cx, unit_ty);
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_istr. { }
      ty::ty_vec(_) { }
      _ { bcx_tcx(cx).sess.bug("non-istr/ivec in trans_append"); }
    }

    let rs = size_of(cx, unit_ty);
    let bcx = rs.bcx;
    let unit_sz = rs.val;

    // Gather the various type descriptors we'll need.

    // FIXME (issue #511): This is needed to prevent a leak.
    let no_tydesc_info = none;

    rs = get_tydesc(bcx, t, false, no_tydesc_info).result;
    bcx = rs.bcx;
    rs = get_tydesc(bcx, unit_ty, false, no_tydesc_info).result;
    bcx = rs.bcx;
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_take_glue, none);
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, none);
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, none);
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_copy_glue, none);
    let rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
    let rhs_len = rhs_len_and_data.len;
    let rhs_data = rhs_len_and_data.data;
    bcx = rhs_len_and_data.bcx;
    rs = reserve_space(bcx, llunitty, lhs, rhs_len);
    let lhs_data = rs.val;
    bcx = rs.bcx;

    // If rhs is lhs then our rhs pointer may have changed
    rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
    rhs_data = rhs_len_and_data.data;
    bcx = rhs_len_and_data.bcx;

    // Work out the end pointer.

    let lhs_unscaled_idx = bcx.build.UDiv(rhs_len, llsize_of(llunitty));
    let lhs_end = bcx.build.InBoundsGEP(lhs_data, [lhs_unscaled_idx]);
    // Now emit the copy loop.

    let dest_ptr = alloca(bcx, T_ptr(llunitty));
    bcx.build.Store(lhs_data, dest_ptr);
    let src_ptr = alloca(bcx, T_ptr(llunitty));
    bcx.build.Store(rhs_data, src_ptr);
    let copy_loop_header_cx = new_sub_block_ctxt(bcx, "copy_loop_header");
    bcx.build.Br(copy_loop_header_cx.llbb);
    let copy_dest_ptr = copy_loop_header_cx.build.Load(dest_ptr);
    let not_yet_at_end =
        copy_loop_header_cx.build.ICmp(lib::llvm::LLVMIntNE,
                                       copy_dest_ptr, lhs_end);
    let copy_loop_body_cx = new_sub_block_ctxt(bcx, "copy_loop_body");
    let next_cx = new_sub_block_ctxt(bcx, "next");
    copy_loop_header_cx.build.CondBr(not_yet_at_end,
                                     copy_loop_body_cx.llbb,
                                     next_cx.llbb);

    let copy_src_ptr = copy_loop_body_cx.build.Load(src_ptr);
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

    post_copy_cx.build.Br(copy_loop_header_cx.llbb);
    ret rslt(next_cx, C_nil());
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
        bcx.build.Mul(llunitsz, C_uint(abi::ivec_default_length));

    let llptr;
    let llunitty = type_of_or_i8(bcx, unit_ty);
    let bcx_result;
    if dynamic {
        let llarraysz = bcx.build.Add(llsize_of(T_opaque_ivec()), llalen);
        let llvecptr = array_alloca(bcx, T_i8(), llarraysz);

        bcx_result = cx;
        llptr =
            bcx_result.build.PointerCast(llvecptr,
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
    let rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
    let rhs_len = rhs_len_and_data.len;
    let rhs_data = rhs_len_and_data.data;
    bcx = rhs_len_and_data.bcx;
    let lllen = bcx.build.Add(lhs_len, rhs_len);
    // We have three cases to handle here:
    // (1) Length is zero ([] + []).
    // (2) Copy onto stack.
    // (3) Allocate on heap and copy there.

    let len_is_zero =
        bcx.build.ICmp(lib::llvm::LLVMIntEQ, lllen, C_int(0));
    let zero_len_cx = new_sub_block_ctxt(bcx, "zero_len");
    let nonzero_len_cx = new_sub_block_ctxt(bcx, "nonzero_len");
    bcx.build.CondBr(len_is_zero, zero_len_cx.llbb, nonzero_len_cx.llbb);
    // Case (1): Length is zero.

    let stub_z = [C_int(0), C_uint(abi::ivec_heap_stub_elt_zero)];
    let stub_a = [C_int(0), C_uint(abi::ivec_heap_stub_elt_alen)];
    let stub_p = [C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];

    let vec_l = [C_int(0), C_uint(abi::ivec_elt_len)];
    let vec_a = [C_int(0), C_uint(abi::ivec_elt_alen)];

    let stub_ptr_zero =
        zero_len_cx.build.PointerCast(llvecptr,
                                      T_ptr(T_ivec_heap(llunitty)));
    zero_len_cx.build.Store(C_int(0),
                            zero_len_cx.build.InBoundsGEP(stub_ptr_zero,
                                                          stub_z));
    zero_len_cx.build.Store(llalen,
                            zero_len_cx.build.InBoundsGEP(stub_ptr_zero,
                                                          stub_a));
    zero_len_cx.build.Store(C_null(T_ptr(llheappartty)),
                            zero_len_cx.build.InBoundsGEP(stub_ptr_zero,
                                                          stub_p));
    let next_cx = new_sub_block_ctxt(bcx, "next");
    zero_len_cx.build.Br(next_cx.llbb);
    // Determine whether we need to spill to the heap.

    let on_stack =
        nonzero_len_cx.build.ICmp(lib::llvm::LLVMIntULE, lllen, llalen);
    let stack_cx = new_sub_block_ctxt(bcx, "stack");
    let heap_cx = new_sub_block_ctxt(bcx, "heap");
    nonzero_len_cx.build.CondBr(on_stack, stack_cx.llbb, heap_cx.llbb);
    // Case (2): Copy onto stack.

    stack_cx.build.Store(lllen,
                         stack_cx.build.InBoundsGEP(llvecptr, vec_l));
    stack_cx.build.Store(llalen,
                         stack_cx.build.InBoundsGEP(llvecptr, vec_a));
    let dest_ptr_stack =
        stack_cx.build.InBoundsGEP(llvecptr,
                                   [C_int(0), C_uint(abi::ivec_elt_elems),
                                    C_int(0)]);
    let copy_cx = new_sub_block_ctxt(bcx, "copy");
    stack_cx.build.Br(copy_cx.llbb);
    // Case (3): Allocate on heap and copy there.

    let stub_ptr_heap =
        heap_cx.build.PointerCast(llvecptr, T_ptr(T_ivec_heap(llunitty)));
    heap_cx.build.Store(C_int(0),
                        heap_cx.build.InBoundsGEP(stub_ptr_heap, stub_z));
    heap_cx.build.Store(lllen,
                        heap_cx.build.InBoundsGEP(stub_ptr_heap, stub_a));
    let heap_sz = heap_cx.build.Add(llsize_of(llheappartty), lllen);
    let rs = trans_shared_malloc(heap_cx, T_ptr(llheappartty), heap_sz);
    let heap_part = rs.val;
    heap_cx = rs.bcx;
    heap_cx.build.Store(heap_part,
                        heap_cx.build.InBoundsGEP(stub_ptr_heap, stub_p));
    {
        let v = [C_int(0), C_uint(abi::ivec_heap_elt_len)];
        heap_cx.build.Store(lllen,
                            heap_cx.build.InBoundsGEP(heap_part, v));
    }
    let dest_ptr_heap =
        heap_cx.build.InBoundsGEP(heap_part,
                                  [C_int(0),
                                   C_uint(abi::ivec_heap_elt_elems),
                                   C_int(0)]);
    heap_cx.build.Br(copy_cx.llbb);
    // Emit the copy loop.

    let first_dest_ptr =
        copy_cx.build.Phi(T_ptr(llunitty),
                          [dest_ptr_stack, dest_ptr_heap],
                          [stack_cx.llbb, heap_cx.llbb]);

    let lhs_end_ptr;
    let rhs_end_ptr;
    if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
        lhs_end_ptr = copy_cx.build.InBoundsGEP(lhs_data, [lhs_len]);
        rhs_end_ptr = copy_cx.build.InBoundsGEP(rhs_data, [rhs_len]);
    } else {
        let lhs_len_unscaled = copy_cx.build.UDiv(lhs_len, unit_sz);
        lhs_end_ptr =
            copy_cx.build.InBoundsGEP(lhs_data, [lhs_len_unscaled]);
        let rhs_len_unscaled = copy_cx.build.UDiv(rhs_len, unit_sz);
        rhs_end_ptr =
            copy_cx.build.InBoundsGEP(rhs_data, [rhs_len_unscaled]);
    }

    let dest_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
    copy_cx.build.Store(first_dest_ptr, dest_ptr_ptr);
    let lhs_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
    copy_cx.build.Store(lhs_data, lhs_ptr_ptr);
    let rhs_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
    copy_cx.build.Store(rhs_data, rhs_ptr_ptr);
    let lhs_copy_cx = new_sub_block_ctxt(bcx, "lhs_copy");
    copy_cx.build.Br(lhs_copy_cx.llbb);
    // Copy in elements from the LHS.

    let lhs_ptr = lhs_copy_cx.build.Load(lhs_ptr_ptr);
    let not_at_end_lhs =
        lhs_copy_cx.build.ICmp(lib::llvm::LLVMIntNE, lhs_ptr,
                               lhs_end_ptr);
    let lhs_do_copy_cx = new_sub_block_ctxt(bcx, "lhs_do_copy");
    let rhs_copy_cx = new_sub_block_ctxt(bcx, "rhs_copy");
    lhs_copy_cx.build.CondBr(not_at_end_lhs, lhs_do_copy_cx.llbb,
                             rhs_copy_cx.llbb);
    let dest_ptr_lhs_copy = lhs_do_copy_cx.build.Load(dest_ptr_ptr);
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

    lhs_do_copy_cx.build.Br(lhs_copy_cx.llbb);
    // Copy in elements from the RHS.

    let rhs_ptr = rhs_copy_cx.build.Load(rhs_ptr_ptr);
    let not_at_end_rhs =
        rhs_copy_cx.build.ICmp(lib::llvm::LLVMIntNE, rhs_ptr,
                               rhs_end_ptr);
    let rhs_do_copy_cx = new_sub_block_ctxt(bcx, "rhs_do_copy");
    rhs_copy_cx.build.CondBr(not_at_end_rhs, rhs_do_copy_cx.llbb,
                             next_cx.llbb);
    let dest_ptr_rhs_copy = rhs_do_copy_cx.build.Load(dest_ptr_ptr);
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

    rhs_do_copy_cx.build.Br(rhs_copy_cx.llbb);
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
        vptr = cx.build.PointerCast(orig_vptr, T_ptr(T_opaque_ivec()));
    } else { vptr = orig_vptr; }

    let llunitty = type_of_or_i8(cx, unit_ty);
    let llheappartty = T_ivec_heap_part(llunitty);

    // Check to see if the vector is heapified.
    let stack_len_ptr =
        cx.build.InBoundsGEP(vptr, [C_int(0), C_uint(abi::ivec_elt_len)]);
    let stack_len = cx.build.Load(stack_len_ptr);
    let stack_len_is_zero =
        cx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    let maybe_on_heap_cx = new_sub_block_ctxt(cx, "maybe_on_heap");
    let next_cx = new_sub_block_ctxt(cx, "next");
    cx.build.CondBr(stack_len_is_zero, maybe_on_heap_cx.llbb,
                    next_cx.llbb);

    let stub_ptr =
        maybe_on_heap_cx.build.PointerCast(vptr,
                                           T_ptr(T_ivec_heap(llunitty)));
    let heap_ptr_ptr =
        maybe_on_heap_cx.build.InBoundsGEP(
            stub_ptr,
            [C_int(0),
             C_uint(abi::ivec_heap_stub_elt_ptr)]);
    let heap_ptr = maybe_on_heap_cx.build.Load(heap_ptr_ptr);
    let heap_ptr_is_nonnull =
        maybe_on_heap_cx.build.ICmp(lib::llvm::LLVMIntNE, heap_ptr,
                                    C_null(T_ptr(llheappartty)));
    let on_heap_cx = new_sub_block_ctxt(cx, "on_heap");
    maybe_on_heap_cx.build.CondBr(heap_ptr_is_nonnull, on_heap_cx.llbb,
                                  next_cx.llbb);

    // Ok, the vector is on the heap. Copy the heap part.
    let alen_ptr =
        on_heap_cx.build.InBoundsGEP(
            stub_ptr,
            [C_int(0),
             C_uint(abi::ivec_heap_stub_elt_alen)]);
    let alen = on_heap_cx.build.Load(alen_ptr);

    let heap_part_sz =
        on_heap_cx.build.Add(alen, llsize_of(T_opaque_ivec_heap_part()));
    let rs =
        trans_shared_malloc(on_heap_cx, T_ptr(llheappartty),
                            heap_part_sz);
    on_heap_cx = rs.bcx;
    let new_heap_ptr = rs.val;

    rs = call_memmove(on_heap_cx, new_heap_ptr, heap_ptr, heap_part_sz);
    on_heap_cx = rs.bcx;

    on_heap_cx.build.Store(new_heap_ptr, heap_ptr_ptr);
    on_heap_cx.build.Br(next_cx.llbb);

    ret rslt(next_cx, C_nil());
}

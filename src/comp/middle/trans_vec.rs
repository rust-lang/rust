// Translation of vector operations to LLVM IR, in destination-passing style.

import back::abi;
import lib::llvm::llvm;
import llvm::ValueRef;
import middle::trans;
import middle::trans_common;
import middle::trans_dps;
import middle::ty;
import syntax::ast;
import syntax::codemap::span;
import trans::alloca;
import trans::load_inbounds;
import trans::new_sub_block_ctxt;
import trans::type_of_or_i8;
import trans_common::block_ctxt;
import trans_common::struct_elt;
import trans_common::C_int;
import trans_common::C_null;
import trans_common::C_uint;
import trans_common::T_int;
import trans_common::T_ivec_heap;
import trans_common::T_ivec_heap_part;
import trans_common::T_opaque_ivec;
import trans_common::T_ptr;
import trans_common::bcx_ccx;
import trans_common::bcx_tcx;
import trans_dps::dest;
import trans_dps::llsize_of;
import trans_dps::mk_temp;

import std::option::none;
import std::option::some;
import tc = middle::trans_common;

// Returns the length of an interior vector and a pointer to its first
// element, in that order.
//
// TODO: We can optimize this in the cases in which we statically know the
// vector must be on the stack.
fn get_len_and_data(&@block_ctxt cx, ty::t t, ValueRef llvecptr)
        -> rec(@block_ctxt bcx, ValueRef len, ValueRef data) {
    auto bcx = cx;

    // If this interior vector has dynamic size, we can't assume anything
    // about the LLVM type of the value passed in, so we cast it to an
    // opaque vector type.
    auto unit_ty = ty::sequence_element_type(bcx_tcx(bcx), t);
    auto v;
    if (ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty)) {
        v = bcx.build.PointerCast(llvecptr, T_ptr(T_opaque_ivec()));
    } else {
        v = llvecptr;
    }

    auto llunitty = type_of_or_i8(bcx, unit_ty);
    auto stack_len = load_inbounds(bcx, v, ~[C_int(0),
                                             C_uint(abi::ivec_elt_len)]);
    auto stack_elem =
        bcx.build.InBoundsGEP(v,
                              ~[C_int(0), C_uint(abi::ivec_elt_elems),
                                C_int(0)]);
    auto on_heap =
        bcx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    auto on_heap_cx = new_sub_block_ctxt(bcx, "on_heap");
    auto next_cx = new_sub_block_ctxt(bcx, "next");
    bcx.build.CondBr(on_heap, on_heap_cx.llbb, next_cx.llbb);
    auto heap_stub =
        on_heap_cx.build.PointerCast(v, T_ptr(T_ivec_heap(llunitty)));
    auto heap_ptr = load_inbounds(on_heap_cx, heap_stub,
                                  ~[C_int(0),
                                    C_uint(abi::ivec_heap_stub_elt_ptr)]);

    // Check whether the heap pointer is null. If it is, the vector length
    // is truly zero.

    auto llstubty = T_ivec_heap(llunitty);
    auto llheapptrty = struct_elt(llstubty, abi::ivec_heap_stub_elt_ptr);
    auto heap_ptr_is_null =
        on_heap_cx.build.ICmp(lib::llvm::LLVMIntEQ, heap_ptr,
                              C_null(T_ptr(llheapptrty)));
    auto zero_len_cx = new_sub_block_ctxt(bcx, "zero_len");
    auto nonzero_len_cx = new_sub_block_ctxt(bcx, "nonzero_len");
    on_heap_cx.build.CondBr(heap_ptr_is_null, zero_len_cx.llbb,
                            nonzero_len_cx.llbb);
    // Technically this context is unnecessary, but it makes this function
    // clearer.

    auto zero_len = C_int(0);
    auto zero_elem = C_null(T_ptr(llunitty));
    zero_len_cx.build.Br(next_cx.llbb);
    // If we're here, then we actually have a heapified vector.

    auto heap_len = load_inbounds(nonzero_len_cx, heap_ptr,
                                  ~[C_int(0),
                                    C_uint(abi::ivec_heap_elt_len)]);
    auto heap_elem =
        {
            auto v = ~[C_int(0), C_uint(abi::ivec_heap_elt_elems),
                       C_int(0)];
            nonzero_len_cx.build.InBoundsGEP(heap_ptr,v)
        };

    nonzero_len_cx.build.Br(next_cx.llbb);

    // Now we can figure out the length of |v| and get a pointer to its
    // first element.

    auto len =
        next_cx.build.Phi(T_int(), ~[stack_len, zero_len, heap_len],
                          ~[bcx.llbb, zero_len_cx.llbb,
                            nonzero_len_cx.llbb]);
    auto elem =
        next_cx.build.Phi(T_ptr(llunitty),
                          ~[stack_elem, zero_elem, heap_elem],
                          ~[bcx.llbb, zero_len_cx.llbb,
                            nonzero_len_cx.llbb]);
    ret rec(bcx=next_cx, len=len, data=elem);
}

fn trans_concat(&@block_ctxt cx, &dest in_dest, &span sp, ty::t t,
                &@ast::expr lhs, &@ast::expr rhs) -> @block_ctxt {
    auto bcx = cx;

    // TODO: Detect "a = a + b" and promote to trans_append.
    // TODO: Detect "a + [ literal ]" and optimize to copying the literal
    //       elements in directly.

    auto t = ty::expr_ty(bcx_tcx(bcx), lhs);
    auto skip_null = ty::type_is_str(bcx_tcx(bcx), t);

    // Translate the LHS and RHS. Pull out their length and data.
    auto lhs_tmp = trans_dps::dest_alias(bcx_tcx(bcx), t);
    bcx = trans_dps::trans_expr(bcx, lhs_tmp, lhs);
    auto lllhsptr = trans_dps::dest_ptr(lhs_tmp);

    auto rhs_tmp = trans_dps::dest_alias(bcx_tcx(bcx), t);
    bcx = trans_dps::trans_expr(bcx, rhs_tmp, rhs);
    auto llrhsptr = trans_dps::dest_ptr(rhs_tmp);

    auto r0 = get_len_and_data(bcx, t, lllhsptr);
    bcx = r0.bcx; auto lllhslen = r0.len; auto lllhsdata = r0.data;
    r0 = get_len_and_data(bcx, t, llrhsptr);
    bcx = r0.bcx; auto llrhslen = r0.len; auto llrhsdata = r0.data;

    if skip_null { lllhslen = bcx.build.Sub(lllhslen, C_int(1)); }

    // Allocate the destination.
    auto r1 = trans_dps::spill_alias(bcx, in_dest, t);
    bcx = r1.bcx; auto dest = r1.dest;

    auto unit_t = ty::sequence_element_type(bcx_tcx(bcx), t);
    auto unit_sz = trans_dps::size_of(bcx_ccx(bcx), sp, unit_t);

    auto stack_elems_sz = unit_sz * abi::ivec_default_length;
    auto lldestptr = trans_dps::dest_ptr(dest);
    auto llunitty = trans::type_of(bcx_ccx(bcx), sp, unit_t);

    // Decide whether to allocate the result on the stack or on the heap.
    auto llnewlen = bcx.build.Add(lllhslen, llrhslen);
    auto llonstack = bcx.build.ICmp(lib::llvm::LLVMIntULE, llnewlen,
                                    C_uint(stack_elems_sz));
    auto on_stack_bcx = new_sub_block_ctxt(bcx, "on_stack");
    auto on_heap_bcx = new_sub_block_ctxt(bcx, "on_heap");
    bcx.build.CondBr(llonstack, on_stack_bcx.llbb, on_heap_bcx.llbb);

    // On-stack case.
    auto next_bcx = new_sub_block_ctxt(bcx, "next");
    trans::store_inbounds(on_stack_bcx, llnewlen, lldestptr,
                          ~[C_int(0), C_uint(abi::ivec_elt_len)]);
    trans::store_inbounds(on_stack_bcx, C_uint(stack_elems_sz), lldestptr,
                          ~[C_int(0), C_uint(abi::ivec_elt_alen)]);
    auto llonstackdataptr =
        on_stack_bcx.build.InBoundsGEP(lldestptr,
                                       ~[C_int(0),
                                         C_uint(abi::ivec_elt_elems),
                                         C_int(0)]);
    on_stack_bcx.build.Br(next_bcx.llbb);

    // On-heap case.
    auto llheappartty = tc::T_ivec_heap(llunitty);
    auto lldeststubptr =
        on_heap_bcx.build.PointerCast(lldestptr, tc::T_ptr(llheappartty));
    trans::store_inbounds(on_heap_bcx, C_int(0), lldeststubptr,
                          ~[C_int(0), C_uint(abi::ivec_elt_len)]);
    trans::store_inbounds(on_heap_bcx, llnewlen, lldeststubptr,
                          ~[C_int(0), C_uint(abi::ivec_elt_alen)]);

    auto llheappartptrptr =
        on_heap_bcx.build.InBoundsGEP(lldeststubptr,
                                      ~[C_int(0),
                                        C_uint(abi::ivec_elt_elems)]);
    auto llsizeofint = C_uint(llsize_of(bcx_ccx(bcx), tc::T_int()));
    on_heap_bcx = trans_dps::malloc(on_heap_bcx, llheappartptrptr,
                                    trans_dps::hp_shared,
                                    some(on_heap_bcx.build.Add(llnewlen,
                                                               llsizeofint)));
    auto llheappartptr = on_heap_bcx.build.Load(llheappartptrptr);
    trans::store_inbounds(on_heap_bcx, llnewlen, llheappartptr,
                          ~[C_int(0), C_uint(abi::ivec_heap_elt_len)]);
    auto llheapdataptr =
        on_heap_bcx.build.InBoundsGEP(llheappartptr,
                                      ~[C_int(0),
                                        C_uint(abi::ivec_heap_elt_elems),
                                        C_int(0)]);
    on_heap_bcx.build.Br(next_bcx.llbb);

    // Perform the memmove.
    auto lldataptr =
        next_bcx.build.Phi(T_ptr(llunitty),
                           ~[llonstackdataptr, llheapdataptr],
                           ~[on_stack_bcx.llbb, on_heap_bcx.llbb]);
    trans_dps::memmove(next_bcx, lldataptr, lllhsdata, lllhslen);
    trans_dps::memmove(next_bcx,
                       next_bcx.build.InBoundsGEP(lldataptr, ~[lllhslen]),
                       llrhsdata, llrhslen);

    ret next_bcx;
}


import syntax::ast;
import driver::session::session;
import lib::llvm::{ValueRef, TypeRef};
import back::abi;
import base::{call_memmove, shared_malloc,
               INIT, copy_val, load_if_immediate, get_tydesc,
               sub_block, do_spill_noroot,
               dest, bcx_icx};
import shape::llsize_of;
import build::*;
import common::*;

fn get_fill(bcx: block, vptr: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt("tvec::get_fill");
    Load(bcx, GEPi(bcx, vptr, [0, abi::vec_elt_fill]))
}
fn set_fill(bcx: block, vptr: ValueRef, fill: ValueRef) {
    Store(bcx, fill, GEPi(bcx, vptr, [0, abi::vec_elt_fill]));
}
fn get_alloc(bcx: block, vptr: ValueRef) -> ValueRef {
    Load(bcx, GEPi(bcx, vptr, [0, abi::vec_elt_alloc]))
}
fn get_dataptr(bcx: block, vptr: ValueRef, unit_ty: TypeRef)
    -> ValueRef {
    let _icx = bcx.insn_ctxt("tvec::get_dataptr");
    let ptr = GEPi(bcx, vptr, [0, abi::vec_elt_elems]);
    PointerCast(bcx, ptr, T_ptr(unit_ty))
}

fn pointer_add(bcx: block, ptr: ValueRef, bytes: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt("tvec::pointer_add");
    let old_ty = val_ty(ptr);
    let bptr = PointerCast(bcx, ptr, T_ptr(T_i8()));
    ret PointerCast(bcx, InBoundsGEP(bcx, bptr, [bytes]), old_ty);
}

fn alloc_raw(bcx: block, fill: ValueRef, alloc: ValueRef) -> result {
    let _icx = bcx.insn_ctxt("tvec::alloc_raw");
    let ccx = bcx.ccx();
    let llvecty = ccx.opaque_vec_type;
    let vecsize = Add(bcx, alloc, llsize_of(ccx, llvecty));
    let vecptr = shared_malloc(bcx, T_ptr(llvecty), vecsize);
    Store(bcx, fill, GEPi(bcx, vecptr, [0, abi::vec_elt_fill]));
    Store(bcx, alloc, GEPi(bcx, vecptr, [0, abi::vec_elt_alloc]));
    ret {bcx: bcx, val: vecptr};
}

type alloc_result =
    {bcx: block,
     val: ValueRef,
     unit_ty: ty::t,
     llunitty: TypeRef};

fn alloc(bcx: block, vec_ty: ty::t, elts: uint) -> alloc_result {
    let _icx = bcx.insn_ctxt("tvec::alloc");
    let ccx = bcx.ccx();
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let llunitty = type_of::type_of(ccx, unit_ty);
    let llvecty = T_vec(ccx, llunitty);
    let unit_sz = llsize_of(ccx, llunitty);

    let fill = Mul(bcx, C_uint(ccx, elts), unit_sz);
    let alloc = if elts < 4u { Mul(bcx, C_int(ccx, 4), unit_sz) }
                else { fill };
    let {bcx: bcx, val: vptr} = alloc_raw(bcx, fill, alloc);
    let vptr = PointerCast(bcx, vptr, T_ptr(llvecty));

    ret {bcx: bcx,
         val: vptr,
         unit_ty: unit_ty,
         llunitty: llunitty};
}

fn duplicate(bcx: block, vptr: ValueRef, vec_ty: ty::t) -> result {
    let _icx = bcx.insn_ctxt("tvec::duplicate");
    let ccx = bcx.ccx();
    let fill = get_fill(bcx, vptr);
    let size = Add(bcx, fill, llsize_of(ccx, ccx.opaque_vec_type));
    let newptr = shared_malloc(bcx, val_ty(vptr), size);
    call_memmove(bcx, newptr, vptr, size);
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    Store(bcx, fill, GEPi(bcx, newptr, [0, abi::vec_elt_alloc]));
    let bcx = if ty::type_needs_drop(bcx.tcx(), unit_ty) {
        iter_vec(bcx, newptr, vec_ty, base::take_ty)
    } else { bcx };
    ret rslt(bcx, newptr);
}
fn make_free_glue(bcx: block, vptr: ValueRef, vec_ty: ty::t) ->
   block {
    let _icx = bcx.insn_ctxt("tvec::make_free_glue");
    let tcx = bcx.tcx(), unit_ty = ty::sequence_element_type(tcx, vec_ty);
    base::with_cond(bcx, IsNotNull(bcx, vptr)) {|bcx|
        let bcx = if ty::type_needs_drop(tcx, unit_ty) {
            iter_vec(bcx, vptr, vec_ty, base::drop_ty)
        } else { bcx };
        base::trans_shared_free(bcx, vptr)
    }
}

fn trans_vec(bcx: block, args: [@ast::expr], id: ast::node_id,
             dest: dest) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_vec");
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    if dest == base::ignore {
        for arg in args {
            bcx = base::trans_expr(bcx, arg, base::ignore);
        }
        ret bcx;
    }
    let vec_ty = node_id_type(bcx, id);
    let mut {bcx: bcx,
             val: vptr,
             unit_ty: unit_ty,
             llunitty: llunitty} = alloc(bcx, vec_ty, args.len());

    add_clean_free(bcx, vptr, true);
    // Store the individual elements.
    let dataptr = get_dataptr(bcx, vptr, llunitty);
    let mut i = 0u, temp_cleanups = [vptr];
    for e in args {
        let lleltptr = InBoundsGEP(bcx, dataptr, [C_uint(ccx, i)]);
        bcx = base::trans_expr_save_in(bcx, e, lleltptr);
        add_clean_temp_mem(bcx, lleltptr, unit_ty);
        temp_cleanups += [lleltptr];
        i += 1u;
    }
    for cln in temp_cleanups { revoke_clean(bcx, cln); }
    ret base::store_in_dest(bcx, vptr, dest);
}

fn trans_str(bcx: block, s: str, dest: dest) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_str");
    let veclen = str::len(s) + 1u; // +1 for \0
    let {bcx: bcx, val: sptr, _} =
        alloc(bcx, ty::mk_str(bcx.tcx()), veclen);

    let ccx = bcx.ccx();
    let llcstr = C_cstr(ccx, s);
    call_memmove(bcx, get_dataptr(bcx, sptr, T_i8()), llcstr,
                 C_uint(ccx, veclen));
    ret base::store_in_dest(bcx, sptr, dest);
}

fn trans_append(bcx: block, vec_ty: ty::t, lhsptr: ValueRef,
                rhs: ValueRef) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_append");
    // Cast to opaque interior vector types if necessary.
    let ccx = bcx.ccx();
    let unit_ty = ty::sequence_element_type(ccx.tcx, vec_ty);
    let strings = alt check ty::get(vec_ty).struct {
      ty::ty_str { true }
      ty::ty_vec(_) { false }
    };

    let llunitty = type_of::type_of(ccx, unit_ty);

    let lhs = Load(bcx, lhsptr);
    let self_append = ICmp(bcx, lib::llvm::IntEQ, lhs, rhs);
    let lfill = get_fill(bcx, lhs);
    let rfill = get_fill(bcx, rhs);
    let mut new_fill = Add(bcx, lfill, rfill);
    if strings { new_fill = Sub(bcx, new_fill, C_int(ccx, 1)); }
    let opaque_lhs = PointerCast(bcx, lhsptr,
                                 T_ptr(T_ptr(ccx.opaque_vec_type)));
    Call(bcx, ccx.upcalls.vec_grow,
         [opaque_lhs, new_fill]);
    // Was overwritten if we resized
    let lhs = Load(bcx, lhsptr);
    let rhs = Select(bcx, self_append, lhs, rhs);

    let lhs_data = get_dataptr(bcx, lhs, llunitty);
    let mut lhs_off = lfill;
    if strings { lhs_off = Sub(bcx, lhs_off, C_int(ccx, 1)); }
    let write_ptr = pointer_add(bcx, lhs_data, lhs_off);
    let write_ptr_ptr = do_spill_noroot(bcx, write_ptr);
    iter_vec_raw(bcx, rhs, vec_ty, rfill, {|bcx, addr, _ty|
        let write_ptr = Load(bcx, write_ptr_ptr);
        let bcx = copy_val(bcx, INIT, write_ptr,
                           load_if_immediate(bcx, addr, unit_ty), unit_ty);
        Store(bcx, InBoundsGEP(bcx, write_ptr, [C_int(ccx, 1)]),
              write_ptr_ptr);
        ret bcx;
    })
}

fn trans_append_literal(bcx: block, vptrptr: ValueRef, vec_ty: ty::t,
                        vals: [@ast::expr]) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_append_literal");
    let mut bcx = bcx, ccx = bcx.ccx();
    let elt_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let elt_llty = type_of::type_of(ccx, elt_ty);
    let elt_sz = shape::llsize_of(ccx, elt_llty);
    let scratch = base::alloca(bcx, elt_llty);
    for val in vals {
        bcx = base::trans_expr_save_in(bcx, val, scratch);
        let vptr = Load(bcx, vptrptr);
        let old_fill = get_fill(bcx, vptr);
        let new_fill = Add(bcx, old_fill, elt_sz);
        let do_grow = ICmp(bcx, lib::llvm::IntUGT, new_fill,
                           get_alloc(bcx, vptr));
        bcx = base::with_cond(bcx, do_grow) {|bcx|
            let pt = PointerCast(bcx, vptrptr,
                                 T_ptr(T_ptr(ccx.opaque_vec_type)));
            Call(bcx, ccx.upcalls.vec_grow, [pt, new_fill]);
            bcx
        };
        let vptr = Load(bcx, vptrptr);
        set_fill(bcx, vptr, new_fill);
        let targetptr = pointer_add(bcx, get_dataptr(bcx, vptr, elt_llty),
                                    old_fill);
        call_memmove(bcx, targetptr, scratch, elt_sz);
    }
    bcx
}

fn trans_add(bcx: block, vec_ty: ty::t, lhs: ValueRef,
             rhs: ValueRef, dest: dest) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_add");
    let ccx = bcx.ccx();

    if ty::get(vec_ty).struct == ty::ty_str {
        let n = Call(bcx, ccx.upcalls.str_concat, [lhs, rhs]);
        ret base::store_in_dest(bcx, n, dest);
    }

    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let llunitty = type_of::type_of(ccx, unit_ty);

    let lhs_fill = get_fill(bcx, lhs);
    let rhs_fill = get_fill(bcx, rhs);
    let new_fill = Add(bcx, lhs_fill, rhs_fill);
    let mut {bcx: bcx, val: new_vec_ptr} = alloc_raw(bcx, new_fill, new_fill);
    new_vec_ptr = PointerCast(bcx, new_vec_ptr, T_ptr(T_vec(ccx, llunitty)));

    let write_ptr_ptr = do_spill_noroot
        (bcx, get_dataptr(bcx, new_vec_ptr, llunitty));
    let copy_fn = fn@(bcx: block, addr: ValueRef,
                      _ty: ty::t) -> block {
        let ccx = bcx.ccx();
        let write_ptr = Load(bcx, write_ptr_ptr);
        let bcx = copy_val(bcx, INIT, write_ptr,
                           load_if_immediate(bcx, addr, unit_ty), unit_ty);
        Store(bcx, InBoundsGEP(bcx, write_ptr, [C_int(ccx, 1)]),
              write_ptr_ptr);
        ret bcx;
    };

    let bcx = iter_vec_raw(bcx, lhs, vec_ty, lhs_fill, copy_fn);
    let bcx = iter_vec_raw(bcx, rhs, vec_ty, rhs_fill, copy_fn);
    ret base::store_in_dest(bcx, new_vec_ptr, dest);
}

type val_and_ty_fn = fn@(block, ValueRef, ty::t) -> result;

type iter_vec_block = fn(block, ValueRef, ty::t) -> block;

fn iter_vec_raw(bcx: block, vptr: ValueRef, vec_ty: ty::t,
                fill: ValueRef, f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt("tvec::iter_vec_raw");
    let ccx = bcx.ccx();
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let llunitty = type_of::type_of(ccx, unit_ty);
    let vptr = PointerCast(bcx, vptr, T_ptr(T_vec(ccx, llunitty)));
    let data_ptr = get_dataptr(bcx, vptr, llunitty);

    // Calculate the last pointer address we want to handle.
    // FIXME: Optimize this when the size of the unit type is statically
    // known to not use pointer casts, which tend to confuse LLVM.
    let data_end_ptr = pointer_add(bcx, data_ptr, fill);

    // Now perform the iteration.
    let header_cx = sub_block(bcx, "iter_vec_loop_header");
    Br(bcx, header_cx.llbb);
    let data_ptr = Phi(header_cx, val_ty(data_ptr), [data_ptr], [bcx.llbb]);
    let not_yet_at_end =
        ICmp(header_cx, lib::llvm::IntULT, data_ptr, data_end_ptr);
    let body_cx = sub_block(header_cx, "iter_vec_loop_body");
    let next_cx = sub_block(header_cx, "iter_vec_next");
    CondBr(header_cx, not_yet_at_end, body_cx.llbb, next_cx.llbb);
    let body_cx = f(body_cx, data_ptr, unit_ty);
    AddIncomingToPhi(data_ptr, InBoundsGEP(body_cx, data_ptr,
                                           [C_int(ccx, 1)]), body_cx.llbb);
    Br(body_cx, header_cx.llbb);
    ret next_cx;
}

fn iter_vec(bcx: block, vptr: ValueRef, vec_ty: ty::t,
            f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt("tvec::iter_vec");
    let vptr = PointerCast(bcx, vptr, T_ptr(bcx.ccx().opaque_vec_type));
    ret iter_vec_raw(bcx, vptr, vec_ty, get_fill(bcx, vptr), f);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

import syntax::ast;
import driver::session::session;
import lib::llvm::{ValueRef, TypeRef};
import back::abi;
import base::{call_memmove, shared_malloc,
               INIT, copy_val, load_if_immediate, get_tydesc,
               sub_block, do_spill_noroot,
               dest, bcx_icx};
import syntax::codemap::span;
import shape::llsize_of;
import build::*;
import common::*;

fn get_fill(bcx: block, vptr: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt("tvec::get_fill");
    Load(bcx, GEPi(bcx, vptr, [0u, abi::vec_elt_fill]))
}
fn set_fill(bcx: block, vptr: ValueRef, fill: ValueRef) {
    Store(bcx, fill, GEPi(bcx, vptr, [0u, abi::vec_elt_fill]));
}
fn get_alloc(bcx: block, vptr: ValueRef) -> ValueRef {
    Load(bcx, GEPi(bcx, vptr, [0u, abi::vec_elt_alloc]))
}
fn get_dataptr(bcx: block, vptr: ValueRef, unit_ty: TypeRef)
    -> ValueRef {
    let _icx = bcx.insn_ctxt("tvec::get_dataptr");
    let ptr = GEPi(bcx, vptr, [0u, abi::vec_elt_elems]);
    PointerCast(bcx, ptr, T_ptr(unit_ty))
}

fn pointer_add(bcx: block, ptr: ValueRef, bytes: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt("tvec::pointer_add");
    let old_ty = val_ty(ptr);
    let bptr = PointerCast(bcx, ptr, T_ptr(T_i8()));
    ret PointerCast(bcx, InBoundsGEP(bcx, bptr, [bytes]), old_ty);
}

fn alloc_uniq_raw(bcx: block, fill: ValueRef, alloc: ValueRef) -> result {
    let _icx = bcx.insn_ctxt("tvec::alloc_uniq_raw");
    let ccx = bcx.ccx();
    let llvecty = ccx.opaque_vec_type;
    let vecsize = Add(bcx, alloc, llsize_of(ccx, llvecty));
    let vecptr = shared_malloc(bcx, T_ptr(llvecty), vecsize);
    Store(bcx, fill, GEPi(bcx, vecptr, [0u, abi::vec_elt_fill]));
    Store(bcx, alloc, GEPi(bcx, vecptr, [0u, abi::vec_elt_alloc]));
    ret {bcx: bcx, val: vecptr};
}

fn alloc_uniq(bcx: block, llunitty: TypeRef, elts: uint) -> result {
    let _icx = bcx.insn_ctxt("tvec::alloc_uniq");
    let ccx = bcx.ccx();
    let llvecty = T_vec(ccx, llunitty);
    let unit_sz = llsize_of(ccx, llunitty);

    let fill = Mul(bcx, C_uint(ccx, elts), unit_sz);
    let alloc = if elts < 4u { Mul(bcx, C_int(ccx, 4), unit_sz) }
                else { fill };
    let {bcx: bcx, val: vptr} = alloc_uniq_raw(bcx, fill, alloc);
    let vptr = PointerCast(bcx, vptr, T_ptr(llvecty));

    ret {bcx: bcx, val: vptr};
}

fn duplicate_uniq(bcx: block, vptr: ValueRef, vec_ty: ty::t) -> result {
    let _icx = bcx.insn_ctxt("tvec::duplicate_uniq");
    let ccx = bcx.ccx();
    let fill = get_fill(bcx, vptr);
    let size = Add(bcx, fill, llsize_of(ccx, ccx.opaque_vec_type));
    let newptr = shared_malloc(bcx, val_ty(vptr), size);
    call_memmove(bcx, newptr, vptr, size);
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    Store(bcx, fill, GEPi(bcx, newptr, [0u, abi::vec_elt_alloc]));
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

fn trans_evec(bcx: block, args: [@ast::expr],
              vst: ast::vstore, id: ast::node_id, dest: dest) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_evec");
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    if dest == base::ignore {
        for vec::each(args) {|arg|
            bcx = base::trans_expr(bcx, arg, base::ignore);
        }
        ret bcx;
    }

    let vec_ty = node_id_type(bcx, id);
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let llunitty = type_of::type_of(ccx, unit_ty);
    let unit_sz = llsize_of(ccx, llunitty);

    let mut {bcx, val, dataptr} =
        alt vst {
          ast::vstore_fixed(_) {
            // Destination should be pre-allocated for us.
            let v = alt dest {
              base::save_in(v) {
                PointerCast(bcx, v, T_ptr(llunitty))
              }
              _ {
                bcx.ccx().sess.bug("bad dest for vstore_fixed \
                                    in tvec::trans_evec");
              }
            };
            {bcx: bcx, val: v, dataptr: v}
          }
          ast::vstore_slice(_) {
            let n = vec::len(args);
            let n = C_uint(ccx, n);
            let vp = base::arrayalloca(bcx, llunitty, n);
            let len = Mul(bcx, n, unit_sz);

            let p = base::alloca(bcx, T_struct([T_ptr(llunitty),
                                                ccx.int_type]));
            Store(bcx, vp, GEPi(bcx, p, [0u, abi::slice_elt_base]));
            Store(bcx, len, GEPi(bcx, p, [0u, abi::slice_elt_len]));

            {bcx: bcx, val: p, dataptr: vp}
          }
          ast::vstore_uniq {
            let {bcx, val} = alloc_uniq(bcx, llunitty, args.len());
            add_clean_free(bcx, val, true);
            let dataptr = get_dataptr(bcx, val, llunitty);
            {bcx: bcx, val: val, dataptr: dataptr}
          }
          ast::vstore_box {
            bcx.ccx().sess.unimpl("unhandled tvec::trans_evec");
          }
        };


    // Store the individual elements.
    let mut i = 0u, temp_cleanups = [val];
    #debug("trans_evec: v: %s, dataptr: %s",
           val_str(ccx.tn, val),
           val_str(ccx.tn, dataptr));
    for vec::each(args) {|e|
        let lleltptr = InBoundsGEP(bcx, dataptr, [C_uint(ccx, i)]);
        bcx = base::trans_expr_save_in(bcx, e, lleltptr);
        add_clean_temp_mem(bcx, lleltptr, unit_ty);
        temp_cleanups += [lleltptr];
        i += 1u;
    }

    for vec::each(temp_cleanups) {|cln| revoke_clean(bcx, cln); }

    alt vst {
      ast::vstore_fixed(_) {
        // We wrote into the destination in the fixed case.
        ret bcx;
      }
      ast::vstore_slice(_) {
        ret base::store_in_dest(bcx, Load(bcx, val), dest);
      }
      _ {
        ret base::store_in_dest(bcx, val, dest);
      }
    }
}

fn trans_vstore(bcx: block, e: @ast::expr,
                v: ast::vstore, dest: dest) -> block {
    alt e.node {
      ast::expr_lit(@{node: ast::lit_str(s), span: _}) {
        ret trans_estr(bcx, s, v, dest);
      }
      ast::expr_vec(es, mutbl) {
        ret trans_evec(bcx, es, v, e.id, dest);
      }
      _ {
        bcx.sess().span_bug(e.span, "vstore on non-sequence type");
      }
    }
}

fn get_base_and_len(cx: block, v: ValueRef, e_ty: ty::t)
    -> (ValueRef, ValueRef) {

    let ccx = cx.ccx();
    let tcx = ccx.tcx;
    let vec_ty = ty::type_autoderef(tcx, e_ty);
    let unit_ty = ty::sequence_element_type(tcx, vec_ty);
    let llunitty = type_of::type_of(ccx, unit_ty);
    let unit_sz = llsize_of(ccx, llunitty);

    let vstore = alt ty::get(vec_ty).struct {
      ty::ty_estr(vst) | ty::ty_evec(_, vst) { vst }
      _ { ty::vstore_uniq }
    };

    alt vstore {
      ty::vstore_fixed(n) {
        let base = GEPi(cx, v, [0u, 0u]);
        let n = if ty::type_is_str(e_ty) { n + 1u } else { n };
        let len = Mul(cx, C_uint(ccx, n), unit_sz);
        (base, len)
      }
      ty::vstore_slice(_) {
        let base = Load(cx, GEPi(cx, v, [0u, abi::slice_elt_base]));
        let len = Load(cx, GEPi(cx, v, [0u, abi::slice_elt_len]));
        (base, len)
      }
      ty::vstore_uniq {
        let base = tvec::get_dataptr(cx, v, llunitty);
        let len = tvec::get_fill(cx, v);
        (base, len)
      }
      ty::vstore_box {
        cx.ccx().sess.unimpl("unhandled tvec::get_base_and_len");
      }
    }
}

fn trans_estr(bcx: block, s: str, vstore: ast::vstore,
              dest: dest) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_estr");
    let ccx = bcx.ccx();

    let c = alt vstore {
      ast::vstore_fixed(_)
      {
        // "hello"/_  =>  "hello"/5  =>  [i8 x 6] in llvm
        #debug("trans_estr: fixed: %s", s);
        C_postr(s)
      }

      ast::vstore_slice(_) {
        // "hello"  =>  (*i8, 6u) in llvm
        #debug("trans_estr: slice '%s'", s);
        let cs = PointerCast(bcx, C_cstr(ccx, s), T_ptr(T_i8()));
        C_struct([cs, C_uint(ccx, str::len(s) + 1u /* +1 for null */)])
      }

      ast::vstore_uniq {
        let cs = PointerCast(bcx, C_cstr(ccx, s), T_ptr(T_i8()));
        let len = C_uint(ccx, str::len(s));
        Call(bcx, ccx.upcalls.str_new_uniq, [cs, len])
      }

      ast::vstore_box {
        let cs = PointerCast(bcx, C_cstr(ccx, s), T_ptr(T_i8()));
        let len = C_uint(ccx, str::len(s));
        Call(bcx, ccx.upcalls.str_new_shared, [cs, len])
      }
    };

    #debug("trans_estr: type: %s", val_str(ccx.tn, c));
    base::store_in_dest(bcx, c, dest)
}

fn trans_append(bcx: block, vec_ty: ty::t, lhsptr: ValueRef,
                rhs: ValueRef) -> block {
    let _icx = bcx.insn_ctxt("tvec::trans_append");
    // Cast to opaque interior vector types if necessary.
    let ccx = bcx.ccx();
    let unit_ty = ty::sequence_element_type(ccx.tcx, vec_ty);
    let strings = ty::type_is_str(vec_ty);
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
    iter_vec_uniq(bcx, rhs, vec_ty, rfill, {|bcx, addr, _ty|
        let write_ptr = Load(bcx, write_ptr_ptr);
        let bcx = copy_val(bcx, INIT, write_ptr,
                           load_if_immediate(bcx, addr, unit_ty), unit_ty);
        Store(bcx, InBoundsGEP(bcx, write_ptr, [C_int(ccx, 1)]),
              write_ptr_ptr);
        bcx
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
    for vec::each(vals) {|val|
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
    let mut {bcx: bcx, val: new_vec_ptr} =
        alloc_uniq_raw(bcx, new_fill, new_fill);
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

    let bcx = iter_vec_uniq(bcx, lhs, vec_ty, lhs_fill, copy_fn);
    let bcx = iter_vec_uniq(bcx, rhs, vec_ty, rhs_fill, copy_fn);
    ret base::store_in_dest(bcx, new_vec_ptr, dest);
}

type val_and_ty_fn = fn@(block, ValueRef, ty::t) -> result;

type iter_vec_block = fn(block, ValueRef, ty::t) -> block;

fn iter_vec_raw(bcx: block, data_ptr: ValueRef, vec_ty: ty::t,
                fill: ValueRef, f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt("tvec::iter_vec_uniq");

    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);

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
                                           [C_int(bcx.ccx(), 1)]),
                     body_cx.llbb);
    Br(body_cx, header_cx.llbb);
    ret next_cx;

}

fn iter_vec_uniq(bcx: block, vptr: ValueRef, vec_ty: ty::t,
                 fill: ValueRef, f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt("tvec::iter_vec_uniq");
    let ccx = bcx.ccx();
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let llunitty = type_of::type_of(ccx, unit_ty);
    let vptr = PointerCast(bcx, vptr, T_ptr(T_vec(ccx, llunitty)));
    let data_ptr = get_dataptr(bcx, vptr, llunitty);
    iter_vec_raw(bcx, data_ptr, vec_ty, fill, f)
}

fn iter_vec(bcx: block, vptr: ValueRef, vec_ty: ty::t,
            f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt("tvec::iter_vec");
    let vptr = PointerCast(bcx, vptr, T_ptr(bcx.ccx().opaque_vec_type));
    ret iter_vec_uniq(bcx, vptr, vec_ty, get_fill(bcx, vptr), f);
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

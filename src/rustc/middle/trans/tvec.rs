import syntax::ast;
import driver::session::session;
import lib::llvm::{ValueRef, TypeRef};
import back::abi;
import base::{call_memmove,
              INIT, copy_val, load_if_immediate, get_tydesc,
              sub_block, do_spill_noroot,
              dest, non_gc_box_cast, move_val, lval_owned};
import syntax::codemap::span;
import shape::llsize_of;
import build::*;
import common::*;
import util::ppaux::ty_to_str;

// Boxed vector types are in some sense currently a "shorthand" for a box
// containing an unboxed vector. This expands a boxed vector type into such an
// expanded type. It doesn't respect mutability, but that doesn't matter at
// this point.
fn expand_boxed_vec_ty(tcx: ty::ctxt, t: ty::t) -> ty::t {
    let unit_ty = ty::sequence_element_type(tcx, t);
    let unboxed_vec_ty = ty::mk_mut_unboxed_vec(tcx, unit_ty);
    match ty::get(t).struct {
      ty::ty_estr(ty::vstore_uniq) | ty::ty_evec(_, ty::vstore_uniq) => {
        ty::mk_imm_uniq(tcx, unboxed_vec_ty)
      }
      ty::ty_estr(ty::vstore_box) | ty::ty_evec(_, ty::vstore_box) => {
        ty::mk_imm_box(tcx, unboxed_vec_ty)
      }
      _ => tcx.sess.bug(~"non boxed-vec type \
                          in tvec::expand_boxed_vec_ty")
    }
}

fn get_fill(bcx: block, vptr: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt(~"tvec::get_fill");
    Load(bcx, GEPi(bcx, vptr, ~[0u, abi::vec_elt_fill]))
}
fn set_fill(bcx: block, vptr: ValueRef, fill: ValueRef) {
    Store(bcx, fill, GEPi(bcx, vptr, ~[0u, abi::vec_elt_fill]));
}
fn get_alloc(bcx: block, vptr: ValueRef) -> ValueRef {
    Load(bcx, GEPi(bcx, vptr, ~[0u, abi::vec_elt_alloc]))
}

fn get_bodyptr(bcx: block, vptr: ValueRef) -> ValueRef {
    non_gc_box_cast(bcx, GEPi(bcx, vptr, ~[0u, abi::box_field_body]))
}

fn get_dataptr(bcx: block, vptr: ValueRef)
    -> ValueRef {
    let _icx = bcx.insn_ctxt(~"tvec::get_dataptr");
    GEPi(bcx, vptr, ~[0u, abi::vec_elt_elems, 0u])
}

fn pointer_add(bcx: block, ptr: ValueRef, bytes: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt(~"tvec::pointer_add");
    let old_ty = val_ty(ptr);
    let bptr = PointerCast(bcx, ptr, T_ptr(T_i8()));
    return PointerCast(bcx, InBoundsGEP(bcx, bptr, ~[bytes]), old_ty);
}

fn alloc_raw(bcx: block, unit_ty: ty::t,
              fill: ValueRef, alloc: ValueRef, heap: heap) -> result {
    let _icx = bcx.insn_ctxt(~"tvec::alloc_uniq");
    let ccx = bcx.ccx();

    let vecbodyty = ty::mk_mut_unboxed_vec(bcx.tcx(), unit_ty);
    let vecsize = Add(bcx, alloc, llsize_of(ccx, ccx.opaque_vec_type));

    let {bcx, box, body} =
        base::malloc_general_dyn(bcx, vecbodyty, heap, vecsize);
    Store(bcx, fill, GEPi(bcx, body, ~[0u, abi::vec_elt_fill]));
    Store(bcx, alloc, GEPi(bcx, body, ~[0u, abi::vec_elt_alloc]));
    return {bcx: bcx, val: box};
}
fn alloc_uniq_raw(bcx: block, unit_ty: ty::t,
                  fill: ValueRef, alloc: ValueRef) -> result {
    alloc_raw(bcx, unit_ty, fill, alloc, heap_exchange)
}

fn alloc_vec(bcx: block, unit_ty: ty::t, elts: uint, heap: heap) -> result {
    let _icx = bcx.insn_ctxt(~"tvec::alloc_uniq");
    let ccx = bcx.ccx();
    let llunitty = type_of::type_of(ccx, unit_ty);
    let unit_sz = llsize_of(ccx, llunitty);

    let fill = Mul(bcx, C_uint(ccx, elts), unit_sz);
    let alloc = if elts < 4u { Mul(bcx, C_int(ccx, 4), unit_sz) }
                else { fill };
    let {bcx: bcx, val: vptr} = alloc_raw(bcx, unit_ty, fill, alloc, heap);
    return {bcx: bcx, val: vptr};
}

fn duplicate_uniq(bcx: block, vptr: ValueRef, vec_ty: ty::t) -> result {
    let _icx = bcx.insn_ctxt(~"tvec::duplicate_uniq");

    let fill = get_fill(bcx, get_bodyptr(bcx, vptr));
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let {bcx, val: newptr} = alloc_uniq_raw(bcx, unit_ty, fill, fill);

    let data_ptr = get_dataptr(bcx, get_bodyptr(bcx, vptr));
    let new_data_ptr = get_dataptr(bcx, get_bodyptr(bcx, newptr));
    call_memmove(bcx, new_data_ptr, data_ptr, fill);

    let bcx = if ty::type_needs_drop(bcx.tcx(), unit_ty) {
        iter_vec_raw(bcx, new_data_ptr, vec_ty, fill, base::take_ty)
    } else { bcx };
    return rslt(bcx, newptr);
}

fn make_drop_glue_unboxed(bcx: block, vptr: ValueRef, vec_ty: ty::t) ->
   block {
    let _icx = bcx.insn_ctxt(~"tvec::make_drop_glue_unboxed");
    let tcx = bcx.tcx(), unit_ty = ty::sequence_element_type(tcx, vec_ty);
    if ty::type_needs_drop(tcx, unit_ty) {
        iter_vec_unboxed(bcx, vptr, vec_ty, base::drop_ty)
    } else { bcx }
}

enum evec_elements {
    individual_evec(~[@ast::expr]),
    repeating_evec(@ast::expr, uint)
}

fn trans_evec(bcx: block, elements: evec_elements,
              vst: ast::vstore, id: ast::node_id, dest: dest) -> block {
    let _icx = bcx.insn_ctxt(~"tvec::trans_evec");
    let ccx = bcx.ccx();
    let mut bcx = bcx;

    // Handle the ignored case.
    if dest == base::ignore {
        match elements {
            individual_evec(args) => {
                for vec::each(args) |arg| {
                    bcx = base::trans_expr(bcx, arg, base::ignore);
                }
            }
            repeating_evec(element, _) => {
                bcx = base::trans_expr(bcx, element, base::ignore);
            }
        }
        return bcx;
    }

    // Figure out the number of elements we need.
    let count;
    match elements {
        individual_evec(args) => count = args.len(),
        repeating_evec(_, len) => count = len
    }

    let vec_ty = node_id_type(bcx, id);
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let llunitty = type_of::type_of(ccx, unit_ty);
    let unit_sz = llsize_of(ccx, llunitty);

    let mut {bcx, val, dataptr} =
        match vst {
          ast::vstore_fixed(_) => {
            // Destination should be pre-allocated for us.
            let v = match dest {
              base::save_in(v) => {
                PointerCast(bcx, v, T_ptr(llunitty))
              }
              _ => {
                bcx.ccx().sess.bug(~"bad dest for vstore_fixed \
                                    in tvec::trans_evec");
              }
            };
            {bcx: bcx, val: v, dataptr: v}
          }
          ast::vstore_slice(_) => {
            // Make a fake type to use for the cleanup
            let ty = ty::mk_evec(bcx.tcx(),
                                 {ty: unit_ty, mutbl: ast::m_mutbl},
                                 ty::vstore_fixed(count));
            let llty = T_ptr(type_of::type_of(bcx.ccx(), ty));

            let n = C_uint(ccx, count);
            let vp = base::arrayalloca(bcx, llunitty, n);
            // Cast to the fake type we told cleanup to expect.
            let vp0 = BitCast(bcx, vp, llty);
            add_clean(bcx, vp0, ty);

            let len = Mul(bcx, n, unit_sz);

            let p = base::alloca(bcx, T_struct(~[T_ptr(llunitty),
                                                ccx.int_type]));
            Store(bcx, vp, GEPi(bcx, p, ~[0u, abi::slice_elt_base]));
            Store(bcx, len, GEPi(bcx, p, ~[0u, abi::slice_elt_len]));

            {bcx: bcx, val: p, dataptr: vp}
          }
          ast::vstore_uniq => {
            let {bcx, val} = alloc_vec(bcx, unit_ty, count, heap_exchange);
            add_clean_free(bcx, val, heap_exchange);
            let dataptr = get_dataptr(bcx, get_bodyptr(bcx, val));
            {bcx: bcx, val: val, dataptr: dataptr}
          }
          ast::vstore_box => {
            let {bcx, val} = alloc_vec(bcx, unit_ty, count, heap_shared);
            add_clean_free(bcx, val, heap_shared);
            let dataptr = get_dataptr(bcx, get_bodyptr(bcx, val));
            {bcx: bcx, val: val, dataptr: dataptr}
          }
        };


    // Store the individual elements.
    let mut i = 0u, temp_cleanups = ~[val];
    debug!{"trans_evec: v: %s, dataptr: %s",
           val_str(ccx.tn, val),
           val_str(ccx.tn, dataptr)};
    match elements {
        individual_evec(args) => {
            for vec::each(args) |e| {
                let lleltptr = InBoundsGEP(bcx, dataptr, ~[C_uint(ccx, i)]);
                bcx = base::trans_expr_save_in(bcx, e, lleltptr);
                add_clean_temp_mem(bcx, lleltptr, unit_ty);
                vec::push(temp_cleanups, lleltptr);
                i += 1u;
            }
        }
        repeating_evec(e, len) => {
            // We make temporary space in the hope that this will be
            // friendlier to LLVM alias analysis.
            let lltmpspace = base::alloca(bcx, llunitty);
            bcx = base::trans_expr_save_in(bcx, e, lltmpspace);
            add_clean_temp_mem(bcx, lltmpspace, unit_ty);
            vec::push(temp_cleanups, lltmpspace);
            for len.timesi |i| {
                let lleltptr = InBoundsGEP(bcx, dataptr, ~[C_uint(ccx, i)]);
                if i == len - 1 {
                    // Move the last one in.
                    bcx = move_val(bcx, INIT, lleltptr,
                                   lval_owned(bcx, lltmpspace), unit_ty);
                } else {
                    // Copy all but the last one in.
                    let llval = load_if_immediate(bcx, lltmpspace, unit_ty);
                    bcx = copy_val(bcx, INIT, lleltptr, llval, unit_ty);
                }
                add_clean_temp_mem(bcx, lleltptr, unit_ty);
                vec::push(temp_cleanups, lleltptr);
            }
        }
    }

    for vec::each(temp_cleanups) |cln| { revoke_clean(bcx, cln); }

    match vst {
      ast::vstore_fixed(_) => {
        // We wrote into the destination in the fixed case.
        return bcx;
      }
      ast::vstore_slice(_) => {
        return base::store_in_dest(bcx, Load(bcx, val), dest);
      }
      _ => {
        return base::store_in_dest(bcx, val, dest);
      }
    }
}

fn trans_vstore(bcx: block, e: @ast::expr,
                v: ast::vstore, dest: dest) -> block {
    match e.node {
      ast::expr_lit(@{node: ast::lit_str(s), span: _}) => {
        return trans_estr(bcx, s, some(v), dest);
      }
      ast::expr_vec(es, mutbl) => {
        return trans_evec(bcx, individual_evec(es), v, e.id, dest);
      }
      ast::expr_repeat(element, count_expr, mutbl) => {
        let count = ty::eval_repeat_count(bcx.tcx(), count_expr, e.span);
        return trans_evec(bcx, repeating_evec(element, count), v, e.id, dest);
      }
      _ => {
        bcx.sess().span_bug(e.span, ~"vstore on non-sequence type");
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

    let vstore = match ty::get(vec_ty).struct {
      ty::ty_estr(vst) | ty::ty_evec(_, vst) => vst,
      _ => ty::vstore_uniq
    };

    match vstore {
      ty::vstore_fixed(n) => {
        let base = GEPi(cx, v, ~[0u, 0u]);
        let n = if ty::type_is_str(e_ty) { n + 1u } else { n };
        let len = Mul(cx, C_uint(ccx, n), unit_sz);
        (base, len)
      }
      ty::vstore_slice(_) => {
        let base = Load(cx, GEPi(cx, v, ~[0u, abi::slice_elt_base]));
        let len = Load(cx, GEPi(cx, v, ~[0u, abi::slice_elt_len]));
        (base, len)
      }
      ty::vstore_uniq | ty::vstore_box => {
        debug!{"get_base_and_len: %s", val_str(ccx.tn, v)};
        let body = tvec::get_bodyptr(cx, v);
        (tvec::get_dataptr(cx, body), tvec::get_fill(cx, body))
      }
    }
}

fn trans_estr(bcx: block, s: @~str, vstore: option<ast::vstore>,
              dest: dest) -> block {
    let _icx = bcx.insn_ctxt(~"tvec::trans_estr");
    if dest == base::ignore { return bcx; }
    let ccx = bcx.ccx();

    let c = match vstore {
      some(ast::vstore_fixed(_)) => {
        // "hello"/_  =>  "hello"/5  =>  ~[i8 x 6] in llvm
        debug!{"trans_estr: fixed: %s", *s};
        C_postr(*s)
      }

      some(ast::vstore_slice(_)) | none => {
        // "hello"  =>  (*i8, 6u) in llvm
        debug!{"trans_estr: slice '%s'", *s};
        C_estr_slice(ccx, *s)
      }

      some(ast::vstore_uniq) => {
        let cs = PointerCast(bcx, C_cstr(ccx, *s), T_ptr(T_i8()));
        let len = C_uint(ccx, str::len(*s));
        let c = Call(bcx, ccx.upcalls.str_new_uniq, ~[cs, len]);
        PointerCast(bcx, c,
                    T_unique_ptr(T_unique(ccx, T_vec(ccx, T_i8()))))
      }

      some(ast::vstore_box) => {
        let cs = PointerCast(bcx, C_cstr(ccx, *s), T_ptr(T_i8()));
        let len = C_uint(ccx, str::len(*s));
        let c = Call(bcx, ccx.upcalls.str_new_shared, ~[cs, len]);
        PointerCast(bcx, c,
                    T_box_ptr(T_box(ccx, T_vec(ccx, T_i8()))))
      }
    };

    debug!{"trans_estr: type: %s", val_str(ccx.tn, c)};
    base::store_in_dest(bcx, c, dest)
}

type val_and_ty_fn = fn@(block, ValueRef, ty::t) -> result;

type iter_vec_block = fn(block, ValueRef, ty::t) -> block;

fn iter_vec_raw(bcx: block, data_ptr: ValueRef, vec_ty: ty::t,
                fill: ValueRef, f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt(~"tvec::iter_vec_raw");

    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);

    // Calculate the last pointer address we want to handle.
    // FIXME (#2536): Optimize this when the size of the unit type is
    // statically known to not use pointer casts, which tend to confuse
    // LLVM.
    let data_end_ptr = pointer_add(bcx, data_ptr, fill);

    // Now perform the iteration.
    let header_cx = sub_block(bcx, ~"iter_vec_loop_header");
    Br(bcx, header_cx.llbb);
    let data_ptr =
        Phi(header_cx, val_ty(data_ptr), ~[data_ptr], ~[bcx.llbb]);
    let not_yet_at_end =
        ICmp(header_cx, lib::llvm::IntULT, data_ptr, data_end_ptr);
    let body_cx = sub_block(header_cx, ~"iter_vec_loop_body");
    let next_cx = sub_block(header_cx, ~"iter_vec_next");
    CondBr(header_cx, not_yet_at_end, body_cx.llbb, next_cx.llbb);
    let body_cx = f(body_cx, data_ptr, unit_ty);
    AddIncomingToPhi(data_ptr, InBoundsGEP(body_cx, data_ptr,
                                           ~[C_int(bcx.ccx(), 1)]),
                     body_cx.llbb);
    Br(body_cx, header_cx.llbb);
    return next_cx;

}

fn iter_vec_uniq(bcx: block, vptr: ValueRef, vec_ty: ty::t,
                 fill: ValueRef, f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt(~"tvec::iter_vec_uniq");
    let data_ptr = get_dataptr(bcx, get_bodyptr(bcx, vptr));
    iter_vec_raw(bcx, data_ptr, vec_ty, fill, f)
}

fn iter_vec_unboxed(bcx: block, body_ptr: ValueRef, vec_ty: ty::t,
                    f: iter_vec_block) -> block {
    let _icx = bcx.insn_ctxt(~"tvec::iter_vec_unboxed");
    let fill = get_fill(bcx, body_ptr);
    let dataptr = get_dataptr(bcx, body_ptr);
    return iter_vec_raw(bcx, dataptr, vec_ty, fill, f);
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

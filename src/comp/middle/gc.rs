// Routines useful for garbage collection.

import lib::llvm::True;
import lib::llvm::llvm::ValueRef;
import trans::base::get_tydesc;
import trans::common::*;
import trans::base;
import option::none;
import str;
import driver::session::session;

import lll = lib::llvm::llvm;
import bld = trans::build;

type ctxt = @{mutable next_tydesc_num: uint};

fn mk_ctxt() -> ctxt { ret @{mutable next_tydesc_num: 0u}; }

fn add_global(ccx: @crate_ctxt, llval: ValueRef, name: str) -> ValueRef {
    let llglobal =
        str::as_buf(name,
                    {|buf|
                        lll::LLVMAddGlobal(ccx.llmod, val_ty(llval), buf)
                    });
    lll::LLVMSetInitializer(llglobal, llval);
    lll::LLVMSetGlobalConstant(llglobal, True);
    ret llglobal;
}

fn add_gc_root(cx: @block_ctxt, llval: ValueRef, ty: ty::t) -> @block_ctxt {
    let bcx = cx;
    let ccx = bcx_ccx(cx);
    if !type_is_gc_relevant(bcx_tcx(cx), ty) ||
           ty::type_has_dynamic_size(bcx_tcx(cx), ty) {
        ret bcx;
    }

    let gc_cx = bcx_ccx(cx).gc_cx;

    // FIXME (issue #839): For now, we are unconditionally zeroing out all
    // GC-relevant types. Eventually we should use typestate for this.
    bcx = base::zero_alloca(bcx, llval, ty);

    let ti = none;
    let td_r = get_tydesc(bcx, ty, false, ti);
    bcx = td_r.result.bcx;
    let lltydesc = td_r.result.val;

    let gcroot = bcx_ccx(bcx).intrinsics.get("llvm.gcroot");
    let llvalptr = bld::PointerCast(bcx, llval, T_ptr(T_ptr(T_i8())));

    alt td_r.kind {
      tk_derived {
        // It's a derived type descriptor. First, spill it.
        let lltydescptr = base::alloca(bcx, val_ty(lltydesc));

        let llderivedtydescs =
            base::llderivedtydescs_block_ctxt(bcx_fcx(bcx));
        bld::Store(llderivedtydescs, lltydesc, lltydescptr);

        let number = gc_cx.next_tydesc_num;
        gc_cx.next_tydesc_num += 1u;

        let lldestindex =
            add_global(bcx_ccx(bcx), C_struct([C_int(ccx, 0),
                                               C_uint(ccx, number)]),
                       "rust_gc_tydesc_dest_index");
        let llsrcindex =
            add_global(bcx_ccx(bcx), C_struct([C_int(ccx, 1),
                                               C_uint(ccx, number)]),
                       "rust_gc_tydesc_src_index");

        lldestindex = lll::LLVMConstPointerCast(lldestindex, T_ptr(T_i8()));
        llsrcindex = lll::LLVMConstPointerCast(llsrcindex, T_ptr(T_i8()));

        lltydescptr =
            bld::PointerCast(llderivedtydescs, lltydescptr,
                             T_ptr(T_ptr(T_i8())));

        bld::Call(llderivedtydescs, gcroot, [lltydescptr, lldestindex]);
        bld::Call(bcx, gcroot, [llvalptr, llsrcindex]);
      }
      tk_param {
        bcx_tcx(cx).sess.bug("we should never be trying to root values " +
                                 "of a type parameter");
      }
      tk_static {
        // Static type descriptor.

        let llstaticgcmeta =
            add_global(bcx_ccx(bcx), C_struct([C_int(ccx, 2), lltydesc]),
                       "rust_gc_tydesc_static_gc_meta");
        let llstaticgcmetaptr =
            lll::LLVMConstPointerCast(llstaticgcmeta, T_ptr(T_i8()));

        bld::Call(bcx, gcroot, [llvalptr, llstaticgcmetaptr]);
      }
    }

    ret bcx;
}

fn type_is_gc_relevant(cx: ty::ctxt, ty: ty::t) -> bool {
    alt ty::struct(cx, ty) {
      ty::ty_nil | ty::ty_bot | ty::ty_bool | ty::ty_int(_) |
      ty::ty_float(_) | ty::ty_uint(_) | ty::ty_str |
      ty::ty_type | ty::ty_ptr(_) | ty::ty_native(_) {
        ret false;
      }
      ty::ty_rec(fields) {
        for f in fields { if type_is_gc_relevant(cx, f.mt.ty) { ret true; } }
        ret false;
      }
      ty::ty_tup(elts) {
        for elt in elts { if type_is_gc_relevant(cx, elt) { ret true; } }
        ret false;
      }
      ty::ty_enum(did, tps) {
        let variants = ty::enum_variants(cx, did);
        for variant in *variants {
            for aty in variant.args {
                let arg_ty = ty::substitute_type_params(cx, tps, aty);
                if type_is_gc_relevant(cx, arg_ty) { ret true; }
            }
        }
        ret false;
      }
      ty::ty_vec(tm) {
        ret type_is_gc_relevant(cx, tm.ty);
      }
      ty::ty_constr(sub, _) { ret type_is_gc_relevant(cx, sub); }
      ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_fn(_) |
      ty::ty_param(_, _) | ty::ty_res(_, _, _) { ret true; }
      ty::ty_var(_) {
        fail "ty_var in type_is_gc_relevant";
      }
    }
}


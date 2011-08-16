// Routines useful for garbage collection.

import lib::llvm::llvm::ValueRef;
import middle::trans::get_tydesc;
import middle::trans_common::*;
import middle::ty;
import std::option::none;
import std::ptr;
import std::str;
import std::unsafe;

import lll = lib::llvm::llvm;

fn add_gc_root(cx: &@block_ctxt, llval: ValueRef, ty: ty::t) -> @block_ctxt {
    let bcx = cx;
    if !type_is_gc_relevant(bcx_tcx(cx), ty) { ret bcx; }

    let md_kind_name = "rusttydesc";
    let md_kind = lll::LLVMGetMDKindID(str::buf(md_kind_name),
                                       str::byte_len(md_kind_name));

    let ti = none;
    let r = get_tydesc(bcx, ty, false, ti);
    bcx = r.bcx;
    let lltydesc = r.val;

    let llmdnode =
        lll::LLVMMDNode(unsafe::reinterpret_cast(ptr::addr_of(lltydesc)), 1u);
    lll::LLVMSetMetadata(llval, md_kind, llmdnode);
    ret bcx;
}

fn type_is_gc_relevant(cx: &ty::ctxt, ty: &ty::t) -> bool {
    alt ty::struct(cx, ty) {
        ty::ty_nil. | ty::ty_bot. | ty::ty_bool. | ty::ty_int. |
        ty::ty_float. | ty::ty_uint. | ty::ty_machine(_) | ty::ty_char. |
        ty::ty_istr. | ty::ty_type. | ty::ty_native(_) | ty::ty_ptr(_) |
        ty::ty_type. | ty::ty_native(_) {
            ret false;
        }

        ty::ty_rec(fields) {
            for f in fields {
                if type_is_gc_relevant(cx, f.mt.ty) { ret true; }
            }
            ret false;
        }
        ty::ty_tup(elts) {
            for elt in elts {
                if type_is_gc_relevant(cx, elt) { ret true; }
            }
            ret false;
        }

        ty::ty_tag(did, tps) {
            let variants = ty::tag_variants(cx, did);
            for variant in variants {
                for aty in variant.args {
                    let arg_ty = ty::substitute_type_params(cx, tps, aty);
                    if type_is_gc_relevant(cx, arg_ty) {
                        ret true;
                    }
                }
            }
            ret false;
        }

        ty::ty_ivec(tm) { ret type_is_gc_relevant(cx, tm.ty); }
        ty::ty_constr(sub, _) { ret type_is_gc_relevant(cx, sub); }

        ty::ty_str. | ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_vec(_) |
        ty::ty_fn(_,_,_,_,_) | ty::ty_native_fn(_,_,_) | ty::ty_obj(_) |
        ty::ty_param(_,_) | ty::ty_res(_,_,_) { ret true; }

        ty::ty_var(_) { fail "ty_var in type_is_gc_relevant"; }
    }
}


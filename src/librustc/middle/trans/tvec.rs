// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::abi;
use lib;
use lib::llvm::{llvm, ValueRef};
use middle::lang_items::StrDupUniqFnLangItem;
use middle::trans::base;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr::{Dest, Ignore, SaveIn};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::machine::{llsize_of, nonzero_llsize_of};
use middle::trans::type_of;
use middle::ty;
use util::common::indenter;
use util::ppaux::ty_to_str;

use middle::trans::type_::Type;

use std::option::None;
use syntax::ast;
use syntax::codemap;

// Boxed vector types are in some sense currently a "shorthand" for a box
// containing an unboxed vector. This expands a boxed vector type into such an
// expanded type. It doesn't respect mutability, but that doesn't matter at
// this point.
pub fn expand_boxed_vec_ty(tcx: ty::ctxt, t: ty::t) -> ty::t {
    let unit_ty = ty::sequence_element_type(tcx, t);
    let unboxed_vec_ty = ty::mk_mut_unboxed_vec(tcx, unit_ty);
    match ty::get(t).sty {
      ty::ty_estr(ty::vstore_uniq) | ty::ty_evec(_, ty::vstore_uniq) => {
        ty::mk_imm_uniq(tcx, unboxed_vec_ty)
      }
      ty::ty_estr(ty::vstore_box) | ty::ty_evec(_, ty::vstore_box) => {
        ty::mk_imm_box(tcx, unboxed_vec_ty)
      }
      _ => tcx.sess.bug("non boxed-vec type \
                         in tvec::expand_boxed_vec_ty")
    }
}

pub fn get_fill(bcx: @mut Block, vptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::get_fill");
    Load(bcx, GEPi(bcx, vptr, [0u, abi::vec_elt_fill]))
}
pub fn set_fill(bcx: @mut Block, vptr: ValueRef, fill: ValueRef) {
    Store(bcx, fill, GEPi(bcx, vptr, [0u, abi::vec_elt_fill]));
}
pub fn get_alloc(bcx: @mut Block, vptr: ValueRef) -> ValueRef {
    Load(bcx, GEPi(bcx, vptr, [0u, abi::vec_elt_alloc]))
}

pub fn get_bodyptr(bcx: @mut Block, vptr: ValueRef, t: ty::t) -> ValueRef {
    if ty::type_contents(bcx.tcx(), t).contains_managed() {
        GEPi(bcx, vptr, [0u, abi::box_field_body])
    } else {
        vptr
    }
}

pub fn get_dataptr(bcx: @mut Block, vptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::get_dataptr");
    GEPi(bcx, vptr, [0u, abi::vec_elt_elems, 0u])
}

pub fn pointer_add(bcx: @mut Block, ptr: ValueRef, bytes: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::pointer_add");
    let old_ty = val_ty(ptr);
    let bptr = PointerCast(bcx, ptr, Type::i8p());
    return PointerCast(bcx, InBoundsGEP(bcx, bptr, [bytes]), old_ty);
}

pub fn alloc_raw(bcx: @mut Block, unit_ty: ty::t,
                 fill: ValueRef, alloc: ValueRef, heap: heap) -> Result {
    let _icx = push_ctxt("tvec::alloc_uniq");
    let ccx = bcx.ccx();

    let vecbodyty = ty::mk_mut_unboxed_vec(bcx.tcx(), unit_ty);
    let vecsize = Add(bcx, alloc, llsize_of(ccx, ccx.opaque_vec_type));

    if heap == heap_exchange {
        let Result { bcx: bcx, val: val } = malloc_raw_dyn(bcx, vecbodyty, heap_exchange, vecsize);
        Store(bcx, fill, GEPi(bcx, val, [0u, abi::vec_elt_fill]));
        Store(bcx, alloc, GEPi(bcx, val, [0u, abi::vec_elt_alloc]));
        return rslt(bcx, val);
    } else {
        let base::MallocResult {bcx, box: bx, body} =
            base::malloc_general_dyn(bcx, vecbodyty, heap, vecsize);
        Store(bcx, fill, GEPi(bcx, body, [0u, abi::vec_elt_fill]));
        Store(bcx, alloc, GEPi(bcx, body, [0u, abi::vec_elt_alloc]));
        base::maybe_set_managed_unique_rc(bcx, bx, heap);
        return rslt(bcx, bx);
    }
}

pub fn alloc_uniq_raw(bcx: @mut Block, unit_ty: ty::t,
                      fill: ValueRef, alloc: ValueRef) -> Result {
    alloc_raw(bcx, unit_ty, fill, alloc, base::heap_for_unique(bcx, unit_ty))
}

pub fn alloc_vec(bcx: @mut Block,
                 unit_ty: ty::t,
                 elts: uint,
                 heap: heap)
              -> Result {
    let _icx = push_ctxt("tvec::alloc_uniq");
    let ccx = bcx.ccx();
    let llunitty = type_of::type_of(ccx, unit_ty);
    let unit_sz = nonzero_llsize_of(ccx, llunitty);

    let fill = Mul(bcx, C_uint(ccx, elts), unit_sz);
    let alloc = if elts < 4u { Mul(bcx, C_int(ccx, 4), unit_sz) }
                else { fill };
    let Result {bcx: bcx, val: vptr} =
        alloc_raw(bcx, unit_ty, fill, alloc, heap);
    return rslt(bcx, vptr);
}

pub fn make_drop_glue_unboxed(bcx: @mut Block, vptr: ValueRef, vec_ty: ty::t) ->
   @mut Block {
    let _icx = push_ctxt("tvec::make_drop_glue_unboxed");
    let tcx = bcx.tcx();
    let unit_ty = ty::sequence_element_type(tcx, vec_ty);
    if ty::type_needs_drop(tcx, unit_ty) {
        iter_vec_unboxed(bcx, vptr, vec_ty, glue::drop_ty)
    } else { bcx }
}

pub struct VecTypes {
    vec_ty: ty::t,
    unit_ty: ty::t,
    llunit_ty: Type,
    llunit_size: ValueRef
}

impl VecTypes {
    pub fn to_str(&self, ccx: &CrateContext) -> ~str {
        format!("VecTypes \\{vec_ty={}, unit_ty={}, llunit_ty={}, llunit_size={}\\}",
             ty_to_str(ccx.tcx, self.vec_ty),
             ty_to_str(ccx.tcx, self.unit_ty),
             ccx.tn.type_to_str(self.llunit_ty),
             ccx.tn.val_to_str(self.llunit_size))
    }
}

pub fn trans_fixed_vstore(bcx: @mut Block,
                          vstore_expr: &ast::Expr,
                          content_expr: &ast::Expr,
                          dest: expr::Dest)
                       -> @mut Block {
    //!
    //
    // [...] allocates a fixed-size array and moves it around "by value".
    // In this case, it means that the caller has already given us a location
    // to store the array of the suitable size, so all we have to do is
    // generate the content.

    debug2!("trans_fixed_vstore(vstore_expr={}, dest={:?})",
           bcx.expr_to_str(vstore_expr), dest.to_str(bcx.ccx()));
    let _indenter = indenter();

    let vt = vec_types_from_expr(bcx, vstore_expr);

    return match dest {
        Ignore => write_content(bcx, &vt, vstore_expr, content_expr, dest),
        SaveIn(lldest) => {
            // lldest will have type *[T x N], but we want the type *T,
            // so use GEP to convert:
            let lldest = GEPi(bcx, lldest, [0, 0]);
            write_content(bcx, &vt, vstore_expr, content_expr, SaveIn(lldest))
        }
    };
}

pub fn trans_slice_vstore(bcx: @mut Block,
                          vstore_expr: &ast::Expr,
                          content_expr: &ast::Expr,
                          dest: expr::Dest)
                       -> @mut Block {
    //!
    //
    // &[...] allocates memory on the stack and writes the values into it,
    // returning a slice (pair of ptr, len).  &"..." is similar except that
    // the memory can be statically allocated.

    let ccx = bcx.ccx();

    debug2!("trans_slice_vstore(vstore_expr={}, dest={})",
           bcx.expr_to_str(vstore_expr), dest.to_str(ccx));
    let _indenter = indenter();

    // Handle the &"..." case:
    match content_expr.node {
        ast::ExprLit(@codemap::Spanned {node: ast::lit_str(s, _), span: _}) => {
            return trans_lit_str(bcx, content_expr, s, dest);
        }
        _ => {}
    }

    // Handle the &[...] case:
    let vt = vec_types_from_expr(bcx, vstore_expr);
    let count = elements_required(bcx, content_expr);
    debug2!("vt={}, count={:?}", vt.to_str(ccx), count);

    // Make a fixed-length backing array and allocate it on the stack.
    let llcount = C_uint(ccx, count);
    let llfixed = base::arrayalloca(bcx, vt.llunit_ty, llcount);

    // Arrange for the backing array to be cleaned up.
    let fixed_ty = ty::mk_evec(bcx.tcx(),
                               ty::mt {ty: vt.unit_ty, mutbl: ast::MutMutable},
                               ty::vstore_fixed(count));
    let llfixed_ty = type_of::type_of(bcx.ccx(), fixed_ty).ptr_to();
    let llfixed_casted = BitCast(bcx, llfixed, llfixed_ty);
    add_clean(bcx, llfixed_casted, fixed_ty);

    // Generate the content into the backing array.
    let bcx = write_content(bcx, &vt, vstore_expr,
                            content_expr, SaveIn(llfixed));

    // Finally, create the slice pair itself.
    match dest {
        Ignore => {}
        SaveIn(lldest) => {
            Store(bcx, llfixed, GEPi(bcx, lldest, [0u, abi::slice_elt_base]));
            let lllen = Mul(bcx, llcount, vt.llunit_size);
            Store(bcx, lllen, GEPi(bcx, lldest, [0u, abi::slice_elt_len]));
        }
    }

    return bcx;
}

pub fn trans_lit_str(bcx: @mut Block,
                     lit_expr: &ast::Expr,
                     str_lit: @str,
                     dest: Dest)
                  -> @mut Block {
    //!
    //
    // Literal strings translate to slices into static memory.  This is
    // different from trans_slice_vstore() above because it does need to copy
    // the content anywhere.

    debug2!("trans_lit_str(lit_expr={}, dest={})",
           bcx.expr_to_str(lit_expr),
           dest.to_str(bcx.ccx()));
    let _indenter = indenter();

    match dest {
        Ignore => bcx,
        SaveIn(lldest) => {
            unsafe {
                let bytes = str_lit.len();
                let llbytes = C_uint(bcx.ccx(), bytes);
                let llcstr = C_cstr(bcx.ccx(), str_lit);
                let llcstr = llvm::LLVMConstPointerCast(llcstr, Type::i8p().to_ref());
                Store(bcx, llcstr,
                      GEPi(bcx, lldest, [0u, abi::slice_elt_base]));
                Store(bcx, llbytes,
                      GEPi(bcx, lldest, [0u, abi::slice_elt_len]));
                bcx
            }
        }
    }
}


pub fn trans_uniq_or_managed_vstore(bcx: @mut Block, heap: heap, vstore_expr: &ast::Expr,
                                    content_expr: &ast::Expr) -> DatumBlock {
    //!
    //
    // @[...] or ~[...] (also @"..." or ~"...") allocate boxes in the
    // appropriate heap and write the array elements into them.

    debug2!("trans_uniq_or_managed_vstore(vstore_expr={}, heap={:?})",
           bcx.expr_to_str(vstore_expr), heap);
    let _indenter = indenter();

    // Handle ~"".
    match heap {
        heap_exchange => {
            match content_expr.node {
                ast::ExprLit(@codemap::Spanned {
                    node: ast::lit_str(s, _), span
                }) => {
                    let llptrval = C_cstr(bcx.ccx(), s);
                    let llptrval = PointerCast(bcx, llptrval, Type::i8p());
                    let llsizeval = C_uint(bcx.ccx(), s.len());
                    let typ = ty::mk_estr(bcx.tcx(), ty::vstore_uniq);
                    let lldestval = scratch_datum(bcx, typ, "", false);
                    let alloc_fn = langcall(bcx, Some(span), "",
                                            StrDupUniqFnLangItem);
                    let bcx = callee::trans_lang_call(
                        bcx,
                        alloc_fn,
                        [ llptrval, llsizeval ],
                        Some(expr::SaveIn(lldestval.to_ref_llval(bcx)))).bcx;
                    return DatumBlock {
                        bcx: bcx,
                        datum: lldestval
                    };
                }
                _ => {}
            }
        }
        heap_exchange_closure => fail2!("vectors use exchange_alloc"),
        heap_managed | heap_managed_unique => {}
    }

    let vt = vec_types_from_expr(bcx, vstore_expr);
    let count = elements_required(bcx, content_expr);

    let Result {bcx, val} = alloc_vec(bcx, vt.unit_ty, count, heap);

    add_clean_free(bcx, val, heap);
    let dataptr = get_dataptr(bcx, get_bodyptr(bcx, val, vt.vec_ty));

    debug2!("alloc_vec() returned val={}, dataptr={}",
           bcx.val_to_str(val), bcx.val_to_str(dataptr));

    let bcx = write_content(bcx, &vt, vstore_expr,
                            content_expr, SaveIn(dataptr));

    revoke_clean(bcx, val);

    return immediate_rvalue_bcx(bcx, val, vt.vec_ty);
}

pub fn write_content(bcx: @mut Block,
                     vt: &VecTypes,
                     vstore_expr: &ast::Expr,
                     content_expr: &ast::Expr,
                     dest: Dest)
                  -> @mut Block {
    let _icx = push_ctxt("tvec::write_content");
    let mut bcx = bcx;

    debug2!("write_content(vt={}, dest={}, vstore_expr={:?})",
           vt.to_str(bcx.ccx()),
           dest.to_str(bcx.ccx()),
           bcx.expr_to_str(vstore_expr));
    let _indenter = indenter();

    match content_expr.node {
        ast::ExprLit(@codemap::Spanned { node: ast::lit_str(s, _), _ }) => {
            match dest {
                Ignore => {
                    return bcx;
                }
                SaveIn(lldest) => {
                    let bytes = s.len();
                    let llbytes = C_uint(bcx.ccx(), bytes);
                    let llcstr = C_cstr(bcx.ccx(), s);
                    base::call_memcpy(bcx, lldest, llcstr, llbytes, 1);
                    return bcx;
                }
            }
        }
        ast::ExprVec(ref elements, _) => {
            match dest {
                Ignore => {
                    for element in elements.iter() {
                        bcx = expr::trans_into(bcx, *element, Ignore);
                    }
                }

                SaveIn(lldest) => {
                    let mut temp_cleanups = ~[];
                    for (i, element) in elements.iter().enumerate() {
                        let lleltptr = GEPi(bcx, lldest, [i]);
                        debug2!("writing index {:?} with lleltptr={:?}",
                               i, bcx.val_to_str(lleltptr));
                        bcx = expr::trans_into(bcx, *element,
                                               SaveIn(lleltptr));
                        add_clean_temp_mem(bcx, lleltptr, vt.unit_ty);
                        temp_cleanups.push(lleltptr);
                    }
                    for cleanup in temp_cleanups.iter() {
                        revoke_clean(bcx, *cleanup);
                    }
                }
            }
            return bcx;
        }
        ast::ExprRepeat(element, count_expr, _) => {
            match dest {
                Ignore => {
                    return expr::trans_into(bcx, element, Ignore);
                }
                SaveIn(lldest) => {
                    let count = ty::eval_repeat_count(&bcx.tcx(), count_expr);
                    if count == 0 {
                        return bcx;
                    }

                    // Some cleanup would be required in the case in which failure happens
                    // during a copy. But given that copy constructors are not overridable,
                    // this can only happen as a result of OOM. So we just skip out on the
                    // cleanup since things would *probably* be broken at that point anyways.

                    let elem = unpack_datum!(bcx, {
                        expr::trans_to_datum(bcx, element)
                    });

                    let next_bcx = sub_block(bcx, "expr_repeat: while next");
                    let loop_bcx = loop_scope_block(bcx, next_bcx, None, "expr_repeat", None);
                    let cond_bcx = scope_block(loop_bcx, None, "expr_repeat: loop cond");
                    let set_bcx = scope_block(loop_bcx, None, "expr_repeat: body: set");
                    let inc_bcx = scope_block(loop_bcx, None, "expr_repeat: body: inc");
                    Br(bcx, loop_bcx.llbb);

                    let loop_counter = {
                        // i = 0
                        let i = alloca(loop_bcx, bcx.ccx().int_type, "__i");
                        Store(loop_bcx, C_uint(bcx.ccx(), 0), i);

                        Br(loop_bcx, cond_bcx.llbb);
                        i
                    };

                    { // i < count
                        let lhs = Load(cond_bcx, loop_counter);
                        let rhs = C_uint(bcx.ccx(), count);
                        let cond_val = ICmp(cond_bcx, lib::llvm::IntULT, lhs, rhs);

                        CondBr(cond_bcx, cond_val, set_bcx.llbb, next_bcx.llbb);
                    }

                    { // v[i] = elem
                        let i = Load(set_bcx, loop_counter);
                        let lleltptr = InBoundsGEP(set_bcx, lldest, [i]);
                        let set_bcx = elem.copy_to(set_bcx, INIT, lleltptr);

                        Br(set_bcx, inc_bcx.llbb);
                    }

                    { // i += 1
                        let i = Load(inc_bcx, loop_counter);
                        let plusone = Add(inc_bcx, i, C_uint(bcx.ccx(), 1));
                        Store(inc_bcx, plusone, loop_counter);

                        Br(inc_bcx, cond_bcx.llbb);
                    }

                    return next_bcx;

                }
            }
        }
        _ => {
            bcx.tcx().sess.span_bug(content_expr.span,
                                    "Unexpected evec content");
        }
    }
}

pub fn vec_types_from_expr(bcx: @mut Block, vec_expr: &ast::Expr) -> VecTypes {
    let vec_ty = node_id_type(bcx, vec_expr.id);
    vec_types(bcx, vec_ty)
}

pub fn vec_types(bcx: @mut Block, vec_ty: ty::t) -> VecTypes {
    let ccx = bcx.ccx();
    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    let llunit_ty = type_of::type_of(ccx, unit_ty);
    let llunit_size = nonzero_llsize_of(ccx, llunit_ty);

    VecTypes {vec_ty: vec_ty,
              unit_ty: unit_ty,
              llunit_ty: llunit_ty,
              llunit_size: llunit_size}
}

pub fn elements_required(bcx: @mut Block, content_expr: &ast::Expr) -> uint {
    //! Figure out the number of elements we need to store this content

    match content_expr.node {
        ast::ExprLit(@codemap::Spanned { node: ast::lit_str(s, _), _ }) => {
            s.len()
        },
        ast::ExprVec(ref es, _) => es.len(),
        ast::ExprRepeat(_, count_expr, _) => {
            ty::eval_repeat_count(&bcx.tcx(), count_expr)
        }
        _ => bcx.tcx().sess.span_bug(content_expr.span,
                                     "Unexpected evec content")
    }
}

pub fn get_base_and_len(bcx: @mut Block,
                        llval: ValueRef,
                        vec_ty: ty::t) -> (ValueRef, ValueRef) {
    //!
    //
    // Converts a vector into the slice pair.  The vector should be stored in
    // `llval` which should be either immediate or by-ref as appropriate for
    // the vector type.  If you have a datum, you would probably prefer to
    // call `Datum::get_base_and_len()` which will handle any conversions for
    // you.

    let ccx = bcx.ccx();
    let vt = vec_types(bcx, vec_ty);

    let vstore = match ty::get(vt.vec_ty).sty {
      ty::ty_estr(vst) | ty::ty_evec(_, vst) => vst,
      _ => ty::vstore_uniq
    };

    match vstore {
        ty::vstore_fixed(n) => {
            let base = GEPi(bcx, llval, [0u, 0u]);
            let len = Mul(bcx, C_uint(ccx, n), vt.llunit_size);
            (base, len)
        }
        ty::vstore_slice(_) => {
            let base = Load(bcx, GEPi(bcx, llval, [0u, abi::slice_elt_base]));
            let len = Load(bcx, GEPi(bcx, llval, [0u, abi::slice_elt_len]));
            (base, len)
        }
        ty::vstore_uniq | ty::vstore_box => {
            let body = get_bodyptr(bcx, llval, vec_ty);
            (get_dataptr(bcx, body), get_fill(bcx, body))
        }
    }
}

pub type iter_vec_block<'self> = &'self fn(@mut Block, ValueRef, ty::t) -> @mut Block;

pub fn iter_vec_raw(bcx: @mut Block, data_ptr: ValueRef, vec_ty: ty::t,
                    fill: ValueRef, f: iter_vec_block) -> @mut Block {
    let _icx = push_ctxt("tvec::iter_vec_raw");

    let unit_ty = ty::sequence_element_type(bcx.tcx(), vec_ty);

    // Calculate the last pointer address we want to handle.
    // FIXME (#3729): Optimize this when the size of the unit type is
    // statically known to not use pointer casts, which tend to confuse
    // LLVM.
    let data_end_ptr = pointer_add(bcx, data_ptr, fill);

    // Now perform the iteration.
    let header_bcx = base::sub_block(bcx, "iter_vec_loop_header");
    Br(bcx, header_bcx.llbb);
    let data_ptr =
        Phi(header_bcx, val_ty(data_ptr), [data_ptr], [bcx.llbb]);
    let not_yet_at_end =
        ICmp(header_bcx, lib::llvm::IntULT, data_ptr, data_end_ptr);
    let body_bcx = base::sub_block(header_bcx, "iter_vec_loop_body");
    let next_bcx = base::sub_block(header_bcx, "iter_vec_next");
    CondBr(header_bcx, not_yet_at_end, body_bcx.llbb, next_bcx.llbb);
    let body_bcx = f(body_bcx, data_ptr, unit_ty);
    AddIncomingToPhi(data_ptr, InBoundsGEP(body_bcx, data_ptr,
                                           [C_int(bcx.ccx(), 1)]),
                     body_bcx.llbb);
    Br(body_bcx, header_bcx.llbb);
    return next_bcx;

}

pub fn iter_vec_uniq(bcx: @mut Block, vptr: ValueRef, vec_ty: ty::t,
                     fill: ValueRef, f: iter_vec_block) -> @mut Block {
    let _icx = push_ctxt("tvec::iter_vec_uniq");
    let data_ptr = get_dataptr(bcx, get_bodyptr(bcx, vptr, vec_ty));
    iter_vec_raw(bcx, data_ptr, vec_ty, fill, f)
}

pub fn iter_vec_unboxed(bcx: @mut Block, body_ptr: ValueRef, vec_ty: ty::t,
                        f: iter_vec_block) -> @mut Block {
    let _icx = push_ctxt("tvec::iter_vec_unboxed");
    let fill = get_fill(bcx, body_ptr);
    let dataptr = get_dataptr(bcx, body_ptr);
    return iter_vec_raw(bcx, dataptr, vec_ty, fill, f);
}

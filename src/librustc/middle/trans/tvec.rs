// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use back::abi;
use lib;
use lib::llvm::{llvm, ValueRef};
use middle::lang_items::StrDupUniqFnLangItem;
use middle::trans::base::*;
use middle::trans::base;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr::{Dest, Ignore, SaveIn};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::machine::{llsize_of, nonzero_llsize_of, llsize_of_alloc};
use middle::trans::type_::Type;
use middle::trans::type_of;
use middle::ty;
use util::ppaux::ty_to_str;

use syntax::ast;
use syntax::parse::token::InternedString;

pub fn get_fill(bcx: &Block, vptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::get_fill");
    Load(bcx, GEPi(bcx, vptr, [0u, abi::vec_elt_fill]))
}

pub fn get_dataptr(bcx: &Block, vptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::get_dataptr");
    GEPi(bcx, vptr, [0u, abi::vec_elt_elems, 0u])
}

pub fn pointer_add_byte(bcx: &Block, ptr: ValueRef, bytes: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::pointer_add_byte");
    let old_ty = val_ty(ptr);
    let bptr = PointerCast(bcx, ptr, Type::i8p(bcx.ccx()));
    return PointerCast(bcx, InBoundsGEP(bcx, bptr, [bytes]), old_ty);
}

pub fn make_drop_glue_unboxed<'a>(
                              bcx: &'a Block<'a>,
                              vptr: ValueRef,
                              unit_ty: ty::t)
                              -> &'a Block<'a> {
    let _icx = push_ctxt("tvec::make_drop_glue_unboxed");
    let tcx = bcx.tcx();
    if ty::type_needs_drop(tcx, unit_ty) {
        let fill = get_fill(bcx, vptr);
        let dataptr = get_dataptr(bcx, vptr);
        iter_vec_raw(bcx, dataptr, unit_ty, fill, glue::drop_ty)
    } else { bcx }
}

pub struct VecTypes {
    pub unit_ty: ty::t,
    pub llunit_ty: Type,
    pub llunit_size: ValueRef,
    pub llunit_alloc_size: u64
}

impl VecTypes {
    #[cfg(stage0)]
    pub fn to_str(&self, ccx: &CrateContext) -> String {
        format!("VecTypes \\{unit_ty={}, llunit_ty={}, \
                 llunit_size={}, llunit_alloc_size={}\\}",
                ty_to_str(ccx.tcx(), self.unit_ty),
                ccx.tn.type_to_str(self.llunit_ty),
                ccx.tn.val_to_str(self.llunit_size),
                self.llunit_alloc_size)
    }
    #[cfg(not(stage0))]
    pub fn to_str(&self, ccx: &CrateContext) -> String {
        format!("VecTypes {{unit_ty={}, llunit_ty={}, \
                 llunit_size={}, llunit_alloc_size={}}}",
                ty_to_str(ccx.tcx(), self.unit_ty),
                ccx.tn.type_to_str(self.llunit_ty),
                ccx.tn.val_to_str(self.llunit_size),
                self.llunit_alloc_size)
    }
}

pub fn trans_fixed_vstore<'a>(
                          bcx: &'a Block<'a>,
                          vstore_expr: &ast::Expr,
                          content_expr: &ast::Expr,
                          dest: expr::Dest)
                          -> &'a Block<'a> {
    //!
    //
    // [...] allocates a fixed-size array and moves it around "by value".
    // In this case, it means that the caller has already given us a location
    // to store the array of the suitable size, so all we have to do is
    // generate the content.

    debug!("trans_fixed_vstore(vstore_expr={}, dest={:?})",
           bcx.expr_to_str(vstore_expr), dest.to_str(bcx.ccx()));

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

pub fn trans_slice_vstore<'a>(
                          bcx: &'a Block<'a>,
                          vstore_expr: &ast::Expr,
                          content_expr: &ast::Expr,
                          dest: expr::Dest)
                          -> &'a Block<'a> {
    /*!
     * &[...] allocates memory on the stack and writes the values into it,
     * returning a slice (pair of ptr, len).  &"..." is similar except that
     * the memory can be statically allocated.
     */

    let fcx = bcx.fcx;
    let ccx = fcx.ccx;
    let mut bcx = bcx;

    debug!("trans_slice_vstore(vstore_expr={}, dest={})",
           bcx.expr_to_str(vstore_expr), dest.to_str(ccx));

    // Handle the &"..." case:
    match content_expr.node {
        ast::ExprLit(lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => {
                    return trans_lit_str(bcx,
                                         content_expr,
                                         s.clone(),
                                         dest)
                }
                _ => {}
            }
        }
        _ => {}
    }

    // Handle the &[...] case:
    let vt = vec_types_from_expr(bcx, vstore_expr);
    let count = elements_required(bcx, content_expr);
    debug!("vt={}, count={:?}", vt.to_str(ccx), count);

    let llcount = C_uint(ccx, count);
    let llfixed;
    if count == 0 {
        // Zero-length array: just use NULL as the data pointer
        llfixed = C_null(vt.llunit_ty.ptr_to());
    } else {
        // Make a fixed-length backing array and allocate it on the stack.
        llfixed = base::arrayalloca(bcx, vt.llunit_ty, llcount);

        // Arrange for the backing array to be cleaned up.
        let fixed_ty = ty::mk_vec(bcx.tcx(),
                                  ty::mt {ty: vt.unit_ty,
                                          mutbl: ast::MutMutable},
                                  Some(count));
        let llfixed_ty = type_of::type_of(bcx.ccx(), fixed_ty).ptr_to();
        let llfixed_casted = BitCast(bcx, llfixed, llfixed_ty);
        let cleanup_scope = cleanup::temporary_scope(bcx.tcx(), content_expr.id);
        fcx.schedule_drop_mem(cleanup_scope, llfixed_casted, fixed_ty);

        // Generate the content into the backing array.
        bcx = write_content(bcx, &vt, vstore_expr,
                            content_expr, SaveIn(llfixed));
    }

    // Finally, create the slice pair itself.
    match dest {
        Ignore => {}
        SaveIn(lldest) => {
            Store(bcx, llfixed, GEPi(bcx, lldest, [0u, abi::slice_elt_base]));
            Store(bcx, llcount, GEPi(bcx, lldest, [0u, abi::slice_elt_len]));
        }
    }

    return bcx;
}

pub fn trans_lit_str<'a>(
                     bcx: &'a Block<'a>,
                     lit_expr: &ast::Expr,
                     str_lit: InternedString,
                     dest: Dest)
                     -> &'a Block<'a> {
    /*!
     * Literal strings translate to slices into static memory.  This is
     * different from trans_slice_vstore() above because it does need to copy
     * the content anywhere.
     */

    debug!("trans_lit_str(lit_expr={}, dest={})",
           bcx.expr_to_str(lit_expr),
           dest.to_str(bcx.ccx()));

    match dest {
        Ignore => bcx,
        SaveIn(lldest) => {
            unsafe {
                let bytes = str_lit.get().len();
                let llbytes = C_uint(bcx.ccx(), bytes);
                let llcstr = C_cstr(bcx.ccx(), str_lit, false);
                let llcstr = llvm::LLVMConstPointerCast(llcstr, Type::i8p(bcx.ccx()).to_ref());
                Store(bcx, llcstr,
                      GEPi(bcx, lldest, [0u, abi::slice_elt_base]));
                Store(bcx, llbytes,
                      GEPi(bcx, lldest, [0u, abi::slice_elt_len]));
                bcx
            }
        }
    }
}


pub fn trans_uniq_vstore<'a>(bcx: &'a Block<'a>,
                             vstore_expr: &ast::Expr,
                             content_expr: &ast::Expr)
                             -> DatumBlock<'a, Expr> {
    /*!
     * ~[...] and "...".to_string() allocate boxes in the exchange heap and write
     * the array elements into them.
     */

    debug!("trans_uniq_vstore(vstore_expr={})", bcx.expr_to_str(vstore_expr));
    let fcx = bcx.fcx;
    let ccx = fcx.ccx;

    // Handle "".to_string().
    match content_expr.node {
        ast::ExprLit(lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => {
                    let llptrval = C_cstr(ccx, (*s).clone(), false);
                    let llptrval = PointerCast(bcx, llptrval, Type::i8p(ccx));
                    let llsizeval = C_uint(ccx, s.get().len());
                    let typ = ty::mk_uniq(bcx.tcx(), ty::mk_str(bcx.tcx()));
                    let lldestval = rvalue_scratch_datum(bcx,
                                                         typ,
                                                         "");
                    let alloc_fn = langcall(bcx,
                                            Some(lit.span),
                                            "",
                                            StrDupUniqFnLangItem);
                    let bcx = callee::trans_lang_call(
                        bcx,
                        alloc_fn,
                        [ llptrval, llsizeval ],
                        Some(expr::SaveIn(lldestval.val))).bcx;
                    return DatumBlock::new(bcx, lldestval).to_expr_datumblock();
                }
                _ => {}
            }
        }
        _ => {}
    }

    let vec_ty = node_id_type(bcx, vstore_expr.id);
    let vt = vec_types(bcx, ty::sequence_element_type(bcx.tcx(), vec_ty));
    let count = elements_required(bcx, content_expr);

    let llunitty = type_of::type_of(ccx, vt.unit_ty);
    let unit_sz = nonzero_llsize_of(ccx, llunitty);

    let fill = Mul(bcx, C_uint(ccx, count), unit_sz);
    let alloc = if count < 4u { Mul(bcx, C_int(ccx, 4), unit_sz) }
    else { fill };

    let vecsize = Add(bcx, alloc, llsize_of(ccx, ccx.opaque_vec_type));

    // ~[T] is not going to be changed to support alignment, since it's obsolete.
    let align = C_uint(ccx, 8);
    let Result { bcx: bcx, val: val } = malloc_raw_dyn(bcx, vec_ty, vecsize, align);
    Store(bcx, fill, GEPi(bcx, val, [0u, abi::vec_elt_fill]));
    Store(bcx, alloc, GEPi(bcx, val, [0u, abi::vec_elt_alloc]));

    // Create a temporary scope lest execution should fail while
    // constructing the vector.
    let temp_scope = fcx.push_custom_cleanup_scope();

    // FIXME: #13994: the old `Box<[T]> will not support sized deallocation, this is a placeholder
    let content_ty = vt.unit_ty;
    fcx.schedule_free_value(cleanup::CustomScope(temp_scope),
                            val, cleanup::HeapExchange, content_ty);

    let dataptr = get_dataptr(bcx, val);

    debug!("alloc_uniq_vec() returned val={}, dataptr={}",
           bcx.val_to_str(val), bcx.val_to_str(dataptr));

    let bcx = write_content(bcx, &vt, vstore_expr,
                            content_expr, SaveIn(dataptr));

    fcx.pop_custom_cleanup_scope(temp_scope);

    immediate_rvalue_bcx(bcx, val, vec_ty).to_expr_datumblock()
}

pub fn write_content<'a>(
                     bcx: &'a Block<'a>,
                     vt: &VecTypes,
                     vstore_expr: &ast::Expr,
                     content_expr: &ast::Expr,
                     dest: Dest)
                     -> &'a Block<'a> {
    let _icx = push_ctxt("tvec::write_content");
    let fcx = bcx.fcx;
    let mut bcx = bcx;

    debug!("write_content(vt={}, dest={}, vstore_expr={:?})",
           vt.to_str(bcx.ccx()),
           dest.to_str(bcx.ccx()),
           bcx.expr_to_str(vstore_expr));

    match content_expr.node {
        ast::ExprLit(lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => {
                    match dest {
                        Ignore => return bcx,
                        SaveIn(lldest) => {
                            let bytes = s.get().len();
                            let llbytes = C_uint(bcx.ccx(), bytes);
                            let llcstr = C_cstr(bcx.ccx(), (*s).clone(), false);
                            base::call_memcpy(bcx,
                                              lldest,
                                              llcstr,
                                              llbytes,
                                              1);
                            return bcx;
                        }
                    }
                }
                _ => {
                    bcx.tcx().sess.span_bug(content_expr.span,
                                            "unexpected evec content");
                }
            }
        }
        ast::ExprVec(ref elements) => {
            match dest {
                Ignore => {
                    for element in elements.iter() {
                        bcx = expr::trans_into(bcx, &**element, Ignore);
                    }
                }

                SaveIn(lldest) => {
                    let temp_scope = fcx.push_custom_cleanup_scope();
                    for (i, element) in elements.iter().enumerate() {
                        let lleltptr = GEPi(bcx, lldest, [i]);
                        debug!("writing index {:?} with lleltptr={:?}",
                               i, bcx.val_to_str(lleltptr));
                        bcx = expr::trans_into(bcx, &**element,
                                               SaveIn(lleltptr));
                        fcx.schedule_drop_mem(
                            cleanup::CustomScope(temp_scope),
                            lleltptr,
                            vt.unit_ty);
                    }
                    fcx.pop_custom_cleanup_scope(temp_scope);
                }
            }
            return bcx;
        }
        ast::ExprRepeat(ref element, ref count_expr) => {
            match dest {
                Ignore => {
                    return expr::trans_into(bcx, &**element, Ignore);
                }
                SaveIn(lldest) => {
                    let count = ty::eval_repeat_count(bcx.tcx(), &**count_expr);
                    if count == 0 {
                        return bcx;
                    }

                    // Some cleanup would be required in the case in which failure happens
                    // during a copy. But given that copy constructors are not overridable,
                    // this can only happen as a result of OOM. So we just skip out on the
                    // cleanup since things would *probably* be broken at that point anyways.

                    let elem = unpack_datum!(bcx, expr::trans(bcx, &**element));
                    assert!(!ty::type_moves_by_default(bcx.tcx(), elem.ty));

                    let bcx = iter_vec_loop(bcx, lldest, vt,
                                  C_uint(bcx.ccx(), count), |set_bcx, lleltptr, _| {
                        elem.shallow_copy_and_take(set_bcx, lleltptr)
                    });

                    elem.add_clean_if_rvalue(bcx, element.id);
                    bcx
                }
            }
        }
        _ => {
            bcx.tcx().sess.span_bug(content_expr.span,
                                    "unexpected vec content");
        }
    }
}

pub fn vec_types_from_expr(bcx: &Block, vec_expr: &ast::Expr) -> VecTypes {
    let vec_ty = node_id_type(bcx, vec_expr.id);
    vec_types(bcx, ty::sequence_element_type(bcx.tcx(), vec_ty))
}

pub fn vec_types(bcx: &Block, unit_ty: ty::t) -> VecTypes {
    let ccx = bcx.ccx();
    let llunit_ty = type_of::type_of(ccx, unit_ty);
    let llunit_size = nonzero_llsize_of(ccx, llunit_ty);
    let llunit_alloc_size = llsize_of_alloc(ccx, llunit_ty);

    VecTypes {
        unit_ty: unit_ty,
        llunit_ty: llunit_ty,
        llunit_size: llunit_size,
        llunit_alloc_size: llunit_alloc_size
    }
}

pub fn elements_required(bcx: &Block, content_expr: &ast::Expr) -> uint {
    //! Figure out the number of elements we need to store this content

    match content_expr.node {
        ast::ExprLit(lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => s.get().len(),
                _ => {
                    bcx.tcx().sess.span_bug(content_expr.span,
                                            "unexpected evec content")
                }
            }
        },
        ast::ExprVec(ref es) => es.len(),
        ast::ExprRepeat(_, ref count_expr) => {
            ty::eval_repeat_count(bcx.tcx(), &**count_expr)
        }
        _ => bcx.tcx().sess.span_bug(content_expr.span,
                                     "unexpected vec content")
    }
}

pub fn get_fixed_base_and_byte_len(bcx: &Block,
                                   llval: ValueRef,
                                   unit_ty: ty::t,
                                   vec_length: uint)
                                   -> (ValueRef, ValueRef) {
    /*!
     * Converts a fixed-length vector into the slice pair.
     * The vector should be stored in `llval` which should be by ref.
     */

    let ccx = bcx.ccx();
    let vt = vec_types(bcx, unit_ty);

    let base = GEPi(bcx, llval, [0u, 0u]);
    let len = Mul(bcx, C_uint(ccx, vec_length), vt.llunit_size);
    (base, len)
}

pub fn get_base_and_len(bcx: &Block,
                        llval: ValueRef,
                        vec_ty: ty::t)
                        -> (ValueRef, ValueRef) {
    /*!
     * Converts a vector into the slice pair.  The vector should be
     * stored in `llval` which should be by-reference.  If you have a
     * datum, you would probably prefer to call
     * `Datum::get_base_and_len()` which will handle any conversions
     * for you.
     */

    let ccx = bcx.ccx();

    match ty::get(vec_ty).sty {
        ty::ty_vec(_, Some(n)) => {
            let base = GEPi(bcx, llval, [0u, 0u]);
            (base, C_uint(ccx, n))
        }
        ty::ty_rptr(_, mt) => match ty::get(mt.ty).sty {
            ty::ty_vec(_, None) | ty::ty_str => {
                assert!(!type_is_immediate(bcx.ccx(), vec_ty));
                let base = Load(bcx, GEPi(bcx, llval, [0u, abi::slice_elt_base]));
                let count = Load(bcx, GEPi(bcx, llval, [0u, abi::slice_elt_len]));
                (base, count)
            }
            _ => ccx.sess().bug("unexpected type (ty_rptr) in get_base_and_len"),
        },
        ty::ty_uniq(t) => match ty::get(t).sty {
            ty::ty_vec(_, None) | ty::ty_str => {
                assert!(type_is_immediate(bcx.ccx(), vec_ty));
                let vt = vec_types(bcx, ty::sequence_element_type(bcx.tcx(), vec_ty));
                let body = Load(bcx, llval);
                (get_dataptr(bcx, body), UDiv(bcx, get_fill(bcx, body), vt.llunit_size))
            }
            _ => ccx.sess().bug("unexpected type (ty_uniq) in get_base_and_len"),
        },
        _ => ccx.sess().bug("unexpected type in get_base_and_len"),
    }
}

pub type iter_vec_block<'r,'b> =
    |&'b Block<'b>, ValueRef, ty::t|: 'r -> &'b Block<'b>;

pub fn iter_vec_loop<'r,
                     'b>(
                     bcx: &'b Block<'b>,
                     data_ptr: ValueRef,
                     vt: &VecTypes,
                     count: ValueRef,
                     f: iter_vec_block<'r,'b>)
                     -> &'b Block<'b> {
    let _icx = push_ctxt("tvec::iter_vec_loop");
    let fcx = bcx.fcx;

    let next_bcx = fcx.new_temp_block("expr_repeat: while next");
    let loop_bcx = fcx.new_temp_block("expr_repeat");
    let cond_bcx = fcx.new_temp_block("expr_repeat: loop cond");
    let body_bcx = fcx.new_temp_block("expr_repeat: body: set");
    let inc_bcx = fcx.new_temp_block("expr_repeat: body: inc");
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
        let rhs = count;
        let cond_val = ICmp(cond_bcx, lib::llvm::IntULT, lhs, rhs);

        CondBr(cond_bcx, cond_val, body_bcx.llbb, next_bcx.llbb);
    }

    { // loop body
        let i = Load(body_bcx, loop_counter);
        let lleltptr = if vt.llunit_alloc_size == 0 {
            data_ptr
        } else {
            InBoundsGEP(body_bcx, data_ptr, [i])
        };
        let body_bcx = f(body_bcx, lleltptr, vt.unit_ty);

        Br(body_bcx, inc_bcx.llbb);
    }

    { // i += 1
        let i = Load(inc_bcx, loop_counter);
        let plusone = Add(inc_bcx, i, C_uint(bcx.ccx(), 1));
        Store(inc_bcx, plusone, loop_counter);

        Br(inc_bcx, cond_bcx.llbb);
    }

    next_bcx
}

pub fn iter_vec_raw<'r,
                    'b>(
                    bcx: &'b Block<'b>,
                    data_ptr: ValueRef,
                    unit_ty: ty::t,
                    fill: ValueRef,
                    f: iter_vec_block<'r,'b>)
                    -> &'b Block<'b> {
    let _icx = push_ctxt("tvec::iter_vec_raw");
    let fcx = bcx.fcx;

    let vt = vec_types(bcx, unit_ty);
    if vt.llunit_alloc_size == 0 {
        // Special-case vectors with elements of size 0  so they don't go out of bounds (#9890)
        iter_vec_loop(bcx, data_ptr, &vt, fill, f)
    } else {
        // Calculate the last pointer address we want to handle.
        // FIXME (#3729): Optimize this when the size of the unit type is
        // statically known to not use pointer casts, which tend to confuse
        // LLVM.
        let data_end_ptr = pointer_add_byte(bcx, data_ptr, fill);

        // Now perform the iteration.
        let header_bcx = fcx.new_temp_block("iter_vec_loop_header");
        Br(bcx, header_bcx.llbb);
        let data_ptr =
            Phi(header_bcx, val_ty(data_ptr), [data_ptr], [bcx.llbb]);
        let not_yet_at_end =
            ICmp(header_bcx, lib::llvm::IntULT, data_ptr, data_end_ptr);
        let body_bcx = fcx.new_temp_block("iter_vec_loop_body");
        let next_bcx = fcx.new_temp_block("iter_vec_next");
        CondBr(header_bcx, not_yet_at_end, body_bcx.llbb, next_bcx.llbb);
        let body_bcx = f(body_bcx, data_ptr, vt.unit_ty);
        AddIncomingToPhi(data_ptr, InBoundsGEP(body_bcx, data_ptr,
                                               [C_int(bcx.ccx(), 1)]),
                         body_bcx.llbb);
        Br(body_bcx, header_bcx.llbb);
        next_bcx
    }
}

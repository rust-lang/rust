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
use llvm;
use llvm::{ValueRef};
use trans::base::*;
use trans::base;
use trans::build::*;
use trans::cleanup;
use trans::cleanup::CleanupMethods;
use trans::common::*;
use trans::consts;
use trans::datum::*;
use trans::debuginfo::DebugLoc;
use trans::expr::{Dest, Ignore, SaveIn};
use trans::expr;
use trans::glue;
use trans::machine;
use trans::machine::llsize_of_alloc;
use trans::type_::Type;
use trans::type_of;
use middle::ty::{self, Ty};
use util::ppaux::ty_to_string;

use syntax::ast;
use syntax::parse::token::InternedString;

fn get_len(bcx: Block, vptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::get_lenl");
    Load(bcx, expr::get_len(bcx, vptr))
}

fn get_dataptr(bcx: Block, vptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("tvec::get_dataptr");
    Load(bcx, expr::get_dataptr(bcx, vptr))
}

pub fn make_drop_glue_unboxed<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                          vptr: ValueRef,
                                          unit_ty: Ty<'tcx>,
                                          should_deallocate: bool)
                                          -> Block<'blk, 'tcx> {
    let not_null = IsNotNull(bcx, vptr);
    with_cond(bcx, not_null, |bcx| {
        let ccx = bcx.ccx();
        let tcx = bcx.tcx();
        let _icx = push_ctxt("tvec::make_drop_glue_unboxed");

        let dataptr = get_dataptr(bcx, vptr);
        let bcx = if type_needs_drop(tcx, unit_ty) {
            let len = get_len(bcx, vptr);
            iter_vec_raw(bcx,
                         dataptr,
                         unit_ty,
                         len,
                         |bb, vv, tt| glue::drop_ty(bb, vv, tt, DebugLoc::None))
        } else {
            bcx
        };

        if should_deallocate {
            let llty = type_of::type_of(ccx, unit_ty);
            let unit_size = llsize_of_alloc(ccx, llty);
            if unit_size != 0 {
                let len = get_len(bcx, vptr);
                let not_empty = ICmp(bcx,
                                     llvm::IntNE,
                                     len,
                                     C_uint(ccx, 0_u32),
                                     DebugLoc::None);
                with_cond(bcx, not_empty, |bcx| {
                    let llalign = C_uint(ccx, machine::llalign_of_min(ccx, llty));
                    let size = Mul(bcx, C_uint(ccx, unit_size), len, DebugLoc::None);
                    glue::trans_exchange_free_dyn(bcx,
                                                  dataptr,
                                                  size,
                                                  llalign,
                                                  DebugLoc::None)
                })
            } else {
                bcx
            }
        } else {
            bcx
        }
    })
}

#[derive(Copy)]
pub struct VecTypes<'tcx> {
    pub unit_ty: Ty<'tcx>,
    pub llunit_ty: Type,
    pub llunit_alloc_size: u64
}

impl<'tcx> VecTypes<'tcx> {
    pub fn to_string<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> String {
        format!("VecTypes {{unit_ty={}, llunit_ty={}, llunit_alloc_size={}}}",
                ty_to_string(ccx.tcx(), self.unit_ty),
                ccx.tn().type_to_string(self.llunit_ty),
                self.llunit_alloc_size)
    }
}

pub fn trans_fixed_vstore<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      expr: &ast::Expr,
                                      dest: expr::Dest)
                                      -> Block<'blk, 'tcx> {
    //!
    //
    // [...] allocates a fixed-size array and moves it around "by value".
    // In this case, it means that the caller has already given us a location
    // to store the array of the suitable size, so all we have to do is
    // generate the content.

    debug!("trans_fixed_vstore(expr={}, dest={})",
           bcx.expr_to_string(expr), dest.to_string(bcx.ccx()));

    let vt = vec_types_from_expr(bcx, expr);

    return match dest {
        Ignore => write_content(bcx, &vt, expr, expr, dest),
        SaveIn(lldest) => {
            // lldest will have type *[T x N], but we want the type *T,
            // so use GEP to convert:
            let lldest = GEPi(bcx, lldest, &[0, 0]);
            write_content(bcx, &vt, expr, expr, SaveIn(lldest))
        }
    };
}

/// &[...] allocates memory on the stack and writes the values into it, returning the vector (the
/// caller must make the reference).  "..." is similar except that the memory can be statically
/// allocated and we return a reference (strings are always by-ref).
pub fn trans_slice_vec<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   slice_expr: &ast::Expr,
                                   content_expr: &ast::Expr)
                                   -> DatumBlock<'blk, 'tcx, Expr> {
    let fcx = bcx.fcx;
    let ccx = fcx.ccx;
    let mut bcx = bcx;

    debug!("trans_slice_vec(slice_expr={})",
           bcx.expr_to_string(slice_expr));

    let vec_ty = node_id_type(bcx, slice_expr.id);

    // Handle the "..." case (returns a slice since strings are always unsized):
    if let ast::ExprLit(ref lit) = content_expr.node {
        if let ast::LitStr(ref s, _) = lit.node {
            let scratch = rvalue_scratch_datum(bcx, vec_ty, "");
            bcx = trans_lit_str(bcx,
                                content_expr,
                                s.clone(),
                                SaveIn(scratch.val));
            return DatumBlock::new(bcx, scratch.to_expr_datum());
        }
    }

    // Handle the &[...] case:
    let vt = vec_types_from_expr(bcx, content_expr);
    let count = elements_required(bcx, content_expr);
    debug!("    vt={}, count={}", vt.to_string(ccx), count);

    let fixed_ty = ty::mk_vec(bcx.tcx(),
                              vt.unit_ty,
                              Some(count));
    let llfixed_ty = type_of::type_of(bcx.ccx(), fixed_ty);

    // Always create an alloca even if zero-sized, to preserve
    // the non-null invariant of the inner slice ptr
    let llfixed = base::alloca(bcx, llfixed_ty, "");

    if count > 0 {
        // Arrange for the backing array to be cleaned up.
        let cleanup_scope = cleanup::temporary_scope(bcx.tcx(), content_expr.id);
        fcx.schedule_lifetime_end(cleanup_scope, llfixed);
        fcx.schedule_drop_mem(cleanup_scope, llfixed, fixed_ty);

        // Generate the content into the backing array.
        // llfixed has type *[T x N], but we want the type *T,
        // so use GEP to convert
        bcx = write_content(bcx, &vt, slice_expr, content_expr,
                            SaveIn(GEPi(bcx, llfixed, &[0, 0])));
    };

    immediate_rvalue_bcx(bcx, llfixed, vec_ty).to_expr_datumblock()
}

/// Literal strings translate to slices into static memory.  This is different from
/// trans_slice_vstore() above because it doesn't need to copy the content anywhere.
pub fn trans_lit_str<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 lit_expr: &ast::Expr,
                                 str_lit: InternedString,
                                 dest: Dest)
                                 -> Block<'blk, 'tcx> {
    debug!("trans_lit_str(lit_expr={}, dest={})",
           bcx.expr_to_string(lit_expr),
           dest.to_string(bcx.ccx()));

    match dest {
        Ignore => bcx,
        SaveIn(lldest) => {
            let bytes = str_lit.len();
            let llbytes = C_uint(bcx.ccx(), bytes);
            let llcstr = C_cstr(bcx.ccx(), str_lit, false);
            let llcstr = consts::ptrcast(llcstr, Type::i8p(bcx.ccx()));
            Store(bcx, llcstr, GEPi(bcx, lldest, &[0, abi::FAT_PTR_ADDR]));
            Store(bcx, llbytes, GEPi(bcx, lldest, &[0, abi::FAT_PTR_EXTRA]));
            bcx
        }
    }
}

pub fn write_content<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 vt: &VecTypes<'tcx>,
                                 vstore_expr: &ast::Expr,
                                 content_expr: &ast::Expr,
                                 dest: Dest)
                                 -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("tvec::write_content");
    let fcx = bcx.fcx;
    let mut bcx = bcx;

    debug!("write_content(vt={}, dest={}, vstore_expr={})",
           vt.to_string(bcx.ccx()),
           dest.to_string(bcx.ccx()),
           bcx.expr_to_string(vstore_expr));

    match content_expr.node {
        ast::ExprLit(ref lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => {
                    match dest {
                        Ignore => return bcx,
                        SaveIn(lldest) => {
                            let bytes = s.len();
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
                    for element in elements {
                        bcx = expr::trans_into(bcx, &**element, Ignore);
                    }
                }

                SaveIn(lldest) => {
                    let temp_scope = fcx.push_custom_cleanup_scope();
                    for (i, element) in elements.iter().enumerate() {
                        let lleltptr = GEPi(bcx, lldest, &[i]);
                        debug!("writing index {} with lleltptr={}",
                               i, bcx.val_to_string(lleltptr));
                        bcx = expr::trans_into(bcx, &**element,
                                               SaveIn(lleltptr));
                        let scope = cleanup::CustomScope(temp_scope);
                        fcx.schedule_lifetime_end(scope, lleltptr);
                        fcx.schedule_drop_mem(scope, lleltptr, vt.unit_ty);
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
                    match ty::eval_repeat_count(bcx.tcx(), &**count_expr) {
                        0 => bcx,
                        1 => expr::trans_into(bcx, &**element, SaveIn(lldest)),
                        count => {
                            let elem = unpack_datum!(bcx, expr::trans(bcx, &**element));
                            let bcx = iter_vec_loop(bcx, lldest, vt,
                                                    C_uint(bcx.ccx(), count),
                                                    |set_bcx, lleltptr, _| {
                                                        elem.shallow_copy(set_bcx, lleltptr)
                                                    });
                            bcx
                        }
                    }
                }
            }
        }
        _ => {
            bcx.tcx().sess.span_bug(content_expr.span,
                                    "unexpected vec content");
        }
    }
}

pub fn vec_types_from_expr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                       vec_expr: &ast::Expr)
                                       -> VecTypes<'tcx> {
    let vec_ty = node_id_type(bcx, vec_expr.id);
    vec_types(bcx, ty::sequence_element_type(bcx.tcx(), vec_ty))
}

pub fn vec_types<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             unit_ty: Ty<'tcx>)
                             -> VecTypes<'tcx> {
    let ccx = bcx.ccx();
    let llunit_ty = type_of::type_of(ccx, unit_ty);
    let llunit_alloc_size = llsize_of_alloc(ccx, llunit_ty);

    VecTypes {
        unit_ty: unit_ty,
        llunit_ty: llunit_ty,
        llunit_alloc_size: llunit_alloc_size
    }
}

pub fn elements_required(bcx: Block, content_expr: &ast::Expr) -> uint {
    //! Figure out the number of elements we need to store this content

    match content_expr.node {
        ast::ExprLit(ref lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => s.len(),
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

/// Converts a fixed-length vector into the slice pair. The vector should be stored in `llval`
/// which should be by ref.
pub fn get_fixed_base_and_len(bcx: Block,
                              llval: ValueRef,
                              vec_length: uint)
                              -> (ValueRef, ValueRef) {
    let ccx = bcx.ccx();

    let base = expr::get_dataptr(bcx, llval);
    let len = C_uint(ccx, vec_length);
    (base, len)
}

fn get_slice_base_and_len(bcx: Block,
                          llval: ValueRef)
                          -> (ValueRef, ValueRef) {
    let base = Load(bcx, GEPi(bcx, llval, &[0, abi::FAT_PTR_ADDR]));
    let len = Load(bcx, GEPi(bcx, llval, &[0, abi::FAT_PTR_EXTRA]));
    (base, len)
}

/// Converts a vector into the slice pair.  The vector should be stored in `llval` which should be
/// by-reference.  If you have a datum, you would probably prefer to call
/// `Datum::get_base_and_len()` which will handle any conversions for you.
pub fn get_base_and_len(bcx: Block,
                        llval: ValueRef,
                        vec_ty: Ty)
                        -> (ValueRef, ValueRef) {
    let ccx = bcx.ccx();

    match vec_ty.sty {
        ty::ty_vec(_, Some(n)) => get_fixed_base_and_len(bcx, llval, n),
        ty::ty_open(ty) => match ty.sty {
            ty::ty_vec(_, None) | ty::ty_str => get_slice_base_and_len(bcx, llval),
            _ => ccx.sess().bug("unexpected type in get_base_and_len")
        },

        // Only used for pattern matching.
        ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ty, ..}) => match ty.sty {
            ty::ty_vec(_, None) | ty::ty_str => get_slice_base_and_len(bcx, llval),
            ty::ty_vec(_, Some(n)) => {
                let base = GEPi(bcx, Load(bcx, llval), &[0, 0]);
                (base, C_uint(ccx, n))
            }
            _ => ccx.sess().bug("unexpected type in get_base_and_len"),
        },
        _ => ccx.sess().bug("unexpected type in get_base_and_len"),
    }
}

pub fn iter_vec_loop<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                     data_ptr: ValueRef,
                                     vt: &VecTypes<'tcx>,
                                     count: ValueRef,
                                     f: F)
                                     -> Block<'blk, 'tcx> where
    F: FnOnce(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>,
{
    let _icx = push_ctxt("tvec::iter_vec_loop");
    let fcx = bcx.fcx;

    let loop_bcx = fcx.new_temp_block("expr_repeat");
    let next_bcx = fcx.new_temp_block("expr_repeat: next");

    Br(bcx, loop_bcx.llbb, DebugLoc::None);

    let loop_counter = Phi(loop_bcx, bcx.ccx().int_type(),
                           &[C_uint(bcx.ccx(), 0 as usize)], &[bcx.llbb]);

    let bcx = loop_bcx;

    let lleltptr = if vt.llunit_alloc_size == 0 {
        data_ptr
    } else {
        InBoundsGEP(bcx, data_ptr, &[loop_counter])
    };
    let bcx = f(bcx, lleltptr, vt.unit_ty);
    let plusone = Add(bcx, loop_counter, C_uint(bcx.ccx(), 1us), DebugLoc::None);
    AddIncomingToPhi(loop_counter, plusone, bcx.llbb);

    let cond_val = ICmp(bcx, llvm::IntULT, plusone, count, DebugLoc::None);
    CondBr(bcx, cond_val, loop_bcx.llbb, next_bcx.llbb, DebugLoc::None);

    next_bcx
}

pub fn iter_vec_raw<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                   data_ptr: ValueRef,
                                   unit_ty: Ty<'tcx>,
                                   len: ValueRef,
                                   f: F)
                                   -> Block<'blk, 'tcx> where
    F: FnOnce(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>,
{
    let _icx = push_ctxt("tvec::iter_vec_raw");
    let fcx = bcx.fcx;

    let vt = vec_types(bcx, unit_ty);

    if vt.llunit_alloc_size == 0 {
        // Special-case vectors with elements of size 0  so they don't go out of bounds (#9890)
        iter_vec_loop(bcx, data_ptr, &vt, len, f)
    } else {
        // Calculate the last pointer address we want to handle.
        let data_end_ptr = InBoundsGEP(bcx, data_ptr, &[len]);

        // Now perform the iteration.
        let header_bcx = fcx.new_temp_block("iter_vec_loop_header");
        Br(bcx, header_bcx.llbb, DebugLoc::None);
        let data_ptr =
            Phi(header_bcx, val_ty(data_ptr), &[data_ptr], &[bcx.llbb]);
        let not_yet_at_end =
            ICmp(header_bcx, llvm::IntULT, data_ptr, data_end_ptr, DebugLoc::None);
        let body_bcx = fcx.new_temp_block("iter_vec_loop_body");
        let next_bcx = fcx.new_temp_block("iter_vec_next");
        CondBr(header_bcx, not_yet_at_end, body_bcx.llbb, next_bcx.llbb, DebugLoc::None);
        let body_bcx = f(body_bcx, data_ptr, vt.unit_ty);
        AddIncomingToPhi(data_ptr, InBoundsGEP(body_bcx, data_ptr,
                                               &[C_int(bcx.ccx(), 1)]),
                         body_bcx.llbb);
        Br(body_bcx, header_bcx.llbb, DebugLoc::None);
        next_bcx
    }
}

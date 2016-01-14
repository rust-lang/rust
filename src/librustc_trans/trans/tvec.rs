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

use llvm;
use llvm::ValueRef;
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
use trans::machine::llsize_of_alloc;
use trans::type_::Type;
use trans::type_of;
use middle::ty::{self, Ty};

use rustc_front::hir;

use syntax::ast;
use syntax::parse::token::InternedString;

#[derive(Copy, Clone)]
struct VecTypes<'tcx> {
    unit_ty: Ty<'tcx>,
    llunit_ty: Type
}

impl<'tcx> VecTypes<'tcx> {
    pub fn to_string<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> String {
        format!("VecTypes {{unit_ty={}, llunit_ty={}}}",
                self.unit_ty,
                ccx.tn().type_to_string(self.llunit_ty))
    }
}

pub fn trans_fixed_vstore<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      expr: &hir::Expr,
                                      dest: expr::Dest)
                                      -> Block<'blk, 'tcx> {
    //!
    //
    // [...] allocates a fixed-size array and moves it around "by value".
    // In this case, it means that the caller has already given us a location
    // to store the array of the suitable size, so all we have to do is
    // generate the content.

    debug!("trans_fixed_vstore(expr={:?}, dest={})",
           expr, dest.to_string(bcx.ccx()));

    let vt = vec_types_from_expr(bcx, expr);

    return match dest {
        Ignore => write_content(bcx, &vt, expr, expr, dest),
        SaveIn(lldest) => {
            // lldest will have type *[T x N], but we want the type *T,
            // so use GEP to convert:
            let lldest = StructGEP(bcx, lldest, 0);
            write_content(bcx, &vt, expr, expr, SaveIn(lldest))
        }
    };
}

/// &[...] allocates memory on the stack and writes the values into it, returning the vector (the
/// caller must make the reference).  "..." is similar except that the memory can be statically
/// allocated and we return a reference (strings are always by-ref).
pub fn trans_slice_vec<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   slice_expr: &hir::Expr,
                                   content_expr: &hir::Expr)
                                   -> DatumBlock<'blk, 'tcx, Expr> {
    let fcx = bcx.fcx;
    let ccx = fcx.ccx;
    let mut bcx = bcx;

    debug!("trans_slice_vec(slice_expr={:?})",
           slice_expr);

    let vec_ty = node_id_type(bcx, slice_expr.id);

    // Handle the "..." case (returns a slice since strings are always unsized):
    if let hir::ExprLit(ref lit) = content_expr.node {
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

    let fixed_ty = bcx.tcx().mk_array(vt.unit_ty, count);

    // Always create an alloca even if zero-sized, to preserve
    // the non-null invariant of the inner slice ptr
    let llfixed;
    // Issue 30018: ensure state is initialized as dropped if necessary.
    if fcx.type_needs_drop(vt.unit_ty) {
        llfixed = base::alloc_ty_init(bcx, fixed_ty, InitAlloca::Dropped, "");
    } else {
        let uninit = InitAlloca::Uninit("fcx says vt.unit_ty is non-drop");
        llfixed = base::alloc_ty_init(bcx, fixed_ty, uninit, "");
        call_lifetime_start(bcx, llfixed);
    };

    if count > 0 {
        // Arrange for the backing array to be cleaned up.
        let cleanup_scope = cleanup::temporary_scope(bcx.tcx(), content_expr.id);
        fcx.schedule_lifetime_end(cleanup_scope, llfixed);
        fcx.schedule_drop_mem(cleanup_scope, llfixed, fixed_ty, None);

        // Generate the content into the backing array.
        // llfixed has type *[T x N], but we want the type *T,
        // so use GEP to convert
        bcx = write_content(bcx, &vt, slice_expr, content_expr,
                            SaveIn(StructGEP(bcx, llfixed, 0)));
    };

    immediate_rvalue_bcx(bcx, llfixed, vec_ty).to_expr_datumblock()
}

/// Literal strings translate to slices into static memory.  This is different from
/// trans_slice_vstore() above because it doesn't need to copy the content anywhere.
pub fn trans_lit_str<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 lit_expr: &hir::Expr,
                                 str_lit: InternedString,
                                 dest: Dest)
                                 -> Block<'blk, 'tcx> {
    debug!("trans_lit_str(lit_expr={:?}, dest={})",
           lit_expr,
           dest.to_string(bcx.ccx()));

    match dest {
        Ignore => bcx,
        SaveIn(lldest) => {
            let bytes = str_lit.len();
            let llbytes = C_uint(bcx.ccx(), bytes);
            let llcstr = C_cstr(bcx.ccx(), str_lit, false);
            let llcstr = consts::ptrcast(llcstr, Type::i8p(bcx.ccx()));
            Store(bcx, llcstr, expr::get_dataptr(bcx, lldest));
            Store(bcx, llbytes, expr::get_meta(bcx, lldest));
            bcx
        }
    }
}

fn write_content<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             vt: &VecTypes<'tcx>,
                             vstore_expr: &hir::Expr,
                             content_expr: &hir::Expr,
                             dest: Dest)
                             -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("tvec::write_content");
    let fcx = bcx.fcx;
    let mut bcx = bcx;

    debug!("write_content(vt={}, dest={}, vstore_expr={:?})",
           vt.to_string(bcx.ccx()),
           dest.to_string(bcx.ccx()),
           vstore_expr);

    match content_expr.node {
        hir::ExprLit(ref lit) => {
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
        hir::ExprVec(ref elements) => {
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
                        // Issue #30822: mark memory as dropped after running destructor
                        fcx.schedule_drop_and_fill_mem(scope, lleltptr, vt.unit_ty, None);
                    }
                    fcx.pop_custom_cleanup_scope(temp_scope);
                }
            }
            return bcx;
        }
        hir::ExprRepeat(ref element, ref count_expr) => {
            match dest {
                Ignore => {
                    return expr::trans_into(bcx, &**element, Ignore);
                }
                SaveIn(lldest) => {
                    match bcx.tcx().eval_repeat_count(&**count_expr) {
                        0 => expr::trans_into(bcx, &**element, Ignore),
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

fn vec_types_from_expr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, vec_expr: &hir::Expr)
                                   -> VecTypes<'tcx> {
    let vec_ty = node_id_type(bcx, vec_expr.id);
    vec_types(bcx, vec_ty.sequence_element_type(bcx.tcx()))
}

fn vec_types<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, unit_ty: Ty<'tcx>)
                         -> VecTypes<'tcx> {
    VecTypes {
        unit_ty: unit_ty,
        llunit_ty: type_of::type_of(bcx.ccx(), unit_ty)
    }
}

fn elements_required(bcx: Block, content_expr: &hir::Expr) -> usize {
    //! Figure out the number of elements we need to store this content

    match content_expr.node {
        hir::ExprLit(ref lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => s.len(),
                _ => {
                    bcx.tcx().sess.span_bug(content_expr.span,
                                            "unexpected evec content")
                }
            }
        },
        hir::ExprVec(ref es) => es.len(),
        hir::ExprRepeat(_, ref count_expr) => {
            bcx.tcx().eval_repeat_count(&**count_expr)
        }
        _ => bcx.tcx().sess.span_bug(content_expr.span,
                                     "unexpected vec content")
    }
}

/// Converts a fixed-length vector into the slice pair. The vector should be stored in `llval`
/// which should be by ref.
pub fn get_fixed_base_and_len(bcx: Block,
                              llval: ValueRef,
                              vec_length: usize)
                              -> (ValueRef, ValueRef) {
    let ccx = bcx.ccx();

    let base = expr::get_dataptr(bcx, llval);
    let len = C_uint(ccx, vec_length);
    (base, len)
}

/// Converts a vector into the slice pair.  The vector should be stored in `llval` which should be
/// by-reference.  If you have a datum, you would probably prefer to call
/// `Datum::get_base_and_len()` which will handle any conversions for you.
pub fn get_base_and_len<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    llval: ValueRef,
                                    vec_ty: Ty<'tcx>)
                                    -> (ValueRef, ValueRef) {
    let ccx = bcx.ccx();

    match vec_ty.sty {
        ty::TyArray(_, n) => get_fixed_base_and_len(bcx, llval, n),
        ty::TySlice(_) | ty::TyStr => {
            let base = Load(bcx, expr::get_dataptr(bcx, llval));
            let len = Load(bcx, expr::get_meta(bcx, llval));
            (base, len)
        }

        // Only used for pattern matching.
        ty::TyBox(ty) | ty::TyRef(_, ty::TypeAndMut{ty, ..}) => {
            let inner = if type_is_sized(bcx.tcx(), ty) {
                Load(bcx, llval)
            } else {
                llval
            };
            get_base_and_len(bcx, inner, ty)
        },
        _ => ccx.sess().bug("unexpected type in get_base_and_len"),
    }
}

fn iter_vec_loop<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                data_ptr: ValueRef,
                                vt: &VecTypes<'tcx>,
                                count: ValueRef,
                                f: F)
                                -> Block<'blk, 'tcx> where
    F: FnOnce(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>,
{
    let _icx = push_ctxt("tvec::iter_vec_loop");

    if bcx.unreachable.get() {
        return bcx;
    }

    let fcx = bcx.fcx;
    let loop_bcx = fcx.new_temp_block("expr_repeat");
    let next_bcx = fcx.new_temp_block("expr_repeat: next");

    Br(bcx, loop_bcx.llbb, DebugLoc::None);

    let loop_counter = Phi(loop_bcx, bcx.ccx().int_type(),
                           &[C_uint(bcx.ccx(), 0 as usize)], &[bcx.llbb]);

    let bcx = loop_bcx;

    let lleltptr = if llsize_of_alloc(bcx.ccx(), vt.llunit_ty) == 0 {
        data_ptr
    } else {
        InBoundsGEP(bcx, data_ptr, &[loop_counter])
    };
    let bcx = f(bcx, lleltptr, vt.unit_ty);
    let plusone = Add(bcx, loop_counter, C_uint(bcx.ccx(), 1usize), DebugLoc::None);
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

    if llsize_of_alloc(bcx.ccx(), vt.llunit_ty) == 0 {
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
        let body_bcx = f(body_bcx, data_ptr, unit_ty);
        AddIncomingToPhi(data_ptr, InBoundsGEP(body_bcx, data_ptr,
                                               &[C_int(bcx.ccx(), 1)]),
                         body_bcx.llbb);
        Br(body_bcx, header_bcx.llbb, DebugLoc::None);
        next_bcx
    }
}

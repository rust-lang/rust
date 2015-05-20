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
use util::ppaux::ty_to_string;

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
                ty_to_string(ccx.tcx(), self.unit_ty),
                ccx.tn().type_to_string(self.llunit_ty))
    }
}

pub fn trans_fixed_vstore<'r, 'blk, 'tcx>(bcx: &mut BlockContext<'r, 'blk, 'tcx>,
                                          expr: &ast::Expr,
                                          dest: expr::Dest)
                                          -> &'blk Block {
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
pub fn trans_slice_vec<'r, 'blk, 'tcx>
                      (&mut BlockContext { bl, ref mut fcx }: &mut BlockContext<'r, 'blk, 'tcx>,
                       slice_expr: &ast::Expr,
                       content_expr: &ast::Expr)
                       -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = &mut bl.with_fcx(fcx);
    let ccx = bcx.fcx.ccx;

    debug!("trans_slice_vec(slice_expr={})",
           bcx.expr_to_string(slice_expr));

    let vec_ty = node_id_type(bcx, slice_expr.id);

    // Handle the "..." case (returns a slice since strings are always unsized):
    if let ast::ExprLit(ref lit) = content_expr.node {
        if let ast::LitStr(ref s, _) = lit.node {
            let scratch = rvalue_scratch_datum(bcx, vec_ty, "");
            bcx.bl = trans_lit_str(bcx,
                                   content_expr,
                                   s.clone(),
                                   SaveIn(scratch.val));
            return DatumBlock::new(bcx.bl, scratch.to_expr_datum());
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
        bcx.fcx.schedule_lifetime_end(cleanup_scope, llfixed);
        bcx.fcx.schedule_drop_mem(cleanup_scope, llfixed, fixed_ty);

        // Generate the content into the backing array.
        // llfixed has type *[T x N], but we want the type *T,
        // so use GEP to convert
        let fp = GEPi(bcx, llfixed, &[0, 0]);
        bcx.bl = write_content(bcx, &vt, slice_expr, content_expr,
                               SaveIn(fp));
    };

    immediate_rvalue_bcx(bcx, llfixed, vec_ty).to_expr_datumblock()
}

/// Literal strings translate to slices into static memory.  This is different from
/// trans_slice_vstore() above because it doesn't need to copy the content anywhere.
pub fn trans_lit_str<'r, 'blk, 'tcx>(bcx: &mut BlockContext<'r, 'blk, 'tcx>,
                                     lit_expr: &ast::Expr,
                                     str_lit: InternedString,
                                     dest: Dest)
                                     -> &'blk Block {
    debug!("trans_lit_str(lit_expr={}, dest={})",
           bcx.expr_to_string(lit_expr),
           dest.to_string(bcx.ccx()));

    match dest {
        Ignore => bcx.bl,
        SaveIn(lldest) => {
            let bytes = str_lit.len();
            let llbytes = C_uint(bcx.ccx(), bytes);
            let llcstr = C_cstr(bcx.ccx(), str_lit, false);
            let llcstr = consts::ptrcast(llcstr, Type::i8p(bcx.ccx()));
            let fpa = GEPi(bcx, lldest, &[0, abi::FAT_PTR_ADDR]);
            Store(bcx, llcstr, fpa);
            let fpe = GEPi(bcx, lldest, &[0, abi::FAT_PTR_EXTRA]);
            Store(bcx, llbytes, fpe);
            bcx.bl
        }
    }
}

fn write_content<'r, 'blk, 'tcx>
                (&mut BlockContext { bl, ref mut fcx }: &mut BlockContext<'r, 'blk, 'tcx>,
                 vt: &VecTypes<'tcx>,
                 vstore_expr: &ast::Expr,
                 content_expr: &ast::Expr,
                 dest: Dest)
                 -> &'blk Block {
    let _icx = push_ctxt("tvec::write_content");
    let mut bcx = &mut bl.with_fcx(fcx);

    debug!("write_content(vt={}, dest={}, vstore_expr={})",
           vt.to_string(bcx.ccx()),
           dest.to_string(bcx.ccx()),
           bcx.expr_to_string(vstore_expr));

    match content_expr.node {
        ast::ExprLit(ref lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => {
                    match dest {
                        Ignore => return bcx.bl,
                        SaveIn(lldest) => {
                            let bytes = s.len();
                            let llbytes = C_uint(bcx.ccx(), bytes);
                            let llcstr = C_cstr(bcx.ccx(), (*s).clone(), false);
                            base::call_memcpy(bcx,
                                              lldest,
                                              llcstr,
                                              llbytes,
                                              1);
                            return bcx.bl;
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
                        bcx.bl = expr::trans_into(bcx, &**element, Ignore);
                    }
                }

                SaveIn(lldest) => {
                    let temp_scope = bcx.fcx.push_custom_cleanup_scope();
                    for (i, element) in elements.iter().enumerate() {
                        let lleltptr = GEPi(bcx, lldest, &[i]);
                        debug!("writing index {} with lleltptr={}",
                               i, bcx.val_to_string(lleltptr));
                        bcx.bl = expr::trans_into(bcx, &**element,
                                                    SaveIn(lleltptr));
                        let scope = cleanup::CustomScope(temp_scope);
                        bcx.fcx.schedule_lifetime_end(scope, lleltptr);
                        bcx.fcx.schedule_drop_mem(scope, lleltptr, vt.unit_ty);
                    }
                    bcx.fcx.pop_custom_cleanup_scope(temp_scope);
                }
            }
            return bcx.bl;
        }
        ast::ExprRepeat(ref element, ref count_expr) => {
            match dest {
                Ignore => {
                    return expr::trans_into(bcx, &**element, Ignore);
                }
                SaveIn(lldest) => {
                    match ty::eval_repeat_count(bcx.tcx(), &**count_expr) {
                        0 => expr::trans_into(bcx, &**element, Ignore),
                        1 => expr::trans_into(bcx, &**element, SaveIn(lldest)),
                        count => {
                            let elem = unpack_datum!(bcx, expr::trans(bcx, &**element));
                            let ty = C_uint(bcx.ccx(), count);
                            let bcx = iter_vec_loop(bcx, lldest, vt,
                                                    ty,
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

fn vec_types_from_expr<'r, 'blk, 'tcx>(bcx: &mut BlockContext<'r, 'blk, 'tcx>, vec_expr: &ast::Expr)
                                       -> VecTypes<'tcx> {
    let vec_ty = node_id_type(bcx, vec_expr.id);
    let ty = ty::sequence_element_type(bcx.tcx(), vec_ty);
    vec_types(bcx, ty)
}

fn vec_types<'r, 'blk, 'tcx>(bcx: &mut BlockContext<'r, 'blk, 'tcx>, unit_ty: Ty<'tcx>)
                             -> VecTypes<'tcx> {
    VecTypes {
        unit_ty: unit_ty,
        llunit_ty: type_of::type_of(bcx.ccx(), unit_ty)
    }
}

fn elements_required(bcx: &mut BlockContext, content_expr: &ast::Expr) -> usize {
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
pub fn get_fixed_base_and_len(bcx: &mut BlockContext,
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
pub fn get_base_and_len<'r, 'blk, 'tcx>(bcx: &mut BlockContext<'r, 'blk, 'tcx>,
                                        llval: ValueRef,
                                        vec_ty: Ty<'tcx>)
                                        -> (ValueRef, ValueRef) {
    let ccx = bcx.ccx();

    match vec_ty.sty {
        ty::ty_vec(_, Some(n)) => get_fixed_base_and_len(bcx, llval, n),
        ty::ty_vec(_, None) | ty::ty_str => {
            let dp = expr::get_dataptr(bcx, llval);
            let base = Load(bcx, dp);
            let dl = expr::get_len(bcx, llval);
            let len = Load(bcx, dl);
            (base, len)
        }

        // Only used for pattern matching.
        ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ty, ..}) => {
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

fn iter_vec_loop<'r, 'blk, 'tcx, F>
                (&mut BlockContext { bl, ref mut fcx }: &mut BlockContext<'r, 'blk, 'tcx>,
                 data_ptr: ValueRef,
                 vt: &VecTypes<'tcx>,
                 count: ValueRef,
                 f: F)
                 -> &'blk Block where
    F: for<'a> FnOnce(&mut BlockContext<'a, 'blk, 'tcx>, ValueRef, Ty<'tcx>) -> &'blk Block
{
    let _icx = push_ctxt("tvec::iter_vec_loop");

    if bl.unreachable.get() {
        return bl;
    }

    let loop_bcx = fcx.new_temp_block("expr_repeat");
    let next_bcx = fcx.new_temp_block("expr_repeat: next");

    let mut bcx = &mut bl.with_fcx(fcx);
    Br(bcx, loop_bcx.llbb, DebugLoc::None);

    let ty = bcx.ccx().int_type();
    let v = [C_uint(bcx.ccx(), 0 as usize)];
    let bb = [bcx.bl.llbb];
    let loop_counter = Phi(&mut loop_bcx.with_fcx(bcx.fcx), ty, &v, &bb);

    let bcx = &mut loop_bcx.with_fcx(bcx.fcx);

    let lleltptr = if llsize_of_alloc(bcx.ccx(), vt.llunit_ty) == 0 {
        data_ptr
    } else {
        InBoundsGEP(bcx, data_ptr, &[loop_counter])
    };
    let bcx = &mut f(bcx, lleltptr, vt.unit_ty).with_fcx(bcx.fcx);
    let one = C_uint(bcx.ccx(), 1usize);
    let plusone = Add(bcx, loop_counter, one, DebugLoc::None);
    AddIncomingToPhi(loop_counter, plusone, bcx.bl.llbb);

    let cond_val = ICmp(bcx, llvm::IntULT, plusone, count, DebugLoc::None);
    CondBr(bcx, cond_val, loop_bcx.llbb, next_bcx.llbb, DebugLoc::None);

    next_bcx
}

pub fn iter_vec_raw<'r, 'blk, 'tcx, F>
                   (&mut BlockContext { bl, ref mut fcx }: &mut BlockContext<'r, 'blk, 'tcx>,
                    data_ptr: ValueRef,
                    unit_ty: Ty<'tcx>,
                    len: ValueRef,
                    f: F)
                    -> &'blk Block where
    F: for<'a> FnOnce(&mut BlockContext<'a, 'blk, 'tcx>, ValueRef, Ty<'tcx>) -> &'blk Block,
{
    let _icx = push_ctxt("tvec::iter_vec_raw");

    let mut bcx = &mut bl.with_fcx(fcx);
    let vt = vec_types(bcx, unit_ty);

    if llsize_of_alloc(bcx.ccx(), vt.llunit_ty) == 0 {
        // Special-case vectors with elements of size 0  so they don't go out of bounds (#9890)
        iter_vec_loop(bcx, data_ptr, &vt, len, f)
    } else {
        // Calculate the last pointer address we want to handle.
        let data_end_ptr = InBoundsGEP(bcx, data_ptr, &[len]);

        // Now perform the iteration.
        let header_bcx = bcx.fcx.new_temp_block("iter_vec_loop_header");
        Br(bcx, header_bcx.llbb, DebugLoc::None);
        let data_ptr =
            Phi(&mut header_bcx.with_fcx(bcx.fcx), val_ty(data_ptr), &[data_ptr], &[bcx.bl.llbb]);
        let not_yet_at_end =
            ICmp(&mut header_bcx.with_fcx(bcx.fcx),
                 llvm::IntULT, data_ptr, data_end_ptr, DebugLoc::None);
        let body_bcx = bcx.fcx.new_temp_block("iter_vec_loop_body");
        let next_bcx = bcx.fcx.new_temp_block("iter_vec_next");
        CondBr(&mut header_bcx.with_fcx(bcx.fcx), not_yet_at_end,
               body_bcx.llbb, next_bcx.llbb, DebugLoc::None);
        let body_bcx = f(&mut body_bcx.with_fcx(bcx.fcx), data_ptr, unit_ty);
        let one = [C_int(bcx.ccx(), 1)];
        AddIncomingToPhi(data_ptr, InBoundsGEP(&mut body_bcx.with_fcx(bcx.fcx), data_ptr,
                                               &one),
                         body_bcx.llbb);
        Br(&mut body_bcx.with_fcx(bcx.fcx), header_bcx.llbb, DebugLoc::None);
        next_bcx
    }
}

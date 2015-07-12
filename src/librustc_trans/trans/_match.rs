// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Compilation of match statements
//!
//! ## Matching
//!
//! The basic algorithm for figuring out which body to execute is
//! straightforward. In pseudocode for an interpreter:
//! ```
//! for arm in arms {
//!     for pat in arm.pats {
//!         if pat_matches_discriminant && guard_succeeded {
//!             run_body;
//!             return;
//!         }
//!     }
//! }
//! ```
//
//! We just generate code to do this directly; see `compile_pattern`. The
//! LLVM optimizer is pretty powerful for optimizing this sort of construct,
//! so we don't need to do it ourselves; for example, it will synthesize
//! a switch instruction automatically for simple integers.
//!
//! The main problem with this algorithm in terms of optimization is
//! redundant pattern checks... but it's not clear that's a problem in
//! practice, and this code will probably be gone before anyone gets
//! around to optimizing it.
//!
//! ## Bindings
//!
//! We store information about the bound variables for each arm as part of the
//! per-arm `ArmData` struct.  There is a mapping from identifiers to
//! `BindingInfo` structs.  These structs contain the mode/id/type of the
//! binding, but they also contain an LLVM value which points at an alloca
//! called `llmatch`. For by value bindings that are Copy, we also create
//! an extra alloca that we copy the matched value to so that any changes
//! we do to our copy is not reflected in the original and vice-versa.
//! We don't do this if it's a move since the original value can't be used
//! and thus allowing us to cheat in not creating an extra alloca.
//!
//! The `llmatch` binding always stores a pointer into the value being matched
//! which points at the data for the binding.  If the value being matched has
//! type `T`, then, `llmatch` will point at an alloca of type `T*` (and hence
//! `llmatch` has type `T**`).  So, if you have a pattern like:
//!
//!    let a: A = ...;
//!    let b: B = ...;
//!    match (a, b) { (ref c, d) => { ... } }
//!
//! For `c` and `d`, we would generate allocas of type `C*` and `D*`
//! respectively.  These are called the `llmatch`.  As we match, when we come
//! up against an identifier, we store the current pointer into the
//! corresponding alloca.
//!
//! Once a pattern is completely matched, and assuming that there is no guard
//! pattern, we will branch to a block that leads to the body itself.  For any
//! by-value bindings, this block will first load the ptr from `llmatch` (the
//! one of type `D*`) and then load a second time to get the actual value (the
//! one of type `D`). For by ref bindings, the value of the local variable is
//! simply the first alloca.
//!
//! So, for the example above, we would generate a setup kind of like this:
//!
//!        +-------+
//!        | Entry |
//!        +-------+
//!            |
//!        +--------------------------------------------+
//!        | llmatch_c = (addr of first half of tuple)  |
//!        | llmatch_d = (addr of second half of tuple) |
//!        +--------------------------------------------+
//!            |
//!        +--------------------------------------+
//!        | *llbinding_d = **llmatch_d           |
//!        +--------------------------------------+
//!
//! If there is a guard, the situation is slightly different, because we must
//! execute the guard code.  Moreover, we need to do so once for each of the
//! alternatives that lead to the arm, because if the guard fails, they may
//! have different points from which to continue the search. Therefore, in that
//! case, we generate code that looks more like:
//!
//!        +-------+
//!        | Entry |
//!        +-------+
//!            |
//!        +-------------------------------------------+
//!        | llmatch_c = (addr of first half of tuple) |
//!        | llmatch_d = (addr of first half of tuple) |
//!        +-------------------------------------------+
//!            |
//!        +-------------------------------------------------+
//!        | *llbinding_d = **llmatch_d                      |
//!        | check condition                                 |
//!        | if false { goto next case }                     |
//!        | if true { goto body }                           |
//!        +-------------------------------------------------+
//!
//! The handling for the cleanups is a bit... sensitive.  Basically, the body
//! is the one that invokes `add_clean()` for each binding.  During the guard
//! evaluation, we add temporary cleanups and revoke them after the guard is
//! evaluated (it could fail, after all). Note that guards and moves are
//! just plain incompatible.
//!
//! Some relevant helper functions that manage bindings:
//! - `create_bindings_map()`
//! - `insert_lllocals()`
//!

pub use self::TransBindingMode::*;

use back::abi;
use llvm::{self, ValueRef, BasicBlockRef};
use middle::check_match::StaticInliner;
use middle::def;
use middle::expr_use_visitor as euv;
use middle::infer;
use middle::lang_items::StrEqFnLangItem;
use middle::mem_categorization as mc;
use middle::pat_util::*;
use middle::ty::{self, Disr, Ty};
use trans::adt;
use trans::base::*;
use trans::build::{And, Br, CondBr, GEPi, ICmp, InBoundsGEP, Load, Not};
use trans::build::{PointerCast, Store, Sub, Unreachable, add_comment};
use trans::callee;
use trans::cleanup::{self, CleanupMethods};
use trans::common::*;
use trans::consts::const_expr;
use trans::datum::*;
use trans::debuginfo::{self, DebugLoc, ToDebugLoc};
use trans::expr::{self, Dest};
use trans::type_of;
use session::config::NoDebugInfo;
use util::common::indenter;
use util::nodemap::FnvHashMap;

use syntax::ast;
use syntax::ast::NodeId;
use syntax::codemap::Span;
use syntax::fold::Folder;
use syntax::ptr::P;

#[derive(Clone, Copy, PartialEq)]
pub enum TransBindingMode {
    TrByCopy(/* llbinding */ ValueRef),
    TrByMove,
    TrByRef,
}

/// Information about a pattern binding:
/// - `llmatch` is a pointer to a stack slot.  The stack slot contains a
///   pointer into the value being matched.  Hence, llmatch has type `T**`
///   where `T` is the value being matched.
/// - `trmode` is the trans binding mode
/// - `id` is the node id of the binding
/// - `ty` is the Rust type of the binding
#[derive(Clone, Copy)]
pub struct BindingInfo<'tcx> {
    pub llmatch: ValueRef,
    pub trmode: TransBindingMode,
    pub id: ast::NodeId,
    pub span: Span,
    pub ty: Ty<'tcx>,
}

type BindingsMap<'tcx> = FnvHashMap<ast::Ident, BindingInfo<'tcx>>;

/// Context for each arm of a match statement.
struct ArmData<'p, 'blk, 'tcx: 'blk> {
    /// The first block of the body of this arm.
    bodycx: Block<'blk, 'tcx>,
    /// The AST representation of the arm.
    arm: &'p ast::Arm,
    /// A mapping so debug info handles inlined constants correctly.
    bindings_map: BindingsMap<'tcx>,
    /// The patterns for this arm, in a form with constants inlined.
    arm_pats: Vec<P<ast::Pat>>
}

fn extract_variant_args<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    repr: &adt::Repr<'tcx>,
                                    disr_val: ty::Disr,
                                    val: ValueRef)
                                    -> Vec<ValueRef> {
    let _icx = push_ctxt("match::extract_variant_args");
    (0..adt::num_args(repr, disr_val)).map(|i| {
        adt::trans_field_ptr(bcx, repr, val, disr_val, i)
    }).collect()
}

fn match_datum<'tcx>(val: ValueRef, left_ty: Ty<'tcx>) -> Datum<'tcx, Lvalue> {
    Datum::new(val, left_ty, Lvalue)
}

fn bind_subslice_pat(bcx: Block,
                     pat_id: ast::NodeId,
                     val: ValueRef,
                     offset_left: usize,
                     offset_right: usize) -> ValueRef {
    let _icx = push_ctxt("match::bind_subslice_pat");
    let vec_ty = node_id_type(bcx, pat_id);
    let vec_ty_contents = match vec_ty.sty {
        ty::TyBox(ty) => ty,
        ty::TyRef(_, mt) | ty::TyRawPtr(mt) => mt.ty,
        _ => vec_ty
    };
    let unit_ty = vec_ty_contents.sequence_element_type(bcx.tcx());
    let vec_datum = match_datum(val, vec_ty);
    let (base, len) = vec_datum.get_vec_base_and_len(bcx);

    let slice_begin = InBoundsGEP(bcx, base, &[C_uint(bcx.ccx(), offset_left)]);
    let slice_len_offset = C_uint(bcx.ccx(), offset_left + offset_right);
    let slice_len = Sub(bcx, len, slice_len_offset, DebugLoc::None);
    let slice_ty = bcx.tcx().mk_imm_ref(bcx.tcx().mk_region(ty::ReStatic),
                                         bcx.tcx().mk_slice(unit_ty));
    let scratch = rvalue_scratch_datum(bcx, slice_ty, "");
    Store(bcx, slice_begin,
          GEPi(bcx, scratch.val, &[0, abi::FAT_PTR_ADDR]));
    Store(bcx, slice_len, GEPi(bcx, scratch.val, &[0, abi::FAT_PTR_EXTRA]));
    scratch.val
}

fn extract_vec_elems<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 left_ty: Ty<'tcx>,
                                 before: usize,
                                 after: usize,
                                 val: ValueRef)
                                 -> Vec<ValueRef> {
    let _icx = push_ctxt("match::extract_vec_elems");
    let vec_datum = match_datum(val, left_ty);
    let (base, len) = vec_datum.get_vec_base_and_len(bcx);
    let mut elems = vec![];
    elems.extend((0..before).map(|i| GEPi(bcx, base, &[i])));
    elems.extend((0..after).rev().map(|i| {
        InBoundsGEP(bcx, base, &[
            Sub(bcx, len, C_uint(bcx.ccx(), i + 1), DebugLoc::None)
        ])
    }));
    elems
}

// Compiles a comparison between two things.
fn compare_values<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                              lhs: ValueRef,
                              rhs: ValueRef,
                              rhs_t: Ty<'tcx>,
                              debug_loc: DebugLoc)
                              -> Result<'blk, 'tcx> {
    fn compare_str<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                               lhs: ValueRef,
                               rhs: ValueRef,
                               rhs_t: Ty<'tcx>,
                               debug_loc: DebugLoc)
                               -> Result<'blk, 'tcx> {
        let did = langcall(cx,
                           None,
                           &format!("comparison of `{}`", rhs_t),
                           StrEqFnLangItem);
        let lhs_data = Load(cx, expr::get_dataptr(cx, lhs));
        let lhs_len = Load(cx, expr::get_len(cx, lhs));
        let rhs_data = Load(cx, expr::get_dataptr(cx, rhs));
        let rhs_len = Load(cx, expr::get_len(cx, rhs));
        callee::trans_lang_call(cx, did, &[lhs_data, lhs_len, rhs_data, rhs_len], None, debug_loc)
    }

    let _icx = push_ctxt("compare_values");
    if rhs_t.is_scalar() {
        let cmp = compare_scalar_types(cx, lhs, rhs, rhs_t, ast::BiEq, debug_loc);
        return Result::new(cx, cmp);
    }

    match rhs_t.sty {
        ty::TyRef(_, mt) => match mt.ty.sty {
            ty::TyStr => compare_str(cx, lhs, rhs, rhs_t, debug_loc),
            ty::TyArray(ty, _) | ty::TySlice(ty) => match ty.sty {
                ty::TyUint(ast::TyU8) => {
                    // NOTE: cast &[u8] and &[u8; N] to &str and abuse the str_eq lang item,
                    // which calls memcmp().
                    let pat_len = val_ty(rhs).element_type().array_length();
                    let ty_str_slice = cx.tcx().mk_static_str();

                    let rhs_str = alloc_ty(cx, ty_str_slice, "rhs_str");
                    Store(cx, GEPi(cx, rhs, &[0, 0]), expr::get_dataptr(cx, rhs_str));
                    Store(cx, C_uint(cx.ccx(), pat_len), expr::get_len(cx, rhs_str));

                    let lhs_str;
                    if val_ty(lhs) == val_ty(rhs) {
                        // Both the discriminant and the pattern are thin pointers
                        lhs_str = alloc_ty(cx, ty_str_slice, "lhs_str");
                        Store(cx, GEPi(cx, lhs, &[0, 0]), expr::get_dataptr(cx, lhs_str));
                        Store(cx, C_uint(cx.ccx(), pat_len), expr::get_len(cx, lhs_str));
                    }
                    else {
                        // The discriminant is a fat pointer
                        let llty_str_slice = type_of::type_of(cx.ccx(), ty_str_slice).ptr_to();
                        lhs_str = PointerCast(cx, lhs, llty_str_slice);
                    }

                    compare_str(cx, lhs_str, rhs_str, rhs_t, debug_loc)
                },
                _ => cx.sess().bug("only byte strings supported in compare_values"),
            },
            _ => cx.sess().bug("only string and byte strings supported in compare_values"),
        },
        _ => cx.sess().bug("only scalars, byte strings, and strings supported in compare_values"),
    }
}

fn disr_for_pat(tcx: &ty::ctxt, pat_id: ast::NodeId) -> Disr {
    let def = tcx.def_map.borrow().get(&pat_id).unwrap().full_def();
    if let def::DefVariant(enum_id, var_id, _) = def {
        let vinfo = tcx.enum_variant_with_id(enum_id, var_id);
        vinfo.disr_val
    } else {
        0
    }
}

fn def_for_ident_id(bcx: Block, pat_id: ast::NodeId) -> Option<def::Def> {
    bcx.tcx().def_map.borrow().get(&pat_id).map(|d| d.full_def())
}

fn compile_enum_variant<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                    discr_datum: Datum<'tcx, Lvalue>,
                                    r: &adt::Repr<'tcx>,
                                    disr: Disr,
                                    failure_dest: BasicBlockRef)
                                    -> Block<'blk, 'tcx> {
    let switch = adt::trans_switch(bcx, &r, discr_datum.to_llref());
    if let Some(switch) = switch {
        let case = adt::trans_case(bcx.ccx(), &r, disr);
        let cmp = ICmp(bcx, llvm::IntEQ, switch, case, DebugLoc::None);
        let success = bcx.fcx.new_temp_block("variant-matched");
        CondBr(bcx, cmp, success.llbb, failure_dest, DebugLoc::None);
        bcx = success;
    }
    bcx
}

fn compile_pattern<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                               discr_datum: Datum<'tcx, Lvalue>,
                               pat: &ast::Pat,
                               failure_dest: BasicBlockRef,
                               pat_bindings: &mut Vec<(ast::Ident, ValueRef)>)
                               -> Block<'blk, 'tcx> {
    let loc = DebugLoc::None;
    match pat.node {
        ast::PatWild(_) => {
            // Irrefutable
            // FIXME: Special handling for enums with zero variants?
        }
        ast::PatIdent(_, ref ident, Some(ref inner)) => {
            pat_bindings.push((ident.node, discr_datum.to_llref()));
            bcx = compile_pattern(bcx, discr_datum, &inner, failure_dest, pat_bindings);
        }
        ast::PatIdent(_, ref ident, None) => {
            match def_for_ident_id(bcx, pat.id) {
                Some(def::DefConst(..)) | Some(def::DefAssociatedConst(..)) =>
                    bcx.tcx().sess.span_bug(pat.span, "const pattern should've \
                                                       been rewritten"),
                Some(def::DefVariant(_, _, _)) => {
                    let repr = adt::represent_node(bcx, pat.id);
                    let disr = disr_for_pat(bcx.tcx(), pat.id);
                    // Check variant.
                    bcx = compile_enum_variant(bcx, discr_datum, &repr, disr, failure_dest);
                }
                Some(def::DefStruct(_)) => {
                    // Irrefutable
                }
                _ => {
                    // Misc ident; compute binding.
                    pat_bindings.push((ident.node, discr_datum.to_llref()));
                }
            }
        }
        ast::PatEnum(_, ref sub_pats) => {
            let repr = adt::represent_node(bcx, pat.id);
            let disr = disr_for_pat(bcx.tcx(), pat.id);

            // Check variant.
            bcx = compile_enum_variant(bcx, discr_datum, &repr, disr, failure_dest);

            // Check sub-patterns.
            // FIXME: Refactor to share code with PatStruct/PatTuple.
            if let &Some(ref sub_pats) = sub_pats {
                for (field_index, field_pat) in sub_pats.iter().enumerate() {
                    let mut field_datum = discr_datum.get_field(bcx, &repr, disr, field_index);
                    if !type_is_sized(bcx.tcx(), field_datum.ty) {
                        let scratch = rvalue_scratch_datum(bcx, field_datum.ty, "");
                        Store(bcx, field_datum.val, expr::get_dataptr(bcx, scratch.val));
                        let info = Load(bcx, expr::get_len(bcx, discr_datum.val));
                        Store(bcx, info, expr::get_len(bcx, scratch.val));
                        field_datum = Datum::new(scratch.val, scratch.ty, Lvalue);
                    }
                    bcx = compile_pattern(bcx, field_datum, &field_pat, failure_dest, pat_bindings);
                }
            }
        }
        ast::PatStruct(_, ref fields, _) => {
            let repr = adt::represent_node(bcx, pat.id);

            expr::with_field_tys(bcx.tcx(), discr_datum.ty, Some(pat.id), |disr, field_tys| {
                // Check variant.
                bcx = compile_enum_variant(bcx, discr_datum, &repr, disr, failure_dest);

                // Check sub-patterns.
                // FIXME: Refactor to share code with PatEnum.
                for field_pat in fields {
                    let field_name = field_pat.node.ident.name;
                    let field_index = bcx.tcx().field_idx_strict(field_name, field_tys);
                    let mut field_datum = discr_datum.get_field(bcx, &repr, disr, field_index);
                    if !type_is_sized(bcx.tcx(), field_datum.ty) {
                        let scratch = rvalue_scratch_datum(bcx, field_datum.ty, "");
                        Store(bcx, field_datum.val, expr::get_dataptr(bcx, scratch.val));
                        let info = Load(bcx, expr::get_len(bcx, discr_datum.val));
                        Store(bcx, info, expr::get_len(bcx, scratch.val));
                        field_datum = Datum::new(scratch.val, scratch.ty, Lvalue);
                    }
                    bcx = compile_pattern(bcx, field_datum, &field_pat.node.pat, failure_dest, pat_bindings);
                }
            });
        }
        ast::PatTup(ref sub_pats) => {
            let repr = adt::represent_node(bcx, pat.id);
            for (field_index, field_pat) in sub_pats.iter().enumerate() {
                let field_datum = discr_datum.get_field(bcx, &repr, 0, field_index);
                bcx = compile_pattern(bcx, field_datum, &field_pat, failure_dest, pat_bindings);
            }
        }
        ast::PatBox(ref inner) |
        ast::PatRegion(ref inner, _) => {
            // Dereference, and recurse.
            let llval = if type_is_fat_ptr(bcx.tcx(), discr_datum.ty) {
                // "Dereferencing" a fat pointer would normally copy the pointer
                // value to a stack slot, but we can skip that step here.
                discr_datum.to_llref()
            } else {
                Load(bcx, discr_datum.to_llref())
            };
            let inner_ty =  discr_datum.ty.builtin_deref(false).unwrap().ty;
            let inner_datum = Datum::new(llval, inner_ty, Lvalue);
            bcx = compile_pattern(bcx, inner_datum, &inner, failure_dest, pat_bindings);
        }
        ast::PatVec(ref before, ref slice, ref after) => {
            let fat = type_is_fat_ptr(bcx.tcx(), discr_datum.ty);
            let (base, len, elem_ty) = if fat {
                let slice_ty = discr_datum.ty.builtin_deref(false).unwrap().ty;
                let slice_datum = Datum::new(discr_datum.to_llref(), slice_ty, Lvalue);
                let (base, len) = slice_datum.get_vec_base_and_len(bcx);

                let cmp = if slice.is_some() {
                    let min_pat_len = C_uint(bcx.ccx(), before.len() + after.len());
                    ICmp(bcx, llvm::IntUGE, len, min_pat_len, DebugLoc::None)
                } else {
                    assert!(after.is_empty());
                    let pat_len = C_uint(bcx.ccx(), before.len());
                    ICmp(bcx, llvm::IntEQ, len, pat_len, DebugLoc::None)
                };
                let success = bcx.fcx.new_temp_block("slice-length-matched");
                CondBr(bcx, cmp, success.llbb, failure_dest, DebugLoc::None);
                bcx = success;

                let elem_ty = slice_ty.builtin_index().unwrap();
                (base, len, elem_ty)
            } else {
                let (elem_ty, len) = match discr_datum.ty.sty {
                    ty::TyArray(t, n) => (t, n),
                    _ => panic!("Unexpected type")
                };
                let base = GEPi(bcx, discr_datum.to_llref(), &[0, 0]);
                (base, C_uint(bcx.ccx(), len), elem_ty)
            };

            for (i, pat) in before.iter().enumerate() {
                let elem_val = GEPi(bcx, base, &[i]);
                let elem_datum = Datum::new(elem_val, elem_ty, Lvalue);
                bcx = compile_pattern(bcx, elem_datum, &pat, failure_dest, pat_bindings);
            }

            if let &Some(ref slice) = slice {
                let slice_start = GEPi(bcx, base, &[before.len()]);
                let slice_len = Sub(bcx, len, C_uint(bcx.ccx(), before.len() + after.len()), loc);
                let slice_ty = bcx.tcx().mk_slice(elem_ty);
                // FIXME: Making the slice match the address of the slice
                // is slightly crazy.
                let slice_ty = bcx.tcx().mk_imm_ref(bcx.tcx().mk_region(ty::ReStatic), slice_ty);
                let scratch = rvalue_scratch_datum(bcx, slice_ty, "");
                Store(bcx, slice_start, expr::get_dataptr(bcx, scratch.val));
                Store(bcx, slice_len, expr::get_len(bcx, scratch.val));
                let slice_datum = Datum::new(scratch.val, scratch.ty, Lvalue);
                compile_pattern(bcx, slice_datum, &slice, failure_dest, pat_bindings);
            }

            if !after.is_empty() {
                let after_start = Sub(bcx, len, C_uint(bcx.ccx(), after.len()), loc);
                let after_base = InBoundsGEP(bcx, base, &[after_start]);
                for (i, pat) in after.iter().enumerate() {
                    let elem_val = GEPi(bcx, after_base, &[i]);
                    let elem_datum = Datum::new(elem_val, elem_ty, Lvalue);
                    bcx = compile_pattern(bcx, elem_datum, &pat, failure_dest, pat_bindings);
                }
            }
        }
        ast::PatLit(ref lit_expr) => {
            // Load value to test.
            let test_val = if type_is_fat_ptr(bcx.tcx(), discr_datum.ty) {
                // "Dereferencing" a fat pointer would normally copy the pointer
                // value to a stack slot, but we can skip that step here.
                discr_datum.to_llref()
            } else {
                discr_datum.to_llscalarish(bcx)
            };
            // Compute value to compare against.
            let lit_ty = node_id_type(bcx, lit_expr.id);
            let (llval, _) = const_expr(bcx.ccx(), &lit_expr, bcx.fcx.param_substs, None);
            let lit_datum = immediate_rvalue(llval, lit_ty);
            let lit_datum = unpack_datum!(bcx, lit_datum.to_appropriate_datum(bcx));

            // Compare values.
            let Result { bcx: after_cx, val: cmp } =
                compare_values(bcx, test_val, lit_datum.val, lit_ty, loc);
            bcx = after_cx;
            let success = bcx.fcx.new_temp_block("lit-matched");
            CondBr(bcx, cmp, success.llbb, failure_dest, loc);
            bcx = success;
        }
        ast::PatRange(ref from_expr, ref to_expr) => {
            // Load value to test.
            let test_val = discr_datum.to_llscalarish(bcx);

            // Compute range to compare against.
            let (vbegin, _) = const_expr(bcx.ccx(), &from_expr, bcx.fcx.param_substs, None);
            let (vend, _) = const_expr(bcx.ccx(), &to_expr, bcx.fcx.param_substs, None);

            // Compare values.
            let range_ty = bcx.tcx().node_id_to_type(from_expr.id);
            let llge = compare_scalar_types(bcx, test_val, vbegin,
                                            range_ty, ast::BiGe, loc);
            let llle = compare_scalar_types(bcx, test_val, vend,
                                            range_ty, ast::BiLe, loc);
            let cmp = And(bcx, llge, llle, DebugLoc::None);
            let success = bcx.fcx.new_temp_block("range-matched");
            CondBr(bcx, cmp, success.llbb, failure_dest, DebugLoc::None);
            bcx = success;
        }
        ast::PatMac(..) | ast::PatQPath(..) => {
            // Should be normalized away.
            unreachable!()
        }
    }
    bcx
}

/// For each binding in `data.bindings_map`, adds an appropriate entry into the `fcx.lllocals` map
fn insert_lllocals<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                               bindings_map: &BindingsMap<'tcx>,
                               cs: Option<cleanup::ScopeId>)
                               -> Block<'blk, 'tcx> {
    for (&ident, &binding_info) in bindings_map {
        let llval = match binding_info.trmode {
            // By value mut binding for a copy type: load from the ptr
            // into the matched value and copy to our alloca
            TrByCopy(llbinding) => {
                let llval = Load(bcx, binding_info.llmatch);
                let datum = Datum::new(llval, binding_info.ty, Lvalue);
                call_lifetime_start(bcx, llbinding);
                bcx = datum.store_to(bcx, llbinding);
                if let Some(cs) = cs {
                    bcx.fcx.schedule_lifetime_end(cs, llbinding);
                }

                llbinding
            },

            // By value move bindings: load from the ptr into the matched value
            TrByMove => Load(bcx, binding_info.llmatch),

            // By ref binding: use the ptr into the matched value
            TrByRef => binding_info.llmatch
        };

        let datum = Datum::new(llval, binding_info.ty, Lvalue);
        if let Some(cs) = cs {
            bcx.fcx.schedule_lifetime_end(cs, binding_info.llmatch);
            bcx.fcx.schedule_drop_and_fill_mem(cs, llval, binding_info.ty);
        }

        debug!("binding {} to {}", binding_info.id, bcx.val_to_string(llval));
        bcx.fcx.lllocals.borrow_mut().insert(binding_info.id, datum);
        debuginfo::create_match_binding_metadata(bcx, ident.name, binding_info);
    }
    bcx
}

fn compile_guard<'a, 'p, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     guard_expr: &ast::Expr,
                                     data: &ArmData<'p, 'blk, 'tcx>,
                                     failure_dest: BasicBlockRef)
                                     -> Block<'blk, 'tcx> {
    debug!("compile_guard(bcx={}, guard_expr={:?})",
           bcx.to_str(),
           guard_expr);
    let _indenter = indenter();

    let mut bcx = insert_lllocals(bcx, &data.bindings_map, None);

    let val = unpack_datum!(bcx, expr::trans(bcx, guard_expr));
    let val = val.to_llbool(bcx);

    for (_, &binding_info) in &data.bindings_map {
        if let TrByCopy(llbinding) = binding_info.trmode {
            call_lifetime_end(bcx, llbinding);
        }
    }

    for (_, &binding_info) in &data.bindings_map {
        bcx.fcx.lllocals.borrow_mut().remove(&binding_info.id);
    }

    with_cond(bcx, Not(bcx, val, guard_expr.debug_loc()), |bcx| {
        for (_, &binding_info) in &data.bindings_map {
            call_lifetime_end(bcx, binding_info.llmatch);
        }
        Br(bcx, failure_dest, DebugLoc::None);
        bcx
    })
}

/// Checks whether the binding in `discr` is assigned to anywhere in the expression `body`
fn is_discr_reassigned(bcx: Block, discr: &ast::Expr, body: &ast::Expr) -> bool {
    let (vid, field) = match discr.node {
        ast::ExprPath(..) => match bcx.def(discr.id) {
            def::DefLocal(vid) | def::DefUpvar(vid, _) => (vid, None),
            _ => return false
        },
        ast::ExprField(ref base, field) => {
            let vid = match bcx.tcx().def_map.borrow().get(&base.id).map(|d| d.full_def()) {
                Some(def::DefLocal(vid)) | Some(def::DefUpvar(vid, _)) => vid,
                _ => return false
            };
            (vid, Some(mc::NamedField(field.node.name)))
        },
        ast::ExprTupField(ref base, field) => {
            let vid = match bcx.tcx().def_map.borrow().get(&base.id).map(|d| d.full_def()) {
                Some(def::DefLocal(vid)) | Some(def::DefUpvar(vid, _)) => vid,
                _ => return false
            };
            (vid, Some(mc::PositionalField(field.node)))
        },
        _ => return false
    };

    let mut rc = ReassignmentChecker {
        node: vid,
        field: field,
        reassigned: false
    };
    {
        let infcx = infer::normalizing_infer_ctxt(bcx.tcx(), &bcx.tcx().tables);
        let mut visitor = euv::ExprUseVisitor::new(&mut rc, &infcx);
        visitor.walk_expr(body);
    }
    rc.reassigned
}

struct ReassignmentChecker {
    node: ast::NodeId,
    field: Option<mc::FieldName>,
    reassigned: bool
}

// Determine if the expression we're matching on is reassigned to within
// the body of the match's arm.
// We only care for the `mutate` callback since this check only matters
// for cases where the matched value is moved.
impl<'tcx> euv::Delegate<'tcx> for ReassignmentChecker {
    fn consume(&mut self, _: ast::NodeId, _: Span, _: mc::cmt, _: euv::ConsumeMode) {}
    fn matched_pat(&mut self, _: &ast::Pat, _: mc::cmt, _: euv::MatchMode) {}
    fn consume_pat(&mut self, _: &ast::Pat, _: mc::cmt, _: euv::ConsumeMode) {}
    fn borrow(&mut self, _: ast::NodeId, _: Span, _: mc::cmt, _: ty::Region,
              _: ty::BorrowKind, _: euv::LoanCause) {}
    fn decl_without_init(&mut self, _: ast::NodeId, _: Span) {}

    fn mutate(&mut self, _: ast::NodeId, _: Span, cmt: mc::cmt, _: euv::MutateMode) {
        match cmt.cat {
            mc::cat_upvar(mc::Upvar { id: ty::UpvarId { var_id: vid, .. }, .. }) |
            mc::cat_local(vid) => self.reassigned |= self.node == vid,
            mc::cat_interior(ref base_cmt, mc::InteriorField(field)) => {
                match base_cmt.cat {
                    mc::cat_upvar(mc::Upvar { id: ty::UpvarId { var_id: vid, .. }, .. }) |
                    mc::cat_local(vid) => {
                        self.reassigned |= self.node == vid &&
                            (self.field.is_none() || Some(field) == self.field)
                    },
                    _ => {}
                }
            },
            _ => {}
        }
    }
}

fn create_bindings_map<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, pat: &ast::Pat,
                                   discr: &ast::Expr, body: &ast::Expr)
                                   -> BindingsMap<'tcx> {
    // Create the bindings map, which is a mapping from each binding name
    // to an alloca() that will be the value for that local variable.
    // Note that we use the names because each binding will have many ids
    // from the various alternatives.
    let ccx = bcx.ccx();
    let tcx = bcx.tcx();
    let reassigned = is_discr_reassigned(bcx, discr, body);
    let mut bindings_map = FnvHashMap();
    pat_bindings(&tcx.def_map, &*pat, |bm, p_id, span, path1| {
        let ident = path1.node;
        let name = ident.name;
        let variable_ty = node_id_type(bcx, p_id);
        let llvariable_ty = type_of::type_of(ccx, variable_ty);
        let tcx = bcx.tcx();
        let param_env = tcx.empty_parameter_environment();

        let llmatch;
        let trmode;
        match bm {
            ast::BindByValue(_)
                if !variable_ty.moves_by_default(&param_env, span) || reassigned =>
            {
                llmatch = alloca_no_lifetime(bcx,
                                 llvariable_ty.ptr_to(),
                                 "__llmatch");
                trmode = TrByCopy(alloca_no_lifetime(bcx,
                                         llvariable_ty,
                                         &bcx.name(name)));
            }
            ast::BindByValue(_) => {
                // in this case, the final type of the variable will be T,
                // but during matching we need to store a *T as explained
                // above
                llmatch = alloca_no_lifetime(bcx,
                                 llvariable_ty.ptr_to(),
                                 &bcx.name(name));
                trmode = TrByMove;
            }
            ast::BindByRef(_) => {
                llmatch = alloca_no_lifetime(bcx,
                                 llvariable_ty,
                                 &bcx.name(name));
                trmode = TrByRef;
            }
        };
        bindings_map.insert(ident, BindingInfo {
            llmatch: llmatch,
            trmode: trmode,
            id: p_id,
            span: span,
            ty: variable_ty
        });
    });
    return bindings_map;
}

pub fn trans_match<'blk, 'tcx>(scope_cx: Block<'blk, 'tcx>,
                               match_expr: &ast::Expr,
                               discr_expr: &ast::Expr,
                               arms: &[ast::Arm],
                               dest: Dest)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("match::trans_match");
    let fcx = scope_cx.fcx;
    let mut bcx = scope_cx;

    // FIXME: It might be more efficient to specialize this.
    let discr_datum = unpack_datum!(bcx, expr::trans_to_lvalue(bcx, discr_expr,
                                                               "match"));
    // Early return if the matching code isn't reachable.
    if bcx.unreachable.get() {
        return bcx;
    }

    // Create a context for each arm. The patterns are canonicalized by
    // "inlining" any constants (e.g. transforming `C => {}` to `(1,2) => `{}`
    // where C is defined as `const C : (u32, u32) = (1, 2);`.
    let mut pat_renaming_map = if scope_cx.sess().opts.debuginfo != NoDebugInfo {
        Some(FnvHashMap())
    } else {
        None
    };
    let arm_datas: Vec<ArmData> = {
        let mut static_inliner = StaticInliner::new(scope_cx.tcx(),
                                                    pat_renaming_map.as_mut());
        arms.iter().map(|arm| ArmData {
            bodycx: fcx.new_id_block("case_body", arm.body.id),
            arm: arm,
            bindings_map: create_bindings_map(bcx, &arm.pats[0], discr_expr, &arm.body),
            arm_pats: arm.pats.iter().map(|p| static_inliner.fold_pat((*p).clone())).collect()
        }).collect()
    };

    // Perform matching.
    // FIXME: Need optimized algorithm.
    for arm_data in &arm_datas {
        for pat in &arm_data.arm_pats {
            // Verify the pattern.
            let failure_dest = fcx.new_id_block("pat_no_match", pat.id);
            let mut pat_bindings = Vec::new();
            bcx = compile_pattern(bcx, discr_datum, &pat, failure_dest.llbb, &mut pat_bindings);

            // Bind pattern
            for (ident, value_ptr) in pat_bindings {
                let binfo = *arm_data.bindings_map.get(&ident).unwrap();
                call_lifetime_start(bcx, binfo.llmatch);
                if binfo.trmode == TrByRef && type_is_fat_ptr(bcx.tcx(), binfo.ty) {
                    expr::copy_fat_ptr(bcx, value_ptr, binfo.llmatch);
                } else {
                    Store(bcx, value_ptr, binfo.llmatch);
                }

            }

            // Verify the guard.
            if let Some(ref guard_expr) = arm_data.arm.guard {
                bcx = compile_guard(bcx,
                                    &guard_expr,
                                    arm_data,
                                    failure_dest.llbb);
            }
            // All checks succeeded, branch to the body.
            Br(bcx, arm_data.bodycx.llbb, DebugLoc::None);

            // Continue checking the next pattern.
            bcx = failure_dest;
        }
    }
    Unreachable(bcx);

    // Compile the body of each arm.
    let mut arm_cxs = Vec::new();
    for arm_data in &arm_datas {
        let mut bcx = arm_data.bodycx;

        // insert bindings into the lllocals map and add cleanups
        let cs = fcx.push_custom_cleanup_scope();
        bcx = insert_lllocals(bcx, &arm_data.bindings_map, Some(cleanup::CustomScope(cs)));
        bcx = expr::trans_into(bcx, &*arm_data.arm.body, dest);
        bcx = fcx.pop_and_trans_custom_cleanup_scope(bcx, cs);
        arm_cxs.push(bcx);
    }

    bcx = scope_cx.fcx.join_blocks(match_expr.id, &arm_cxs[..]);
    return bcx;
}

/// Generates code for a local variable declaration like `let <pat>;` or `let <pat> =
/// <opt_init_expr>`.
pub fn store_local<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               local: &ast::Local)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("match::store_local");
    let mut bcx = bcx;
    let tcx = bcx.tcx();
    let pat = &*local.pat;

    fn create_dummy_locals<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                       pat: &ast::Pat)
                                       -> Block<'blk, 'tcx> {
        let _icx = push_ctxt("create_dummy_locals");
        // create dummy memory for the variables if we have no
        // value to store into them immediately
        let tcx = bcx.tcx();
        pat_bindings(&tcx.def_map, pat, |_, p_id, _, path1| {
            let scope = cleanup::var_scope(tcx, p_id);
            bcx = mk_binding_alloca(
                bcx, p_id, path1.node.name, scope, (),
                |(), bcx, llval, ty| { drop_done_fill_mem(bcx, llval, ty); bcx });
        });
        bcx
    }

    match local.init {
        Some(ref init_expr) => {
            // Optimize the "let x = expr" case. This just writes
            // the result of evaluating `expr` directly into the alloca
            // for `x`. Often the general path results in similar or the
            // same code post-optimization, but not always. In particular,
            // in unsafe code, you can have expressions like
            //
            //    let x = intrinsics::uninit();
            //
            // In such cases, the more general path is unsafe, because
            // it assumes it is matching against a valid value.
            match simple_identifier(&*pat) {
                Some(ident) => {
                    let var_scope = cleanup::var_scope(tcx, local.id);
                    return mk_binding_alloca(
                        bcx, pat.id, ident.name, var_scope, (),
                        |(), bcx, v, _| expr::trans_into(bcx, &**init_expr,
                                                         expr::SaveIn(v)));
                }

                None => {}
            }

            // General path.
            let init_datum =
                unpack_datum!(bcx, expr::trans_to_lvalue(bcx, &**init_expr, "let"));
            if bcx.sess().asm_comments() {
                add_comment(bcx, "creating zeroable ref llval");
            }
            let var_scope = cleanup::var_scope(tcx, local.id);
            bind_irrefutable_pat(bcx, pat, init_datum.val, var_scope)
        }
        None => {
            create_dummy_locals(bcx, pat)
        }
    }
}

fn mk_binding_alloca<'blk, 'tcx, A, F>(bcx: Block<'blk, 'tcx>,
                                       p_id: ast::NodeId,
                                       name: ast::Name,
                                       cleanup_scope: cleanup::ScopeId,
                                       arg: A,
                                       populate: F)
                                       -> Block<'blk, 'tcx> where
    F: FnOnce(A, Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>,
{
    let var_ty = node_id_type(bcx, p_id);

    // Allocate memory on stack for the binding.
    let llval = alloc_ty(bcx, var_ty, &bcx.name(name));

    // Subtle: be sure that we *populate* the memory *before*
    // we schedule the cleanup.
    let bcx = populate(arg, bcx, llval, var_ty);
    bcx.fcx.schedule_lifetime_end(cleanup_scope, llval);
    bcx.fcx.schedule_drop_mem(cleanup_scope, llval, var_ty);

    // Now that memory is initialized and has cleanup scheduled,
    // create the datum and insert into the local variable map.
    let datum = Datum::new(llval, var_ty, Lvalue);
    bcx.fcx.lllocals.borrow_mut().insert(p_id, datum);
    bcx
}

/// A simple version of the pattern matching code that only handles
/// irrefutable patterns. This is used in let/argument patterns,
/// not in match statements. Unifying this code with the code above
/// sounds nice, but in practice it produces very inefficient code,
/// since the match code is so much more general. In most cases,
/// LLVM is able to optimize the code, but it causes longer compile
/// times and makes the generated code nigh impossible to read.
///
/// # Arguments
/// - bcx: starting basic block context
/// - pat: the irrefutable pattern being matched.
/// - val: the value being matched -- must be an lvalue (by ref, with cleanup)
pub fn bind_irrefutable_pat<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    pat: &ast::Pat,
                                    val: ValueRef,
                                    cleanup_scope: cleanup::ScopeId)
                                    -> Block<'blk, 'tcx> {
    debug!("bind_irrefutable_pat(bcx={}, pat={:?})",
           bcx.to_str(),
           pat);

    if bcx.sess().asm_comments() {
        add_comment(bcx, &format!("bind_irrefutable_pat(pat={:?})",
                                 pat));
    }

    let _indenter = indenter();

    let _icx = push_ctxt("match::bind_irrefutable_pat");
    let mut bcx = bcx;
    let tcx = bcx.tcx();
    let ccx = bcx.ccx();
    match pat.node {
        ast::PatIdent(pat_binding_mode, ref path1, ref inner) => {
            if pat_is_binding(&tcx.def_map, &*pat) {
                // Allocate the stack slot where the value of this
                // binding will live and place it into the appropriate
                // map.
                bcx = mk_binding_alloca(
                    bcx, pat.id, path1.node.name, cleanup_scope, (),
                    |(), bcx, llval, ty| {
                        match pat_binding_mode {
                            ast::BindByValue(_) => {
                                // By value binding: move the value that `val`
                                // points at into the binding's stack slot.
                                let d = Datum::new(val, ty, Lvalue);
                                d.store_to(bcx, llval)
                            }

                            ast::BindByRef(_) => {
                                // By ref binding: the value of the variable
                                // is the pointer `val` itself or fat pointer referenced by `val`
                                if type_is_fat_ptr(bcx.tcx(), ty) {
                                    expr::copy_fat_ptr(bcx, val, llval);
                                }
                                else {
                                    Store(bcx, val, llval);
                                }

                                bcx
                            }
                        }
                    });
            }

            if let Some(ref inner_pat) = *inner {
                bcx = bind_irrefutable_pat(bcx, &**inner_pat, val, cleanup_scope);
            }
        }
        ast::PatEnum(_, ref sub_pats) => {
            let opt_def = bcx.tcx().def_map.borrow().get(&pat.id).map(|d| d.full_def());
            match opt_def {
                Some(def::DefVariant(enum_id, var_id, _)) => {
                    let repr = adt::represent_node(bcx, pat.id);
                    let vinfo = ccx.tcx().enum_variant_with_id(enum_id, var_id);
                    let args = extract_variant_args(bcx,
                                                    &*repr,
                                                    vinfo.disr_val,
                                                    val);
                    if let Some(ref sub_pat) = *sub_pats {
                        for (i, &argval) in args.iter().enumerate() {
                            bcx = bind_irrefutable_pat(bcx, &*sub_pat[i],
                                                       argval, cleanup_scope);
                        }
                    }
                }
                Some(def::DefStruct(..)) => {
                    match *sub_pats {
                        None => {
                            // This is a unit-like struct. Nothing to do here.
                        }
                        Some(ref elems) => {
                            // This is the tuple struct case.
                            let repr = adt::represent_node(bcx, pat.id);
                            for (i, elem) in elems.iter().enumerate() {
                                let fldptr = adt::trans_field_ptr(bcx, &*repr,
                                                                  val, 0, i);
                                bcx = bind_irrefutable_pat(bcx, &**elem,
                                                           fldptr, cleanup_scope);
                            }
                        }
                    }
                }
                _ => {
                    // Nothing to do here.
                }
            }
        }
        ast::PatStruct(_, ref fields, _) => {
            let tcx = bcx.tcx();
            let pat_ty = node_id_type(bcx, pat.id);
            let pat_repr = adt::represent_type(bcx.ccx(), pat_ty);
            expr::with_field_tys(tcx, pat_ty, Some(pat.id), |discr, field_tys| {
                for f in fields {
                    let ix = tcx.field_idx_strict(f.node.ident.name, field_tys);
                    let fldptr = adt::trans_field_ptr(bcx, &*pat_repr, val,
                                                      discr, ix);
                    bcx = bind_irrefutable_pat(bcx, &*f.node.pat, fldptr, cleanup_scope);
                }
            })
        }
        ast::PatTup(ref elems) => {
            let repr = adt::represent_node(bcx, pat.id);
            for (i, elem) in elems.iter().enumerate() {
                let fldptr = adt::trans_field_ptr(bcx, &*repr, val, 0, i);
                bcx = bind_irrefutable_pat(bcx, &**elem, fldptr, cleanup_scope);
            }
        }
        ast::PatBox(ref inner) => {
            let llbox = Load(bcx, val);
            bcx = bind_irrefutable_pat(bcx, &**inner, llbox, cleanup_scope);
        }
        ast::PatRegion(ref inner, _) => {
            let loaded_val = Load(bcx, val);
            bcx = bind_irrefutable_pat(bcx, &**inner, loaded_val, cleanup_scope);
        }
        ast::PatVec(ref before, ref slice, ref after) => {
            let pat_ty = node_id_type(bcx, pat.id);
            let mut extracted = extract_vec_elems(bcx, pat_ty, before.len(), after.len(), val);
            match slice {
                &Some(_) => {
                    extracted.insert(
                        before.len(),
                        bind_subslice_pat(bcx, pat.id, val, before.len(), after.len())
                    );
                }
                &None => ()
            }
            bcx = before
                .iter()
                .chain(slice.iter())
                .chain(after.iter())
                .zip(extracted)
                .fold(bcx, |bcx, (inner, elem)|
                    bind_irrefutable_pat(bcx, &**inner, elem, cleanup_scope)
                );
        }
        ast::PatMac(..) => {
            bcx.sess().span_bug(pat.span, "unexpanded macro");
        }
        ast::PatQPath(..) | ast::PatWild(_) | ast::PatLit(_) |
        ast::PatRange(_, _) => ()
    }
    return bcx;
}

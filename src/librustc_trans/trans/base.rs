// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! Translate the completed AST to the LLVM IR.
//!
//! Some functions here, such as trans_block and trans_expr, return a value --
//! the result of the translation to LLVM -- while others, such as trans_fn,
//! trans_impl, and trans_item, are called only for the side effect of adding a
//! particular definition to the LLVM IR output we're producing.
//!
//! Hopefully useful general knowledge about trans:
//!
//!   * There's no way to find out the Ty type of a ValueRef.  Doing so
//!     would be "trying to get the eggs out of an omelette" (credit:
//!     pcwalton).  You can, instead, find out its TypeRef by calling val_ty,
//!     but one TypeRef corresponds to many `Ty`s; for instance, tup(int, int,
//!     int) and rec(x=int, y=int, z=int) will have the same TypeRef.

#![allow(non_camel_case_types)]

pub use self::ValueOrigin::*;

use super::CrateTranslation;
use super::ModuleTranslation;

use back::link::mangle_exported_name;
use back::{link, abi};
use lint;
use llvm::{BasicBlockRef, Linkage, ValueRef, Vector, get_param};
use llvm;
use middle::cfg;
use middle::cstore::CrateStore;
use middle::def_id::DefId;
use middle::infer;
use middle::lang_items::{LangItem, ExchangeMallocFnLangItem, StartFnLangItem};
use middle::weak_lang_items;
use middle::pat_util::simple_name;
use middle::subst::Substs;
use middle::ty::{self, Ty, HasTypeFlags};
use rustc::front::map as hir_map;
use rustc_mir::mir_map::MirMap;
use session::config::{self, NoDebugInfo, FullDebugInfo};
use session::Session;
use trans::_match;
use trans::adt;
use trans::attributes;
use trans::build::*;
use trans::builder::{Builder, noname};
use trans::callee;
use trans::cleanup::{self, CleanupMethods, DropHint};
use trans::closure;
use trans::common::{Block, C_bool, C_bytes_in_context, C_i32, C_int, C_uint, C_integral};
use trans::common::{C_null, C_struct_in_context, C_u64, C_u8, C_undef};
use trans::common::{CrateContext, DropFlagHintsMap, Field, FunctionContext};
use trans::common::{Result, NodeIdAndSpan, VariantInfo};
use trans::common::{node_id_type, return_type_is_void};
use trans::common::{type_is_immediate, type_is_zero_size, val_ty};
use trans::common;
use trans::consts;
use trans::context::SharedCrateContext;
use trans::controlflow;
use trans::datum;
use trans::debuginfo::{self, DebugLoc, ToDebugLoc};
use trans::declare;
use trans::expr;
use trans::foreign;
use trans::glue;
use trans::intrinsic;
use trans::machine;
use trans::machine::{llsize_of, llsize_of_real};
use trans::meth;
use trans::mir;
use trans::monomorphize;
use trans::tvec;
use trans::type_::Type;
use trans::type_of;
use trans::type_of::*;
use trans::value::Value;
use util::common::indenter;
use util::sha2::Sha256;
use util::nodemap::{NodeMap, NodeSet};

use arena::TypedArena;
use libc::c_uint;
use std::ffi::{CStr, CString};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::str;
use std::{i8, i16, i32, i64};
use syntax::abi::{Rust, RustCall, RustIntrinsic, PlatformIntrinsic, Abi};
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use rustc_front;
use rustc_front::intravisit::{self, Visitor};
use rustc_front::hir;
use syntax::ast;

thread_local! {
    static TASK_LOCAL_INSN_KEY: RefCell<Option<Vec<&'static str>>> = {
        RefCell::new(None)
    }
}

pub fn with_insn_ctxt<F>(blk: F)
    where F: FnOnce(&[&'static str])
{
    TASK_LOCAL_INSN_KEY.with(move |slot| {
        slot.borrow().as_ref().map(move |s| blk(s));
    })
}

pub fn init_insn_ctxt() {
    TASK_LOCAL_INSN_KEY.with(|slot| {
        *slot.borrow_mut() = Some(Vec::new());
    });
}

pub struct _InsnCtxt {
    _cannot_construct_outside_of_this_module: (),
}

impl Drop for _InsnCtxt {
    fn drop(&mut self) {
        TASK_LOCAL_INSN_KEY.with(|slot| {
            match slot.borrow_mut().as_mut() {
                Some(ctx) => {
                    ctx.pop();
                }
                None => {}
            }
        })
    }
}

pub fn push_ctxt(s: &'static str) -> _InsnCtxt {
    debug!("new InsnCtxt: {}", s);
    TASK_LOCAL_INSN_KEY.with(|slot| {
        match slot.borrow_mut().as_mut() {
            Some(ctx) => ctx.push(s),
            None => {}
        }
    });
    _InsnCtxt {
        _cannot_construct_outside_of_this_module: (),
    }
}

pub struct StatRecorder<'a, 'tcx: 'a> {
    ccx: &'a CrateContext<'a, 'tcx>,
    name: Option<String>,
    istart: usize,
}

impl<'a, 'tcx> StatRecorder<'a, 'tcx> {
    pub fn new(ccx: &'a CrateContext<'a, 'tcx>, name: String) -> StatRecorder<'a, 'tcx> {
        let istart = ccx.stats().n_llvm_insns.get();
        StatRecorder {
            ccx: ccx,
            name: Some(name),
            istart: istart,
        }
    }
}

impl<'a, 'tcx> Drop for StatRecorder<'a, 'tcx> {
    fn drop(&mut self) {
        if self.ccx.sess().trans_stats() {
            let iend = self.ccx.stats().n_llvm_insns.get();
            self.ccx
                .stats()
                .fn_stats
                .borrow_mut()
                .push((self.name.take().unwrap(), iend - self.istart));
            self.ccx.stats().n_fns.set(self.ccx.stats().n_fns.get() + 1);
            // Reset LLVM insn count to avoid compound costs.
            self.ccx.stats().n_llvm_insns.set(self.istart);
        }
    }
}

fn get_extern_rust_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                fn_ty: Ty<'tcx>,
                                name: &str,
                                did: DefId)
                                -> ValueRef {
    match ccx.externs().borrow().get(name) {
        Some(n) => return *n,
        None => (),
    }

    let f = declare::declare_rust_fn(ccx, name, fn_ty);

    let attrs = ccx.sess().cstore.item_attrs(did);
    attributes::from_fn_attrs(ccx, &attrs[..], f);

    ccx.externs().borrow_mut().insert(name.to_string(), f);
    f
}

pub fn self_type_for_closure<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                       closure_id: DefId,
                                       fn_ty: Ty<'tcx>)
                                       -> Ty<'tcx> {
    let closure_kind = ccx.tcx().closure_kind(closure_id);
    match closure_kind {
        ty::FnClosureKind => {
            ccx.tcx().mk_imm_ref(ccx.tcx().mk_region(ty::ReStatic), fn_ty)
        }
        ty::FnMutClosureKind => {
            ccx.tcx().mk_mut_ref(ccx.tcx().mk_region(ty::ReStatic), fn_ty)
        }
        ty::FnOnceClosureKind => fn_ty,
    }
}

pub fn kind_for_closure(ccx: &CrateContext, closure_id: DefId) -> ty::ClosureKind {
    *ccx.tcx().tables.borrow().closure_kinds.get(&closure_id).unwrap()
}

pub fn get_extern_const<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                  did: DefId,
                                  t: Ty<'tcx>)
                                  -> ValueRef {
    let name = ccx.sess().cstore.item_symbol(did);
    let ty = type_of(ccx, t);
    match ccx.externs().borrow_mut().get(&name) {
        Some(n) => return *n,
        None => (),
    }
    // FIXME(nagisa): perhaps the map of externs could be offloaded to llvm somehow?
    // FIXME(nagisa): investigate whether it can be changed into define_global
    let c = declare::declare_global(ccx, &name[..], ty);
    // Thread-local statics in some other crate need to *always* be linked
    // against in a thread-local fashion, so we need to be sure to apply the
    // thread-local attribute locally if it was present remotely. If we
    // don't do this then linker errors can be generated where the linker
    // complains that one object files has a thread local version of the
    // symbol and another one doesn't.
    for attr in ccx.tcx().get_attrs(did).iter() {
        if attr.check_name("thread_local") {
            llvm::set_thread_local(c, true);
        }
    }
    if ccx.use_dll_storage_attrs() {
        llvm::SetDLLStorageClass(c, llvm::DLLImportStorageClass);
    }
    ccx.externs().borrow_mut().insert(name.to_string(), c);
    return c;
}

fn require_alloc_fn<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, info_ty: Ty<'tcx>, it: LangItem) -> DefId {
    match bcx.tcx().lang_items.require(it) {
        Ok(id) => id,
        Err(s) => {
            bcx.sess().fatal(&format!("allocation of `{}` {}", info_ty, s));
        }
    }
}

// The following malloc_raw_dyn* functions allocate a box to contain
// a given type, but with a potentially dynamic size.

pub fn malloc_raw_dyn<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  llty_ptr: Type,
                                  info_ty: Ty<'tcx>,
                                  size: ValueRef,
                                  align: ValueRef,
                                  debug_loc: DebugLoc)
                                  -> Result<'blk, 'tcx> {
    let _icx = push_ctxt("malloc_raw_exchange");

    // Allocate space:
    let r = callee::trans_lang_call(bcx,
                                    require_alloc_fn(bcx, info_ty, ExchangeMallocFnLangItem),
                                    &[size, align],
                                    None,
                                    debug_loc);

    Result::new(r.bcx, PointerCast(r.bcx, r.val, llty_ptr))
}


pub fn bin_op_to_icmp_predicate(ccx: &CrateContext,
                                op: hir::BinOp_,
                                signed: bool)
                                -> llvm::IntPredicate {
    match op {
        hir::BiEq => llvm::IntEQ,
        hir::BiNe => llvm::IntNE,
        hir::BiLt => if signed { llvm::IntSLT } else { llvm::IntULT },
        hir::BiLe => if signed { llvm::IntSLE } else { llvm::IntULE },
        hir::BiGt => if signed { llvm::IntSGT } else { llvm::IntUGT },
        hir::BiGe => if signed { llvm::IntSGE } else { llvm::IntUGE },
        op => {
            ccx.sess()
               .bug(&format!("comparison_op_to_icmp_predicate: expected comparison operator, \
                              found {:?}",
                             op));
        }
    }
}

pub fn bin_op_to_fcmp_predicate(ccx: &CrateContext, op: hir::BinOp_) -> llvm::RealPredicate {
    match op {
        hir::BiEq => llvm::RealOEQ,
        hir::BiNe => llvm::RealUNE,
        hir::BiLt => llvm::RealOLT,
        hir::BiLe => llvm::RealOLE,
        hir::BiGt => llvm::RealOGT,
        hir::BiGe => llvm::RealOGE,
        op => {
            ccx.sess()
               .bug(&format!("comparison_op_to_fcmp_predicate: expected comparison operator, \
                              found {:?}",
                             op));
        }
    }
}

pub fn compare_fat_ptrs<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    lhs_addr: ValueRef,
                                    lhs_extra: ValueRef,
                                    rhs_addr: ValueRef,
                                    rhs_extra: ValueRef,
                                    _t: Ty<'tcx>,
                                    op: hir::BinOp_,
                                    debug_loc: DebugLoc)
                                    -> ValueRef {
    match op {
        hir::BiEq => {
            let addr_eq = ICmp(bcx, llvm::IntEQ, lhs_addr, rhs_addr, debug_loc);
            let extra_eq = ICmp(bcx, llvm::IntEQ, lhs_extra, rhs_extra, debug_loc);
            And(bcx, addr_eq, extra_eq, debug_loc)
        }
        hir::BiNe => {
            let addr_eq = ICmp(bcx, llvm::IntNE, lhs_addr, rhs_addr, debug_loc);
            let extra_eq = ICmp(bcx, llvm::IntNE, lhs_extra, rhs_extra, debug_loc);
            Or(bcx, addr_eq, extra_eq, debug_loc)
        }
        hir::BiLe | hir::BiLt | hir::BiGe | hir::BiGt => {
            // a OP b ~ a.0 STRICT(OP) b.0 | (a.0 == b.0 && a.1 OP a.1)
            let (op, strict_op) = match op {
                hir::BiLt => (llvm::IntULT, llvm::IntULT),
                hir::BiLe => (llvm::IntULE, llvm::IntULT),
                hir::BiGt => (llvm::IntUGT, llvm::IntUGT),
                hir::BiGe => (llvm::IntUGE, llvm::IntUGT),
                _ => unreachable!(),
            };

            let addr_eq = ICmp(bcx, llvm::IntEQ, lhs_addr, rhs_addr, debug_loc);
            let extra_op = ICmp(bcx, op, lhs_extra, rhs_extra, debug_loc);
            let addr_eq_extra_op = And(bcx, addr_eq, extra_op, debug_loc);

            let addr_strict = ICmp(bcx, strict_op, lhs_addr, rhs_addr, debug_loc);
            Or(bcx, addr_strict, addr_eq_extra_op, debug_loc)
        }
        _ => {
            bcx.tcx().sess.bug("unexpected fat ptr binop");
        }
    }
}

pub fn compare_scalar_types<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        lhs: ValueRef,
                                        rhs: ValueRef,
                                        t: Ty<'tcx>,
                                        op: hir::BinOp_,
                                        debug_loc: DebugLoc)
                                        -> ValueRef {
    match t.sty {
        ty::TyTuple(ref tys) if tys.is_empty() => {
            // We don't need to do actual comparisons for nil.
            // () == () holds but () < () does not.
            match op {
                hir::BiEq | hir::BiLe | hir::BiGe => return C_bool(bcx.ccx(), true),
                hir::BiNe | hir::BiLt | hir::BiGt => return C_bool(bcx.ccx(), false),
                // refinements would be nice
                _ => bcx.sess().bug("compare_scalar_types: must be a comparison operator"),
            }
        }
        ty::TyBareFn(..) | ty::TyBool | ty::TyUint(_) | ty::TyChar => {
            ICmp(bcx,
                 bin_op_to_icmp_predicate(bcx.ccx(), op, false),
                 lhs,
                 rhs,
                 debug_loc)
        }
        ty::TyRawPtr(mt) if common::type_is_sized(bcx.tcx(), mt.ty) => {
            ICmp(bcx,
                 bin_op_to_icmp_predicate(bcx.ccx(), op, false),
                 lhs,
                 rhs,
                 debug_loc)
        }
        ty::TyRawPtr(_) => {
            let lhs_addr = Load(bcx, GEPi(bcx, lhs, &[0, abi::FAT_PTR_ADDR]));
            let lhs_extra = Load(bcx, GEPi(bcx, lhs, &[0, abi::FAT_PTR_EXTRA]));

            let rhs_addr = Load(bcx, GEPi(bcx, rhs, &[0, abi::FAT_PTR_ADDR]));
            let rhs_extra = Load(bcx, GEPi(bcx, rhs, &[0, abi::FAT_PTR_EXTRA]));
            compare_fat_ptrs(bcx,
                             lhs_addr,
                             lhs_extra,
                             rhs_addr,
                             rhs_extra,
                             t,
                             op,
                             debug_loc)
        }
        ty::TyInt(_) => {
            ICmp(bcx,
                 bin_op_to_icmp_predicate(bcx.ccx(), op, true),
                 lhs,
                 rhs,
                 debug_loc)
        }
        ty::TyFloat(_) => {
            FCmp(bcx,
                 bin_op_to_fcmp_predicate(bcx.ccx(), op),
                 lhs,
                 rhs,
                 debug_loc)
        }
        // Should never get here, because t is scalar.
        _ => bcx.sess().bug("non-scalar type passed to compare_scalar_types"),
    }
}

pub fn compare_simd_types<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      lhs: ValueRef,
                                      rhs: ValueRef,
                                      t: Ty<'tcx>,
                                      ret_ty: Type,
                                      op: hir::BinOp_,
                                      debug_loc: DebugLoc)
                                      -> ValueRef {
    let signed = match t.sty {
        ty::TyFloat(_) => {
            let cmp = bin_op_to_fcmp_predicate(bcx.ccx(), op);
            return SExt(bcx, FCmp(bcx, cmp, lhs, rhs, debug_loc), ret_ty);
        },
        ty::TyUint(_) => false,
        ty::TyInt(_) => true,
        _ => bcx.sess().bug("compare_simd_types: invalid SIMD type"),
    };

    let cmp = bin_op_to_icmp_predicate(bcx.ccx(), op, signed);
    // LLVM outputs an `< size x i1 >`, so we need to perform a sign extension
    // to get the correctly sized type. This will compile to a single instruction
    // once the IR is converted to assembly if the SIMD instruction is supported
    // by the target architecture.
    SExt(bcx, ICmp(bcx, cmp, lhs, rhs, debug_loc), ret_ty)
}

// Iterates through the elements of a structural type.
pub fn iter_structural_ty<'blk, 'tcx, F>(cx: Block<'blk, 'tcx>,
                                         av: ValueRef,
                                         t: Ty<'tcx>,
                                         mut f: F)
                                         -> Block<'blk, 'tcx>
    where F: FnMut(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>
{
    let _icx = push_ctxt("iter_structural_ty");

    fn iter_variant<'blk, 'tcx, F>(cx: Block<'blk, 'tcx>,
                                   repr: &adt::Repr<'tcx>,
                                   av: ValueRef,
                                   variant: ty::VariantDef<'tcx>,
                                   substs: &Substs<'tcx>,
                                   f: &mut F)
                                   -> Block<'blk, 'tcx>
        where F: FnMut(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>
    {
        let _icx = push_ctxt("iter_variant");
        let tcx = cx.tcx();
        let mut cx = cx;

        for (i, field) in variant.fields.iter().enumerate() {
            let arg = monomorphize::field_ty(tcx, substs, field);
            cx = f(cx,
                   adt::trans_field_ptr(cx, repr, av, variant.disr_val, i),
                   arg);
        }
        return cx;
    }

    let (data_ptr, info) = if common::type_is_sized(cx.tcx(), t) {
        (av, None)
    } else {
        let data = expr::get_dataptr(cx, av);
        let info = expr::get_meta(cx, av);
        (Load(cx, data), Some(Load(cx, info)))
    };

    let mut cx = cx;
    match t.sty {
        ty::TyStruct(..) => {
            let repr = adt::represent_type(cx.ccx(), t);
            let VariantInfo { fields, discr } = VariantInfo::from_ty(cx.tcx(), t, None);
            for (i, &Field(_, field_ty)) in fields.iter().enumerate() {
                let llfld_a = adt::trans_field_ptr(cx, &*repr, data_ptr, discr, i);

                let val = if common::type_is_sized(cx.tcx(), field_ty) {
                    llfld_a
                } else {
                    let scratch = datum::rvalue_scratch_datum(cx, field_ty, "__fat_ptr_iter");
                    Store(cx, llfld_a, expr::get_dataptr(cx, scratch.val));
                    Store(cx, info.unwrap(), expr::get_meta(cx, scratch.val));
                    scratch.val
                };
                cx = f(cx, val, field_ty);
            }
        }
        ty::TyClosure(_, ref substs) => {
            let repr = adt::represent_type(cx.ccx(), t);
            for (i, upvar_ty) in substs.upvar_tys.iter().enumerate() {
                let llupvar = adt::trans_field_ptr(cx, &*repr, data_ptr, 0, i);
                cx = f(cx, llupvar, upvar_ty);
            }
        }
        ty::TyArray(_, n) => {
            let (base, len) = tvec::get_fixed_base_and_len(cx, data_ptr, n);
            let unit_ty = t.sequence_element_type(cx.tcx());
            cx = tvec::iter_vec_raw(cx, base, unit_ty, len, f);
        }
        ty::TySlice(_) | ty::TyStr => {
            let unit_ty = t.sequence_element_type(cx.tcx());
            cx = tvec::iter_vec_raw(cx, data_ptr, unit_ty, info.unwrap(), f);
        }
        ty::TyTuple(ref args) => {
            let repr = adt::represent_type(cx.ccx(), t);
            for (i, arg) in args.iter().enumerate() {
                let llfld_a = adt::trans_field_ptr(cx, &*repr, data_ptr, 0, i);
                cx = f(cx, llfld_a, *arg);
            }
        }
        ty::TyEnum(en, substs) => {
            let fcx = cx.fcx;
            let ccx = fcx.ccx;

            let repr = adt::represent_type(ccx, t);
            let n_variants = en.variants.len();

            // NB: we must hit the discriminant first so that structural
            // comparison know not to proceed when the discriminants differ.

            match adt::trans_switch(cx, &*repr, av) {
                (_match::Single, None) => {
                    if n_variants != 0 {
                        assert!(n_variants == 1);
                        cx = iter_variant(cx, &*repr, av, &en.variants[0], substs, &mut f);
                    }
                }
                (_match::Switch, Some(lldiscrim_a)) => {
                    cx = f(cx, lldiscrim_a, cx.tcx().types.isize);

                    // Create a fall-through basic block for the "else" case of
                    // the switch instruction we're about to generate. Note that
                    // we do **not** use an Unreachable instruction here, even
                    // though most of the time this basic block will never be hit.
                    //
                    // When an enum is dropped it's contents are currently
                    // overwritten to DTOR_DONE, which means the discriminant
                    // could have changed value to something not within the actual
                    // range of the discriminant. Currently this function is only
                    // used for drop glue so in this case we just return quickly
                    // from the outer function, and any other use case will only
                    // call this for an already-valid enum in which case the `ret
                    // void` will never be hit.
                    let ret_void_cx = fcx.new_temp_block("enum-iter-ret-void");
                    RetVoid(ret_void_cx, DebugLoc::None);
                    let llswitch = Switch(cx, lldiscrim_a, ret_void_cx.llbb, n_variants);
                    let next_cx = fcx.new_temp_block("enum-iter-next");

                    for variant in &en.variants {
                        let variant_cx = fcx.new_temp_block(&format!("enum-iter-variant-{}",
                                                                     &variant.disr_val
                                                                             .to_string()));
                        let case_val = adt::trans_case(cx, &*repr, variant.disr_val);
                        AddCase(llswitch, case_val, variant_cx.llbb);
                        let variant_cx = iter_variant(variant_cx,
                                                      &*repr,
                                                      data_ptr,
                                                      variant,
                                                      substs,
                                                      &mut f);
                        Br(variant_cx, next_cx.llbb, DebugLoc::None);
                    }
                    cx = next_cx;
                }
                _ => ccx.sess().unimpl("value from adt::trans_switch in iter_structural_ty"),
            }
        }
        _ => {
            cx.sess().unimpl(&format!("type in iter_structural_ty: {}", t))
        }
    }
    return cx;
}


/// Retrieve the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit funny. It is intended for use
/// in an upcast, where the new vtable for an object will be drived
/// from the old one.
pub fn unsized_info<'ccx, 'tcx>(ccx: &CrateContext<'ccx, 'tcx>,
                                source: Ty<'tcx>,
                                target: Ty<'tcx>,
                                old_info: Option<ValueRef>,
                                param_substs: &'tcx Substs<'tcx>)
                                -> ValueRef {
    let (source, target) = ccx.tcx().struct_lockstep_tails(source, target);
    match (&source.sty, &target.sty) {
        (&ty::TyArray(_, len), &ty::TySlice(_)) => C_uint(ccx, len),
        (&ty::TyTrait(_), &ty::TyTrait(_)) => {
            // For now, upcasts are limited to changes in marker
            // traits, and hence never actually require an actual
            // change to the vtable.
            old_info.expect("unsized_info: missing old info for trait upcast")
        }
        (_, &ty::TyTrait(box ty::TraitTy { ref principal, .. })) => {
            // Note that we preserve binding levels here:
            let substs = principal.0.substs.with_self_ty(source).erase_regions();
            let substs = ccx.tcx().mk_substs(substs);
            let trait_ref = ty::Binder(ty::TraitRef {
                def_id: principal.def_id(),
                substs: substs,
            });
            consts::ptrcast(meth::get_vtable(ccx, trait_ref, param_substs),
                            Type::vtable_ptr(ccx))
        }
        _ => ccx.sess().bug(&format!("unsized_info: invalid unsizing {:?} -> {:?}",
                                     source,
                                     target)),
    }
}

/// Coerce `src` to `dst_ty`. `src_ty` must be a thin pointer.
pub fn unsize_thin_ptr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   src: ValueRef,
                                   src_ty: Ty<'tcx>,
                                   dst_ty: Ty<'tcx>)
                                   -> (ValueRef, ValueRef) {
    debug!("unsize_thin_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::TyBox(a), &ty::TyBox(b)) |
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRef(_, ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) => {
            assert!(common::type_is_sized(bcx.tcx(), a));
            let ptr_ty = type_of::in_memory_type_of(bcx.ccx(), b).ptr_to();
            (PointerCast(bcx, src, ptr_ty),
             unsized_info(bcx.ccx(), a, b, None, bcx.fcx.param_substs))
        }
        _ => bcx.sess().bug("unsize_thin_ptr: called on bad types"),
    }
}

/// Coerce `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty` and store the result in `dst`
pub fn coerce_unsized_into<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                       src: ValueRef,
                                       src_ty: Ty<'tcx>,
                                       dst: ValueRef,
                                       dst_ty: Ty<'tcx>) {
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::TyBox(..), &ty::TyBox(..)) |
        (&ty::TyRef(..), &ty::TyRef(..)) |
        (&ty::TyRef(..), &ty::TyRawPtr(..)) |
        (&ty::TyRawPtr(..), &ty::TyRawPtr(..)) => {
            let (base, info) = if common::type_is_fat_ptr(bcx.tcx(), src_ty) {
                // fat-ptr to fat-ptr unsize preserves the vtable
                load_fat_ptr(bcx, src, src_ty)
            } else {
                let base = load_ty(bcx, src, src_ty);
                unsize_thin_ptr(bcx, base, src_ty, dst_ty)
            };
            store_fat_ptr(bcx, base, info, dst, dst_ty);
        }

        // This can be extended to enums and tuples in the future.
        // (&ty::TyEnum(def_id_a, _), &ty::TyEnum(def_id_b, _)) |
        (&ty::TyStruct(def_a, _), &ty::TyStruct(def_b, _)) => {
            assert_eq!(def_a, def_b);

            let src_repr = adt::represent_type(bcx.ccx(), src_ty);
            let src_fields = match &*src_repr {
                &adt::Repr::Univariant(ref s, _) => &s.fields,
                _ => bcx.sess().bug("struct has non-univariant repr"),
            };
            let dst_repr = adt::represent_type(bcx.ccx(), dst_ty);
            let dst_fields = match &*dst_repr {
                &adt::Repr::Univariant(ref s, _) => &s.fields,
                _ => bcx.sess().bug("struct has non-univariant repr"),
            };

            let iter = src_fields.iter().zip(dst_fields).enumerate();
            for (i, (src_fty, dst_fty)) in iter {
                if type_is_zero_size(bcx.ccx(), dst_fty) {
                    continue;
                }

                let src_f = adt::trans_field_ptr(bcx, &src_repr, src, 0, i);
                let dst_f = adt::trans_field_ptr(bcx, &dst_repr, dst, 0, i);
                if src_fty == dst_fty {
                    memcpy_ty(bcx, dst_f, src_f, src_fty);
                } else {
                    coerce_unsized_into(bcx, src_f, src_fty, dst_f, dst_fty);
                }
            }
        }
        _ => bcx.sess().bug(&format!("coerce_unsized_into: invalid coercion {:?} -> {:?}",
                                     src_ty,
                                     dst_ty)),
    }
}

pub fn cast_shift_expr_rhs(cx: Block, op: hir::BinOp_, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs, |a, b| Trunc(cx, a, b), |a, b| ZExt(cx, a, b))
}

pub fn cast_shift_const_rhs(op: hir::BinOp_, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op,
                   lhs,
                   rhs,
                   |a, b| unsafe { llvm::LLVMConstTrunc(a, b.to_ref()) },
                   |a, b| unsafe { llvm::LLVMConstZExt(a, b.to_ref()) })
}

fn cast_shift_rhs<F, G>(op: hir::BinOp_,
                        lhs: ValueRef,
                        rhs: ValueRef,
                        trunc: F,
                        zext: G)
                        -> ValueRef
    where F: FnOnce(ValueRef, Type) -> ValueRef,
          G: FnOnce(ValueRef, Type) -> ValueRef
{
    // Shifts may have any size int on the rhs
    if rustc_front::util::is_shift_binop(op) {
        let mut rhs_llty = val_ty(rhs);
        let mut lhs_llty = val_ty(lhs);
        if rhs_llty.kind() == Vector {
            rhs_llty = rhs_llty.element_type()
        }
        if lhs_llty.kind() == Vector {
            lhs_llty = lhs_llty.element_type()
        }
        let rhs_sz = rhs_llty.int_width();
        let lhs_sz = lhs_llty.int_width();
        if lhs_sz < rhs_sz {
            trunc(rhs, lhs_llty)
        } else if lhs_sz > rhs_sz {
            // FIXME (#1877: If shifting by negative
            // values becomes not undefined then this is wrong.
            zext(rhs, lhs_llty)
        } else {
            rhs
        }
    } else {
        rhs
    }
}

pub fn llty_and_min_for_signed_ty<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                              val_t: Ty<'tcx>)
                                              -> (Type, u64) {
    match val_t.sty {
        ty::TyInt(t) => {
            let llty = Type::int_from_ty(cx.ccx(), t);
            let min = match t {
                ast::TyIs if llty == Type::i32(cx.ccx()) => i32::MIN as u64,
                ast::TyIs => i64::MIN as u64,
                ast::TyI8 => i8::MIN as u64,
                ast::TyI16 => i16::MIN as u64,
                ast::TyI32 => i32::MIN as u64,
                ast::TyI64 => i64::MIN as u64,
            };
            (llty, min)
        }
        _ => unreachable!(),
    }
}

pub fn fail_if_zero_or_overflows<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                             call_info: NodeIdAndSpan,
                                             divrem: hir::BinOp,
                                             lhs: ValueRef,
                                             rhs: ValueRef,
                                             rhs_t: Ty<'tcx>)
                                             -> Block<'blk, 'tcx> {
    let (zero_text, overflow_text) = if divrem.node == hir::BiDiv {
        ("attempted to divide by zero",
         "attempted to divide with overflow")
    } else {
        ("attempted remainder with a divisor of zero",
         "attempted remainder with overflow")
    };
    let debug_loc = call_info.debug_loc();

    let (is_zero, is_signed) = match rhs_t.sty {
        ty::TyInt(t) => {
            let zero = C_integral(Type::int_from_ty(cx.ccx(), t), 0, false);
            (ICmp(cx, llvm::IntEQ, rhs, zero, debug_loc), true)
        }
        ty::TyUint(t) => {
            let zero = C_integral(Type::uint_from_ty(cx.ccx(), t), 0, false);
            (ICmp(cx, llvm::IntEQ, rhs, zero, debug_loc), false)
        }
        ty::TyStruct(def, _) if def.is_simd() => {
            let mut res = C_bool(cx.ccx(), false);
            for i in 0..rhs_t.simd_size(cx.tcx()) {
                res = Or(cx,
                         res,
                         IsNull(cx, ExtractElement(cx, rhs, C_int(cx.ccx(), i as i64))),
                         debug_loc);
            }
            (res, false)
        }
        _ => {
            cx.sess().bug(&format!("fail-if-zero on unexpected type: {}", rhs_t));
        }
    };
    let bcx = with_cond(cx, is_zero, |bcx| {
        controlflow::trans_fail(bcx, call_info, InternedString::new(zero_text))
    });

    // To quote LLVM's documentation for the sdiv instruction:
    //
    //      Division by zero leads to undefined behavior. Overflow also leads
    //      to undefined behavior; this is a rare case, but can occur, for
    //      example, by doing a 32-bit division of -2147483648 by -1.
    //
    // In order to avoid undefined behavior, we perform runtime checks for
    // signed division/remainder which would trigger overflow. For unsigned
    // integers, no action beyond checking for zero need be taken.
    if is_signed {
        let (llty, min) = llty_and_min_for_signed_ty(cx, rhs_t);
        let minus_one = ICmp(bcx,
                             llvm::IntEQ,
                             rhs,
                             C_integral(llty, !0, false),
                             debug_loc);
        with_cond(bcx, minus_one, |bcx| {
            let is_min = ICmp(bcx,
                              llvm::IntEQ,
                              lhs,
                              C_integral(llty, min, true),
                              debug_loc);
            with_cond(bcx, is_min, |bcx| {
                controlflow::trans_fail(bcx, call_info, InternedString::new(overflow_text))
            })
        })
    } else {
        bcx
    }
}

pub fn trans_external_path<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                     did: DefId,
                                     t: Ty<'tcx>)
                                     -> ValueRef {
    let name = ccx.sess().cstore.item_symbol(did);
    match t.sty {
        ty::TyBareFn(_, ref fn_ty) => {
            match ccx.sess().target.target.adjust_abi(fn_ty.abi) {
                Rust | RustCall => {
                    get_extern_rust_fn(ccx, t, &name[..], did)
                }
                RustIntrinsic | PlatformIntrinsic => {
                    ccx.sess().bug("unexpected intrinsic in trans_external_path")
                }
                _ => {
                    let attrs = ccx.sess().cstore.item_attrs(did);
                    foreign::register_foreign_item_fn(ccx, fn_ty.abi, t, &name, &attrs)
                }
            }
        }
        _ => {
            get_extern_const(ccx, did, t)
        }
    }
}

pub fn invoke<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                          llfn: ValueRef,
                          llargs: &[ValueRef],
                          fn_ty: Ty<'tcx>,
                          debug_loc: DebugLoc)
                          -> (ValueRef, Block<'blk, 'tcx>) {
    let _icx = push_ctxt("invoke_");
    if bcx.unreachable.get() {
        return (C_null(Type::i8(bcx.ccx())), bcx);
    }

    let attributes = attributes::from_fn_type(bcx.ccx(), fn_ty);

    match bcx.opt_node_id {
        None => {
            debug!("invoke at ???");
        }
        Some(id) => {
            debug!("invoke at {}", bcx.tcx().map.node_to_string(id));
        }
    }

    if need_invoke(bcx) {
        debug!("invoking {} at {:?}", bcx.val_to_string(llfn), bcx.llbb);
        for &llarg in llargs {
            debug!("arg: {}", bcx.val_to_string(llarg));
        }
        let normal_bcx = bcx.fcx.new_temp_block("normal-return");
        let landing_pad = bcx.fcx.get_landing_pad();

        let llresult = Invoke(bcx,
                              llfn,
                              &llargs[..],
                              normal_bcx.llbb,
                              landing_pad,
                              Some(attributes),
                              debug_loc);
        return (llresult, normal_bcx);
    } else {
        debug!("calling {} at {:?}", bcx.val_to_string(llfn), bcx.llbb);
        for &llarg in llargs {
            debug!("arg: {}", bcx.val_to_string(llarg));
        }

        let llresult = Call(bcx, llfn, &llargs[..], Some(attributes), debug_loc);
        return (llresult, bcx);
    }
}

/// Returns whether this session's target will use SEH-based unwinding.
///
/// This is only true for MSVC targets, and even then the 64-bit MSVC target
/// currently uses SEH-ish unwinding with DWARF info tables to the side (same as
/// 64-bit MinGW) instead of "full SEH".
pub fn wants_msvc_seh(sess: &Session) -> bool {
    sess.target.target.options.is_like_msvc && sess.target.target.arch == "x86"
}

pub fn need_invoke(bcx: Block) -> bool {
    // FIXME(#25869) currently SEH-based unwinding is pretty buggy in LLVM and
    //               is being overhauled as this is being written. Until that
    //               time such that upstream LLVM's implementation is more solid
    //               and we start binding it we need to skip invokes for any
    //               target which wants SEH-based unwinding.
    if bcx.sess().no_landing_pads() || wants_msvc_seh(bcx.sess()) {
        return false;
    }

    // Avoid using invoke if we are already inside a landing pad.
    if bcx.is_lpad {
        return false;
    }

    bcx.fcx.needs_invoke()
}

pub fn load_if_immediate<'blk, 'tcx>(cx: Block<'blk, 'tcx>, v: ValueRef, t: Ty<'tcx>) -> ValueRef {
    let _icx = push_ctxt("load_if_immediate");
    if type_is_immediate(cx.ccx(), t) {
        return load_ty(cx, v, t);
    }
    return v;
}

/// Helper for loading values from memory. Does the necessary conversion if the in-memory type
/// differs from the type used for SSA values. Also handles various special cases where the type
/// gives us better information about what we are loading.
pub fn load_ty<'blk, 'tcx>(cx: Block<'blk, 'tcx>, ptr: ValueRef, t: Ty<'tcx>) -> ValueRef {
    if cx.unreachable.get() || type_is_zero_size(cx.ccx(), t) {
        return C_undef(type_of::type_of(cx.ccx(), t));
    }

    let ptr = to_arg_ty_ptr(cx, ptr, t);
    let align = type_of::align_of(cx.ccx(), t);

    if type_is_immediate(cx.ccx(), t) && type_of::type_of(cx.ccx(), t).is_aggregate() {
        let load = Load(cx, ptr);
        unsafe {
            llvm::LLVMSetAlignment(load, align);
        }
        return load;
    }

    unsafe {
        let global = llvm::LLVMIsAGlobalVariable(ptr);
        if !global.is_null() && llvm::LLVMIsGlobalConstant(global) == llvm::True {
            let val = llvm::LLVMGetInitializer(global);
            if !val.is_null() {
                return to_arg_ty(cx, val, t);
            }
        }
    }

    let val = if t.is_bool() {
        LoadRangeAssert(cx, ptr, 0, 2, llvm::False)
    } else if t.is_char() {
        // a char is a Unicode codepoint, and so takes values from 0
        // to 0x10FFFF inclusive only.
        LoadRangeAssert(cx, ptr, 0, 0x10FFFF + 1, llvm::False)
    } else if (t.is_region_ptr() || t.is_unique()) && !common::type_is_fat_ptr(cx.tcx(), t) {
        LoadNonNull(cx, ptr)
    } else {
        Load(cx, ptr)
    };

    unsafe {
        llvm::LLVMSetAlignment(val, align);
    }

    to_arg_ty(cx, val, t)
}

/// Helper for storing values in memory. Does the necessary conversion if the in-memory type
/// differs from the type used for SSA values.
pub fn store_ty<'blk, 'tcx>(cx: Block<'blk, 'tcx>, v: ValueRef, dst: ValueRef, t: Ty<'tcx>) {
    if cx.unreachable.get() {
        return;
    }

    debug!("store_ty: {} : {:?} <- {}",
           cx.val_to_string(dst),
           t,
           cx.val_to_string(v));

    if common::type_is_fat_ptr(cx.tcx(), t) {
        Store(cx,
              ExtractValue(cx, v, abi::FAT_PTR_ADDR),
              expr::get_dataptr(cx, dst));
        Store(cx,
              ExtractValue(cx, v, abi::FAT_PTR_EXTRA),
              expr::get_meta(cx, dst));
    } else {
        let store = Store(cx, from_arg_ty(cx, v, t), to_arg_ty_ptr(cx, dst, t));
        unsafe {
            llvm::LLVMSetAlignment(store, type_of::align_of(cx.ccx(), t));
        }
    }
}

pub fn store_fat_ptr<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                 data: ValueRef,
                                 extra: ValueRef,
                                 dst: ValueRef,
                                 _ty: Ty<'tcx>) {
    // FIXME: emit metadata
    Store(cx, data, expr::get_dataptr(cx, dst));
    Store(cx, extra, expr::get_meta(cx, dst));
}

pub fn load_fat_ptr<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                src: ValueRef,
                                _ty: Ty<'tcx>)
                                -> (ValueRef, ValueRef) {
    // FIXME: emit metadata
    (Load(cx, expr::get_dataptr(cx, src)),
     Load(cx, expr::get_meta(cx, src)))
}

pub fn from_arg_ty(bcx: Block, val: ValueRef, ty: Ty) -> ValueRef {
    if ty.is_bool() {
        ZExt(bcx, val, Type::i8(bcx.ccx()))
    } else {
        val
    }
}

pub fn to_arg_ty(bcx: Block, val: ValueRef, ty: Ty) -> ValueRef {
    if ty.is_bool() {
        Trunc(bcx, val, Type::i1(bcx.ccx()))
    } else {
        val
    }
}

pub fn to_arg_ty_ptr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, ptr: ValueRef, ty: Ty<'tcx>) -> ValueRef {
    if type_is_immediate(bcx.ccx(), ty) && type_of::type_of(bcx.ccx(), ty).is_aggregate() {
        // We want to pass small aggregates as immediate values, but using an aggregate LLVM type
        // for this leads to bad optimizations, so its arg type is an appropriately sized integer
        // and we have to convert it
        BitCast(bcx, ptr, type_of::arg_type_of(bcx.ccx(), ty).ptr_to())
    } else {
        ptr
    }
}

pub fn init_local<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, local: &hir::Local) -> Block<'blk, 'tcx> {
    debug!("init_local(bcx={}, local.id={})", bcx.to_str(), local.id);
    let _indenter = indenter();
    let _icx = push_ctxt("init_local");
    _match::store_local(bcx, local)
}

pub fn raw_block<'blk, 'tcx>(fcx: &'blk FunctionContext<'blk, 'tcx>,
                             is_lpad: bool,
                             llbb: BasicBlockRef)
                             -> Block<'blk, 'tcx> {
    common::BlockS::new(llbb, is_lpad, None, fcx)
}

pub fn with_cond<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>, val: ValueRef, f: F) -> Block<'blk, 'tcx>
    where F: FnOnce(Block<'blk, 'tcx>) -> Block<'blk, 'tcx>
{
    let _icx = push_ctxt("with_cond");

    if bcx.unreachable.get() || common::const_to_opt_uint(val) == Some(0) {
        return bcx;
    }

    let fcx = bcx.fcx;
    let next_cx = fcx.new_temp_block("next");
    let cond_cx = fcx.new_temp_block("cond");
    CondBr(bcx, val, cond_cx.llbb, next_cx.llbb, DebugLoc::None);
    let after_cx = f(cond_cx);
    if !after_cx.terminated.get() {
        Br(after_cx, next_cx.llbb, DebugLoc::None);
    }
    next_cx
}

pub fn call_lifetime_start(cx: Block, ptr: ValueRef) {
    if cx.sess().opts.optimize == config::No {
        return;
    }

    let _icx = push_ctxt("lifetime_start");
    let ccx = cx.ccx();

    let size = machine::llsize_of_alloc(ccx, val_ty(ptr).element_type());
    if size == 0 {
        return;
    }

    let ptr = PointerCast(cx, ptr, Type::i8p(ccx));
    let lifetime_start = ccx.get_intrinsic(&"llvm.lifetime.start");
    Call(cx,
         lifetime_start,
         &[C_u64(ccx, size), ptr],
         None,
         DebugLoc::None);
}

pub fn call_lifetime_end(cx: Block, ptr: ValueRef) {
    if cx.sess().opts.optimize == config::No {
        return;
    }

    let _icx = push_ctxt("lifetime_end");
    let ccx = cx.ccx();

    let size = machine::llsize_of_alloc(ccx, val_ty(ptr).element_type());
    if size == 0 {
        return;
    }

    let ptr = PointerCast(cx, ptr, Type::i8p(ccx));
    let lifetime_end = ccx.get_intrinsic(&"llvm.lifetime.end");
    Call(cx,
         lifetime_end,
         &[C_u64(ccx, size), ptr],
         None,
         DebugLoc::None);
}

// Generates code for resumption of unwind at the end of a landing pad.
pub fn trans_unwind_resume(bcx: Block, lpval: ValueRef) {
    if !bcx.sess().target.target.options.custom_unwind_resume {
        Resume(bcx, lpval);
    } else {
        let exc_ptr = ExtractValue(bcx, lpval, 0);
        let llunwresume = bcx.fcx.eh_unwind_resume();
        Call(bcx, llunwresume, &[exc_ptr], None, DebugLoc::None);
        Unreachable(bcx);
    }
}


pub fn call_memcpy(cx: Block, dst: ValueRef, src: ValueRef, n_bytes: ValueRef, align: u32) {
    let _icx = push_ctxt("call_memcpy");
    let ccx = cx.ccx();
    let ptr_width = &ccx.sess().target.target.target_pointer_width[..];
    let key = format!("llvm.memcpy.p0i8.p0i8.i{}", ptr_width);
    let memcpy = ccx.get_intrinsic(&key);
    let src_ptr = PointerCast(cx, src, Type::i8p(ccx));
    let dst_ptr = PointerCast(cx, dst, Type::i8p(ccx));
    let size = IntCast(cx, n_bytes, ccx.int_type());
    let align = C_i32(ccx, align as i32);
    let volatile = C_bool(ccx, false);
    Call(cx,
         memcpy,
         &[dst_ptr, src_ptr, size, align, volatile],
         None,
         DebugLoc::None);
}

pub fn memcpy_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, dst: ValueRef, src: ValueRef, t: Ty<'tcx>) {
    let _icx = push_ctxt("memcpy_ty");
    let ccx = bcx.ccx();

    if type_is_zero_size(ccx, t) {
        return;
    }

    if t.is_structural() {
        let llty = type_of::type_of(ccx, t);
        let llsz = llsize_of(ccx, llty);
        let llalign = type_of::align_of(ccx, t);
        call_memcpy(bcx, dst, src, llsz, llalign as u32);
    } else if common::type_is_fat_ptr(bcx.tcx(), t) {
        let (data, extra) = load_fat_ptr(bcx, src, t);
        store_fat_ptr(bcx, data, extra, dst, t);
    } else {
        store_ty(bcx, load_ty(bcx, src, t), dst, t);
    }
}

pub fn drop_done_fill_mem<'blk, 'tcx>(cx: Block<'blk, 'tcx>, llptr: ValueRef, t: Ty<'tcx>) {
    if cx.unreachable.get() {
        return;
    }
    let _icx = push_ctxt("drop_done_fill_mem");
    let bcx = cx;
    memfill(&B(bcx), llptr, t, adt::DTOR_DONE);
}

pub fn init_zero_mem<'blk, 'tcx>(cx: Block<'blk, 'tcx>, llptr: ValueRef, t: Ty<'tcx>) {
    if cx.unreachable.get() {
        return;
    }
    let _icx = push_ctxt("init_zero_mem");
    let bcx = cx;
    memfill(&B(bcx), llptr, t, 0);
}

// Always use this function instead of storing a constant byte to the memory
// in question. e.g. if you store a zero constant, LLVM will drown in vreg
// allocation for large data structures, and the generated code will be
// awful. (A telltale sign of this is large quantities of
// `mov [byte ptr foo],0` in the generated code.)
fn memfill<'a, 'tcx>(b: &Builder<'a, 'tcx>, llptr: ValueRef, ty: Ty<'tcx>, byte: u8) {
    let _icx = push_ctxt("memfill");
    let ccx = b.ccx;

    let llty = type_of::type_of(ccx, ty);
    let ptr_width = &ccx.sess().target.target.target_pointer_width[..];
    let intrinsic_key = format!("llvm.memset.p0i8.i{}", ptr_width);

    let llintrinsicfn = ccx.get_intrinsic(&intrinsic_key);
    let llptr = b.pointercast(llptr, Type::i8(ccx).ptr_to());
    let llzeroval = C_u8(ccx, byte);
    let size = machine::llsize_of(ccx, llty);
    let align = C_i32(ccx, type_of::align_of(ccx, ty) as i32);
    let volatile = C_bool(ccx, false);
    b.call(llintrinsicfn,
           &[llptr, llzeroval, size, align, volatile],
           None);
}

pub fn alloc_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, t: Ty<'tcx>, name: &str) -> ValueRef {
    let _icx = push_ctxt("alloc_ty");
    let ccx = bcx.ccx();
    let ty = type_of::type_of(ccx, t);
    assert!(!t.has_param_types());
    alloca(bcx, ty, name)
}

pub fn alloca(cx: Block, ty: Type, name: &str) -> ValueRef {
    let _icx = push_ctxt("alloca");
    if cx.unreachable.get() {
        unsafe {
            return llvm::LLVMGetUndef(ty.ptr_to().to_ref());
        }
    }
    debuginfo::clear_source_location(cx.fcx);
    Alloca(cx, ty, name)
}

pub fn set_value_name(val: ValueRef, name: &str) {
    unsafe {
        let name = CString::new(name).unwrap();
        llvm::LLVMSetValueName(val, name.as_ptr());
    }
}

// Creates the alloca slot which holds the pointer to the slot for the final return value
pub fn make_return_slot_pointer<'a, 'tcx>(fcx: &FunctionContext<'a, 'tcx>,
                                          output_type: Ty<'tcx>)
                                          -> ValueRef {
    let lloutputtype = type_of::type_of(fcx.ccx, output_type);

    // We create an alloca to hold a pointer of type `output_type`
    // which will hold the pointer to the right alloca which has the
    // final ret value
    if fcx.needs_ret_allocas {
        // Let's create the stack slot
        let slot = AllocaFcx(fcx, lloutputtype.ptr_to(), "llretslotptr");

        // and if we're using an out pointer, then store that in our newly made slot
        if type_of::return_uses_outptr(fcx.ccx, output_type) {
            let outptr = get_param(fcx.llfn, 0);

            let b = fcx.ccx.builder();
            b.position_before(fcx.alloca_insert_pt.get().unwrap());
            b.store(outptr, slot);
        }

        slot

    // But if there are no nested returns, we skip the indirection and have a single
    // retslot
    } else {
        if type_of::return_uses_outptr(fcx.ccx, output_type) {
            get_param(fcx.llfn, 0)
        } else {
            AllocaFcx(fcx, lloutputtype, "sret_slot")
        }
    }
}

struct FindNestedReturn {
    found: bool,
}

impl FindNestedReturn {
    fn new() -> FindNestedReturn {
        FindNestedReturn {
            found: false,
        }
    }
}

impl<'v> Visitor<'v> for FindNestedReturn {
    fn visit_expr(&mut self, e: &hir::Expr) {
        match e.node {
            hir::ExprRet(..) => {
                self.found = true;
            }
            _ => intravisit::walk_expr(self, e),
        }
    }
}

fn build_cfg(tcx: &ty::ctxt, id: ast::NodeId) -> (ast::NodeId, Option<cfg::CFG>) {
    let blk = match tcx.map.find(id) {
        Some(hir_map::NodeItem(i)) => {
            match i.node {
                hir::ItemFn(_, _, _, _, _, ref blk) => {
                    blk
                }
                _ => tcx.sess.bug("unexpected item variant in has_nested_returns"),
            }
        }
        Some(hir_map::NodeTraitItem(trait_item)) => {
            match trait_item.node {
                hir::MethodTraitItem(_, Some(ref body)) => body,
                _ => {
                    tcx.sess.bug("unexpected variant: trait item other than a provided method in \
                                  has_nested_returns")
                }
            }
        }
        Some(hir_map::NodeImplItem(impl_item)) => {
            match impl_item.node {
                hir::ImplItemKind::Method(_, ref body) => body,
                _ => {
                    tcx.sess.bug("unexpected variant: non-method impl item in has_nested_returns")
                }
            }
        }
        Some(hir_map::NodeExpr(e)) => {
            match e.node {
                hir::ExprClosure(_, _, ref blk) => blk,
                _ => tcx.sess.bug("unexpected expr variant in has_nested_returns"),
            }
        }
        Some(hir_map::NodeVariant(..)) |
        Some(hir_map::NodeStructCtor(..)) => return (ast::DUMMY_NODE_ID, None),

        // glue, shims, etc
        None if id == ast::DUMMY_NODE_ID => return (ast::DUMMY_NODE_ID, None),

        _ => tcx.sess.bug(&format!("unexpected variant in has_nested_returns: {}",
                                   tcx.map.path_to_string(id))),
    };

    (blk.id, Some(cfg::CFG::new(tcx, blk)))
}

// Checks for the presence of "nested returns" in a function.
// Nested returns are when the inner expression of a return expression
// (the 'expr' in 'return expr') contains a return expression. Only cases
// where the outer return is actually reachable are considered. Implicit
// returns from the end of blocks are considered as well.
//
// This check is needed to handle the case where the inner expression is
// part of a larger expression that may have already partially-filled the
// return slot alloca. This can cause errors related to clean-up due to
// the clobbering of the existing value in the return slot.
fn has_nested_returns(tcx: &ty::ctxt, cfg: &cfg::CFG, blk_id: ast::NodeId) -> bool {
    for index in cfg.graph.depth_traverse(cfg.entry) {
        let n = cfg.graph.node_data(index);
        match tcx.map.find(n.id()) {
            Some(hir_map::NodeExpr(ex)) => {
                if let hir::ExprRet(Some(ref ret_expr)) = ex.node {
                    let mut visitor = FindNestedReturn::new();
                    intravisit::walk_expr(&mut visitor, &**ret_expr);
                    if visitor.found {
                        return true;
                    }
                }
            }
            Some(hir_map::NodeBlock(blk)) if blk.id == blk_id => {
                let mut visitor = FindNestedReturn::new();
                walk_list!(&mut visitor, visit_expr, &blk.expr);
                if visitor.found {
                    return true;
                }
            }
            _ => {}
        }
    }

    return false;
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_datums_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
//
// Be warned! You must call `init_function` before doing anything with the
// returned function context.
pub fn new_fn_ctxt<'a, 'tcx>(ccx: &'a CrateContext<'a, 'tcx>,
                             llfndecl: ValueRef,
                             id: ast::NodeId,
                             has_env: bool,
                             output_type: ty::FnOutput<'tcx>,
                             param_substs: &'tcx Substs<'tcx>,
                             sp: Option<Span>,
                             block_arena: &'a TypedArena<common::BlockS<'a, 'tcx>>)
                             -> FunctionContext<'a, 'tcx> {
    common::validate_substs(param_substs);

    debug!("new_fn_ctxt(path={}, id={}, param_substs={:?})",
           if id == !0 {
               "".to_string()
           } else {
               ccx.tcx().map.path_to_string(id).to_string()
           },
           id,
           param_substs);

    let uses_outptr = match output_type {
        ty::FnConverging(output_type) => {
            let substd_output_type = monomorphize::apply_param_substs(ccx.tcx(),
                                                                      param_substs,
                                                                      &output_type);
            type_of::return_uses_outptr(ccx, substd_output_type)
        }
        ty::FnDiverging => false,
    };
    let debug_context = debuginfo::create_function_debug_context(ccx, id, param_substs, llfndecl);
    let (blk_id, cfg) = build_cfg(ccx.tcx(), id);
    let nested_returns = if let Some(ref cfg) = cfg {
        has_nested_returns(ccx.tcx(), cfg, blk_id)
    } else {
        false
    };

    let mir = ccx.mir_map().get(&id);

    let mut fcx = FunctionContext {
        mir: mir,
        llfn: llfndecl,
        llenv: None,
        llretslotptr: Cell::new(None),
        param_env: ccx.tcx().empty_parameter_environment(),
        alloca_insert_pt: Cell::new(None),
        llreturn: Cell::new(None),
        needs_ret_allocas: nested_returns,
        personality: Cell::new(None),
        caller_expects_out_pointer: uses_outptr,
        lllocals: RefCell::new(NodeMap()),
        llupvars: RefCell::new(NodeMap()),
        lldropflag_hints: RefCell::new(DropFlagHintsMap::new()),
        id: id,
        param_substs: param_substs,
        span: sp,
        block_arena: block_arena,
        ccx: ccx,
        debug_context: debug_context,
        scopes: RefCell::new(Vec::new()),
        cfg: cfg,
    };

    if has_env {
        fcx.llenv = Some(get_param(fcx.llfn, fcx.env_arg_pos() as c_uint))
    }

    fcx
}

/// Performs setup on a newly created function, creating the entry scope block
/// and allocating space for the return pointer.
pub fn init_function<'a, 'tcx>(fcx: &'a FunctionContext<'a, 'tcx>,
                               skip_retptr: bool,
                               output: ty::FnOutput<'tcx>)
                               -> Block<'a, 'tcx> {
    let entry_bcx = fcx.new_temp_block("entry-block");

    // Use a dummy instruction as the insertion point for all allocas.
    // This is later removed in FunctionContext::cleanup.
    fcx.alloca_insert_pt.set(Some(unsafe {
        Load(entry_bcx, C_null(Type::i8p(fcx.ccx)));
        llvm::LLVMGetFirstInstruction(entry_bcx.llbb)
    }));

    if let ty::FnConverging(output_type) = output {
        // This shouldn't need to recompute the return type,
        // as new_fn_ctxt did it already.
        let substd_output_type = fcx.monomorphize(&output_type);
        if !return_type_is_void(fcx.ccx, substd_output_type) {
            // If the function returns nil/bot, there is no real return
            // value, so do not set `llretslotptr`.
            if !skip_retptr || fcx.caller_expects_out_pointer {
                // Otherwise, we normally allocate the llretslotptr, unless we
                // have been instructed to skip it for immediate return
                // values.
                fcx.llretslotptr.set(Some(make_return_slot_pointer(fcx, substd_output_type)));
            }
        }
    }

    // Create the drop-flag hints for every unfragmented path in the function.
    let tcx = fcx.ccx.tcx();
    let fn_did = tcx.map.local_def_id(fcx.id);
    let mut hints = fcx.lldropflag_hints.borrow_mut();
    let fragment_infos = tcx.fragment_infos.borrow();

    // Intern table for drop-flag hint datums.
    let mut seen = HashMap::new();

    if let Some(fragment_infos) = fragment_infos.get(&fn_did) {
        for &info in fragment_infos {

            let make_datum = |id| {
                let init_val = C_u8(fcx.ccx, adt::DTOR_NEEDED_HINT);
                let llname = &format!("dropflag_hint_{}", id);
                debug!("adding hint {}", llname);
                let ty = tcx.types.u8;
                let ptr = alloc_ty(entry_bcx, ty, llname);
                Store(entry_bcx, init_val, ptr);
                let flag = datum::Lvalue::new_dropflag_hint("base::init_function");
                datum::Datum::new(ptr, ty, flag)
            };

            let (var, datum) = match info {
                ty::FragmentInfo::Moved { var, .. } |
                ty::FragmentInfo::Assigned { var, .. } => {
                    let datum = seen.get(&var).cloned().unwrap_or_else(|| {
                        let datum = make_datum(var);
                        seen.insert(var, datum.clone());
                        datum
                    });
                    (var, datum)
                }
            };
            match info {
                ty::FragmentInfo::Moved { move_expr: expr_id, .. } => {
                    debug!("FragmentInfo::Moved insert drop hint for {}", expr_id);
                    hints.insert(expr_id, DropHint::new(var, datum));
                }
                ty::FragmentInfo::Assigned { assignee_id: expr_id, .. } => {
                    debug!("FragmentInfo::Assigned insert drop hint for {}", expr_id);
                    hints.insert(expr_id, DropHint::new(var, datum));
                }
            }
        }
    }

    entry_bcx
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_datums_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

pub fn arg_kind<'a, 'tcx>(cx: &FunctionContext<'a, 'tcx>, t: Ty<'tcx>) -> datum::Rvalue {
    use trans::datum::{ByRef, ByValue};

    datum::Rvalue {
        mode: if arg_is_indirect(cx.ccx, t) { ByRef } else { ByValue }
    }
}

// create_datums_for_fn_args: creates lvalue datums for each of the
// incoming function arguments.
pub fn create_datums_for_fn_args<'a, 'tcx>(mut bcx: Block<'a, 'tcx>,
                                           args: &[hir::Arg],
                                           arg_tys: &[Ty<'tcx>],
                                           has_tupled_arg: bool,
                                           arg_scope: cleanup::CustomScopeIndex)
                                           -> Block<'a, 'tcx> {
    let _icx = push_ctxt("create_datums_for_fn_args");
    let fcx = bcx.fcx;
    let arg_scope_id = cleanup::CustomScope(arg_scope);

    // Return an array wrapping the ValueRefs that we get from `get_param` for
    // each argument into datums.
    //
    // For certain mode/type combinations, the raw llarg values are passed
    // by value.  However, within the fn body itself, we want to always
    // have all locals and arguments be by-ref so that we can cancel the
    // cleanup and for better interaction with LLVM's debug info.  So, if
    // the argument would be passed by value, we store it into an alloca.
    // This alloca should be optimized away by LLVM's mem-to-reg pass in
    // the event it's not truly needed.
    let mut idx = fcx.arg_offset() as c_uint;
    for (i, &arg_ty) in arg_tys.iter().enumerate() {
        let arg_datum = if !has_tupled_arg || i < arg_tys.len() - 1 {
            if type_of::arg_is_indirect(bcx.ccx(), arg_ty) &&
               bcx.sess().opts.debuginfo != FullDebugInfo {
                // Don't copy an indirect argument to an alloca, the caller
                // already put it in a temporary alloca and gave it up, unless
                // we emit extra-debug-info, which requires local allocas :(.
                let llarg = get_param(fcx.llfn, idx);
                idx += 1;
                bcx.fcx.schedule_lifetime_end(arg_scope_id, llarg);
                bcx.fcx.schedule_drop_mem(arg_scope_id, llarg, arg_ty, None);

                datum::Datum::new(llarg,
                                  arg_ty,
                                  datum::Lvalue::new("create_datum_for_fn_args"))
            } else if common::type_is_fat_ptr(bcx.tcx(), arg_ty) {
                let data = get_param(fcx.llfn, idx);
                let extra = get_param(fcx.llfn, idx + 1);
                idx += 2;
                unpack_datum!(bcx, datum::lvalue_scratch_datum(bcx, arg_ty, "",
                                                        arg_scope_id, (data, extra),
                                                        |(data, extra), bcx, dst| {
                    Store(bcx, data, expr::get_dataptr(bcx, dst));
                    Store(bcx, extra, expr::get_meta(bcx, dst));
                    bcx
                }))
            } else {
                let llarg = get_param(fcx.llfn, idx);
                idx += 1;
                let tmp = datum::Datum::new(llarg, arg_ty, arg_kind(fcx, arg_ty));
                unpack_datum!(bcx,
                              datum::lvalue_scratch_datum(bcx,
                                                          arg_ty,
                                                          "",
                                                          arg_scope_id,
                                                          tmp,
                                                          |tmp, bcx, dst| tmp.store_to(bcx, dst)))
            }
        } else {
            // FIXME(pcwalton): Reduce the amount of code bloat this is responsible for.
            match arg_ty.sty {
                ty::TyTuple(ref tupled_arg_tys) => {
                    unpack_datum!(bcx,
                                  datum::lvalue_scratch_datum(bcx,
                                                              arg_ty,
                                                              "tupled_args",
                                                              arg_scope_id,
                                                              (),
                                                              |(),
                                                               mut bcx,
                                                               llval| {
                        for (j, &tupled_arg_ty) in
                                    tupled_arg_tys.iter().enumerate() {
                            let lldest = StructGEP(bcx, llval, j);
                            if common::type_is_fat_ptr(bcx.tcx(), tupled_arg_ty) {
                                let data = get_param(bcx.fcx.llfn, idx);
                                let extra = get_param(bcx.fcx.llfn, idx + 1);
                                Store(bcx, data, expr::get_dataptr(bcx, lldest));
                                Store(bcx, extra, expr::get_meta(bcx, lldest));
                                idx += 2;
                            } else {
                                let datum = datum::Datum::new(
                                    get_param(bcx.fcx.llfn, idx),
                                    tupled_arg_ty,
                                    arg_kind(bcx.fcx, tupled_arg_ty));
                                idx += 1;
                                bcx = datum.store_to(bcx, lldest);
                            };
                        }
                        bcx
                    }))
                }
                _ => {
                    bcx.tcx()
                       .sess
                       .bug("last argument of a function with `rust-call` ABI isn't a tuple?!")
                }
            }
        };

        let pat = &*args[i].pat;
        bcx = if let Some(name) = simple_name(pat) {
            // Generate nicer LLVM for the common case of fn a pattern
            // like `x: T`
            set_value_name(arg_datum.val, &bcx.name(name));
            bcx.fcx.lllocals.borrow_mut().insert(pat.id, arg_datum);
            bcx
        } else {
            // General path. Copy out the values that are used in the
            // pattern.
            _match::bind_irrefutable_pat(bcx, pat, arg_datum.match_input(), arg_scope_id)
        };
        debuginfo::create_argument_metadata(bcx, &args[i]);
    }

    bcx
}

// Ties up the llstaticallocas -> llloadenv -> lltop edges,
// and builds the return block.
pub fn finish_fn<'blk, 'tcx>(fcx: &'blk FunctionContext<'blk, 'tcx>,
                             last_bcx: Block<'blk, 'tcx>,
                             retty: ty::FnOutput<'tcx>,
                             ret_debug_loc: DebugLoc) {
    let _icx = push_ctxt("finish_fn");

    let ret_cx = match fcx.llreturn.get() {
        Some(llreturn) => {
            if !last_bcx.terminated.get() {
                Br(last_bcx, llreturn, DebugLoc::None);
            }
            raw_block(fcx, false, llreturn)
        }
        None => last_bcx,
    };

    // This shouldn't need to recompute the return type,
    // as new_fn_ctxt did it already.
    let substd_retty = fcx.monomorphize(&retty);
    build_return_block(fcx, ret_cx, substd_retty, ret_debug_loc);

    debuginfo::clear_source_location(fcx);
    fcx.cleanup();
}

// Builds the return block for a function.
pub fn build_return_block<'blk, 'tcx>(fcx: &FunctionContext<'blk, 'tcx>,
                                      ret_cx: Block<'blk, 'tcx>,
                                      retty: ty::FnOutput<'tcx>,
                                      ret_debug_location: DebugLoc) {
    if fcx.llretslotptr.get().is_none() ||
       (!fcx.needs_ret_allocas && fcx.caller_expects_out_pointer) {
        return RetVoid(ret_cx, ret_debug_location);
    }

    let retslot = if fcx.needs_ret_allocas {
        Load(ret_cx, fcx.llretslotptr.get().unwrap())
    } else {
        fcx.llretslotptr.get().unwrap()
    };
    let retptr = Value(retslot);
    match retptr.get_dominating_store(ret_cx) {
        // If there's only a single store to the ret slot, we can directly return
        // the value that was stored and omit the store and the alloca
        Some(s) => {
            let retval = s.get_operand(0).unwrap().get();
            s.erase_from_parent();

            if retptr.has_no_uses() {
                retptr.erase_from_parent();
            }

            let retval = if retty == ty::FnConverging(fcx.ccx.tcx().types.bool) {
                Trunc(ret_cx, retval, Type::i1(fcx.ccx))
            } else {
                retval
            };

            if fcx.caller_expects_out_pointer {
                if let ty::FnConverging(retty) = retty {
                    store_ty(ret_cx, retval, get_param(fcx.llfn, 0), retty);
                }
                RetVoid(ret_cx, ret_debug_location)
            } else {
                Ret(ret_cx, retval, ret_debug_location)
            }
        }
        // Otherwise, copy the return value to the ret slot
        None => match retty {
            ty::FnConverging(retty) => {
                if fcx.caller_expects_out_pointer {
                    memcpy_ty(ret_cx, get_param(fcx.llfn, 0), retslot, retty);
                    RetVoid(ret_cx, ret_debug_location)
                } else {
                    Ret(ret_cx, load_ty(ret_cx, retslot, retty), ret_debug_location)
                }
            }
            ty::FnDiverging => {
                if fcx.caller_expects_out_pointer {
                    RetVoid(ret_cx, ret_debug_location)
                } else {
                    Ret(ret_cx, C_undef(Type::nil(fcx.ccx)), ret_debug_location)
                }
            }
        },
    }
}

/// Builds an LLVM function out of a source function.
///
/// If the function closes over its environment a closure will be returned.
pub fn trans_closure<'a, 'b, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                   decl: &hir::FnDecl,
                                   body: &hir::Block,
                                   llfndecl: ValueRef,
                                   param_substs: &'tcx Substs<'tcx>,
                                   fn_ast_id: ast::NodeId,
                                   attributes: &[ast::Attribute],
                                   output_type: ty::FnOutput<'tcx>,
                                   abi: Abi,
                                   closure_env: closure::ClosureEnv<'b>) {
    ccx.stats().n_closures.set(ccx.stats().n_closures.get() + 1);

    let _icx = push_ctxt("trans_closure");
    attributes::emit_uwtable(llfndecl, true);

    debug!("trans_closure(..., param_substs={:?})", param_substs);

    let has_env = match closure_env {
        closure::ClosureEnv::Closure(..) => true,
        closure::ClosureEnv::NotClosure => false,
    };

    let (arena, fcx): (TypedArena<_>, FunctionContext);
    arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      llfndecl,
                      fn_ast_id,
                      has_env,
                      output_type,
                      param_substs,
                      Some(body.span),
                      &arena);
    let mut bcx = init_function(&fcx, false, output_type);

    if attributes.iter().any(|item| item.check_name("rustc_mir")) {
        mir::trans_mir(bcx);
        fcx.cleanup();
        return;
    }

    // cleanup scope for the incoming arguments
    let fn_cleanup_debug_loc = debuginfo::get_cleanup_debug_loc_for_ast_node(ccx,
                                                                             fn_ast_id,
                                                                             body.span,
                                                                             true);
    let arg_scope = fcx.push_custom_cleanup_scope_with_debug_loc(fn_cleanup_debug_loc);

    let block_ty = node_id_type(bcx, body.id);

    // Set up arguments to the function.
    let monomorphized_arg_types = decl.inputs
                                      .iter()
                                      .map(|arg| node_id_type(bcx, arg.id))
                                      .collect::<Vec<_>>();
    for monomorphized_arg_type in &monomorphized_arg_types {
        debug!("trans_closure: monomorphized_arg_type: {:?}",
               monomorphized_arg_type);
    }
    debug!("trans_closure: function lltype: {}",
           bcx.fcx.ccx.tn().val_to_string(bcx.fcx.llfn));

    let has_tupled_arg = match closure_env {
        closure::ClosureEnv::NotClosure => abi == RustCall,
        _ => false,
    };

    bcx = create_datums_for_fn_args(bcx,
                                    &decl.inputs,
                                    &monomorphized_arg_types,
                                    has_tupled_arg,
                                    arg_scope);

    bcx = closure_env.load(bcx, cleanup::CustomScope(arg_scope));

    // Up until here, IR instructions for this function have explicitly not been annotated with
    // source code location, so we don't step into call setup code. From here on, source location
    // emitting should be enabled.
    debuginfo::start_emitting_source_locations(&fcx);

    let dest = match fcx.llretslotptr.get() {
        Some(_) => expr::SaveIn(fcx.get_ret_slot(bcx, ty::FnConverging(block_ty), "iret_slot")),
        None => {
            assert!(type_is_zero_size(bcx.ccx(), block_ty));
            expr::Ignore
        }
    };

    // This call to trans_block is the place where we bridge between
    // translation calls that don't have a return value (trans_crate,
    // trans_mod, trans_item, et cetera) and those that do
    // (trans_block, trans_expr, et cetera).
    bcx = controlflow::trans_block(bcx, body, dest);

    match dest {
        expr::SaveIn(slot) if fcx.needs_ret_allocas => {
            Store(bcx, slot, fcx.llretslotptr.get().unwrap());
        }
        _ => {}
    }

    match fcx.llreturn.get() {
        Some(_) => {
            Br(bcx, fcx.return_exit_block(), DebugLoc::None);
            fcx.pop_custom_cleanup_scope(arg_scope);
        }
        None => {
            // Microoptimization writ large: avoid creating a separate
            // llreturn basic block
            bcx = fcx.pop_and_trans_custom_cleanup_scope(bcx, arg_scope);
        }
    };

    // Put return block after all other blocks.
    // This somewhat improves single-stepping experience in debugger.
    unsafe {
        let llreturn = fcx.llreturn.get();
        if let Some(llreturn) = llreturn {
            llvm::LLVMMoveBasicBlockAfter(llreturn, bcx.llbb);
        }
    }

    let ret_debug_loc = DebugLoc::At(fn_cleanup_debug_loc.id, fn_cleanup_debug_loc.span);

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(&fcx, bcx, output_type, ret_debug_loc);
}

/// Creates an LLVM function corresponding to a source language function.
pub fn trans_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                          decl: &hir::FnDecl,
                          body: &hir::Block,
                          llfndecl: ValueRef,
                          param_substs: &'tcx Substs<'tcx>,
                          id: ast::NodeId,
                          attrs: &[ast::Attribute]) {
    let _s = StatRecorder::new(ccx, ccx.tcx().map.path_to_string(id).to_string());
    debug!("trans_fn(param_substs={:?})", param_substs);
    let _icx = push_ctxt("trans_fn");
    let fn_ty = ccx.tcx().node_id_to_type(id);
    let fn_ty = monomorphize::apply_param_substs(ccx.tcx(), param_substs, &fn_ty);
    let sig = fn_ty.fn_sig();
    let sig = ccx.tcx().erase_late_bound_regions(&sig);
    let sig = infer::normalize_associated_type(ccx.tcx(), &sig);
    let output_type = sig.output;
    let abi = fn_ty.fn_abi();
    trans_closure(ccx,
                  decl,
                  body,
                  llfndecl,
                  param_substs,
                  id,
                  attrs,
                  output_type,
                  abi,
                  closure::ClosureEnv::NotClosure);
}

pub fn trans_enum_variant<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    ctor_id: ast::NodeId,
                                    disr: ty::Disr,
                                    param_substs: &'tcx Substs<'tcx>,
                                    llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_enum_variant");

    trans_enum_variant_or_tuple_like_struct(ccx, ctor_id, disr, param_substs, llfndecl);
}

pub fn trans_named_tuple_constructor<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                                 ctor_ty: Ty<'tcx>,
                                                 disr: ty::Disr,
                                                 args: callee::CallArgs,
                                                 dest: expr::Dest,
                                                 debug_loc: DebugLoc)
                                                 -> Result<'blk, 'tcx> {

    let ccx = bcx.fcx.ccx;

    let sig = ccx.tcx().erase_late_bound_regions(&ctor_ty.fn_sig());
    let sig = infer::normalize_associated_type(ccx.tcx(), &sig);
    let result_ty = sig.output.unwrap();

    // Get location to store the result. If the user does not care about
    // the result, just make a stack slot
    let llresult = match dest {
        expr::SaveIn(d) => d,
        expr::Ignore => {
            if !type_is_zero_size(ccx, result_ty) {
                let llresult = alloc_ty(bcx, result_ty, "constructor_result");
                call_lifetime_start(bcx, llresult);
                llresult
            } else {
                C_undef(type_of::type_of(ccx, result_ty).ptr_to())
            }
        }
    };

    if !type_is_zero_size(ccx, result_ty) {
        match args {
            callee::ArgExprs(exprs) => {
                let fields = exprs.iter().map(|x| &**x).enumerate().collect::<Vec<_>>();
                bcx = expr::trans_adt(bcx,
                                      result_ty,
                                      disr,
                                      &fields[..],
                                      None,
                                      expr::SaveIn(llresult),
                                      debug_loc);
            }
            _ => ccx.sess().bug("expected expr as arguments for variant/struct tuple constructor"),
        }
    } else {
        // Just eval all the expressions (if any). Since expressions in Rust can have arbitrary
        // contents, there could be side-effects we need from them.
        match args {
            callee::ArgExprs(exprs) => {
                for expr in exprs {
                    bcx = expr::trans_into(bcx, expr, expr::Ignore);
                }
            }
            _ => (),
        }
    }

    // If the caller doesn't care about the result
    // drop the temporary we made
    let bcx = match dest {
        expr::SaveIn(_) => bcx,
        expr::Ignore => {
            let bcx = glue::drop_ty(bcx, llresult, result_ty, debug_loc);
            if !type_is_zero_size(ccx, result_ty) {
                call_lifetime_end(bcx, llresult);
            }
            bcx
        }
    };

    Result::new(bcx, llresult)
}

pub fn trans_tuple_struct<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    ctor_id: ast::NodeId,
                                    param_substs: &'tcx Substs<'tcx>,
                                    llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_tuple_struct");

    trans_enum_variant_or_tuple_like_struct(ccx, ctor_id, 0, param_substs, llfndecl);
}

fn trans_enum_variant_or_tuple_like_struct<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                     ctor_id: ast::NodeId,
                                                     disr: ty::Disr,
                                                     param_substs: &'tcx Substs<'tcx>,
                                                     llfndecl: ValueRef) {
    let ctor_ty = ccx.tcx().node_id_to_type(ctor_id);
    let ctor_ty = monomorphize::apply_param_substs(ccx.tcx(), param_substs, &ctor_ty);

    let sig = ccx.tcx().erase_late_bound_regions(&ctor_ty.fn_sig());
    let sig = infer::normalize_associated_type(ccx.tcx(), &sig);
    let arg_tys = sig.inputs;
    let result_ty = sig.output;

    let (arena, fcx): (TypedArena<_>, FunctionContext);
    arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      llfndecl,
                      ctor_id,
                      false,
                      result_ty,
                      param_substs,
                      None,
                      &arena);
    let bcx = init_function(&fcx, false, result_ty);

    assert!(!fcx.needs_ret_allocas);

    if !type_is_zero_size(fcx.ccx, result_ty.unwrap()) {
        let dest = fcx.get_ret_slot(bcx, result_ty, "eret_slot");
        let repr = adt::represent_type(ccx, result_ty.unwrap());
        let mut llarg_idx = fcx.arg_offset() as c_uint;
        for (i, arg_ty) in arg_tys.into_iter().enumerate() {
            let lldestptr = adt::trans_field_ptr(bcx, &*repr, dest, disr, i);
            if common::type_is_fat_ptr(bcx.tcx(), arg_ty) {
                Store(bcx,
                      get_param(fcx.llfn, llarg_idx),
                      expr::get_dataptr(bcx, lldestptr));
                Store(bcx,
                      get_param(fcx.llfn, llarg_idx + 1),
                      expr::get_meta(bcx, lldestptr));
                llarg_idx += 2;
            } else {
                let arg = get_param(fcx.llfn, llarg_idx);
                llarg_idx += 1;

                if arg_is_indirect(ccx, arg_ty) {
                    memcpy_ty(bcx, lldestptr, arg, arg_ty);
                } else {
                    store_ty(bcx, arg, lldestptr, arg_ty);
                }
            }
        }
        adt::trans_set_discr(bcx, &*repr, dest, disr);
    }

    finish_fn(&fcx, bcx, result_ty, DebugLoc::None);
}

fn enum_variant_size_lint(ccx: &CrateContext, enum_def: &hir::EnumDef, sp: Span, id: ast::NodeId) {
    let mut sizes = Vec::new(); // does no allocation if no pushes, thankfully

    let print_info = ccx.sess().print_enum_sizes();

    let levels = ccx.tcx().node_lint_levels.borrow();
    let lint_id = lint::LintId::of(lint::builtin::VARIANT_SIZE_DIFFERENCES);
    let lvlsrc = levels.get(&(id, lint_id));
    let is_allow = lvlsrc.map_or(true, |&(lvl, _)| lvl == lint::Allow);

    if is_allow && !print_info {
        // we're not interested in anything here
        return;
    }

    let ty = ccx.tcx().node_id_to_type(id);
    let avar = adt::represent_type(ccx, ty);
    match *avar {
        adt::General(_, ref variants, _) => {
            for var in variants {
                let mut size = 0;
                for field in var.fields.iter().skip(1) {
                    // skip the discriminant
                    size += llsize_of_real(ccx, sizing_type_of(ccx, *field));
                }
                sizes.push(size);
            }
        },
        _ => { /* its size is either constant or unimportant */ }
    }

    let (largest, slargest, largest_index) = sizes.iter().enumerate().fold((0, 0, 0),
        |(l, s, li), (idx, &size)|
            if size > l {
                (size, l, idx)
            } else if size > s {
                (l, size, li)
            } else {
                (l, s, li)
            }
    );

    if print_info {
        let llty = type_of::sizing_type_of(ccx, ty);

        let sess = &ccx.tcx().sess;
        sess.span_note(sp, &*format!("total size: {} bytes", llsize_of_real(ccx, llty)));
        match *avar {
            adt::General(..) => {
                for (i, var) in enum_def.variants.iter().enumerate() {
                    ccx.tcx()
                       .sess
                       .span_note(var.span, &*format!("variant data: {} bytes", sizes[i]));
                }
            }
            _ => {}
        }
    }

    // we only warn if the largest variant is at least thrice as large as
    // the second-largest.
    if !is_allow && largest > slargest * 3 && slargest > 0 {
        // Use lint::raw_emit_lint rather than sess.add_lint because the lint-printing
        // pass for the latter already ran.
        lint::raw_emit_lint(&ccx.tcx().sess,
                            lint::builtin::VARIANT_SIZE_DIFFERENCES,
                            *lvlsrc.unwrap(),
                            Some(sp),
                            &format!("enum variant is more than three times larger ({} bytes) \
                                      than the next largest (ignoring padding)",
                                     largest));

        ccx.sess().span_note(enum_def.variants[largest_index].span,
                             "this variant is the largest");
    }
}

pub fn llvm_linkage_by_name(name: &str) -> Option<Linkage> {
    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but whitelist them anyway and trust that
    // the user knows what s/he's doing. Who knows, unanticipated use cases
    // may pop up in the future.
    //
    // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
    // and don't have to be, LLVM treats them as no-ops.
    match name {
        "appending" => Some(llvm::AppendingLinkage),
        "available_externally" => Some(llvm::AvailableExternallyLinkage),
        "common" => Some(llvm::CommonLinkage),
        "extern_weak" => Some(llvm::ExternalWeakLinkage),
        "external" => Some(llvm::ExternalLinkage),
        "internal" => Some(llvm::InternalLinkage),
        "linkonce" => Some(llvm::LinkOnceAnyLinkage),
        "linkonce_odr" => Some(llvm::LinkOnceODRLinkage),
        "private" => Some(llvm::PrivateLinkage),
        "weak" => Some(llvm::WeakAnyLinkage),
        "weak_odr" => Some(llvm::WeakODRLinkage),
        _ => None,
    }
}


/// Enum describing the origin of an LLVM `Value`, for linkage purposes.
#[derive(Copy, Clone)]
pub enum ValueOrigin {
    /// The LLVM `Value` is in this context because the corresponding item was
    /// assigned to the current compilation unit.
    OriginalTranslation,
    /// The `Value`'s corresponding item was assigned to some other compilation
    /// unit, but the `Value` was translated in this context anyway because the
    /// item is marked `#[inline]`.
    InlinedCopy,
}

/// Set the appropriate linkage for an LLVM `ValueRef` (function or global).
/// If the `llval` is the direct translation of a specific Rust item, `id`
/// should be set to the `NodeId` of that item.  (This mapping should be
/// 1-to-1, so monomorphizations and drop/visit glue should have `id` set to
/// `None`.)  `llval_origin` indicates whether `llval` is the translation of an
/// item assigned to `ccx`'s compilation unit or an inlined copy of an item
/// assigned to a different compilation unit.
pub fn update_linkage(ccx: &CrateContext,
                      llval: ValueRef,
                      id: Option<ast::NodeId>,
                      llval_origin: ValueOrigin) {
    match llval_origin {
        InlinedCopy => {
            // `llval` is a translation of an item defined in a separate
            // compilation unit.  This only makes sense if there are at least
            // two compilation units.
            assert!(ccx.sess().opts.cg.codegen_units > 1);
            // `llval` is a copy of something defined elsewhere, so use
            // `AvailableExternallyLinkage` to avoid duplicating code in the
            // output.
            llvm::SetLinkage(llval, llvm::AvailableExternallyLinkage);
            return;
        },
        OriginalTranslation => {},
    }

    if let Some(id) = id {
        let item = ccx.tcx().map.get(id);
        if let hir_map::NodeItem(i) = item {
            if let Some(name) = attr::first_attr_value_str_by_name(&i.attrs, "linkage") {
                if let Some(linkage) = llvm_linkage_by_name(&name) {
                    llvm::SetLinkage(llval, linkage);
                } else {
                    ccx.sess().span_fatal(i.span, "invalid linkage specified");
                }
                return;
            }
        }
    }

    match id {
        Some(id) if ccx.reachable().contains(&id) => {
            llvm::SetLinkage(llval, llvm::ExternalLinkage);
        },
        _ => {
            // `id` does not refer to an item in `ccx.reachable`.
            if ccx.sess().opts.cg.codegen_units > 1 {
                llvm::SetLinkage(llval, llvm::ExternalLinkage);
            } else {
                llvm::SetLinkage(llval, llvm::InternalLinkage);
            }
        },
    }
}

fn set_global_section(ccx: &CrateContext, llval: ValueRef, i: &hir::Item) {
    match attr::first_attr_value_str_by_name(&i.attrs, "link_section") {
        Some(sect) => {
            if contains_null(&sect) {
                ccx.sess().fatal(&format!("Illegal null byte in link_section value: `{}`", &sect));
            }
            unsafe {
                let buf = CString::new(sect.as_bytes()).unwrap();
                llvm::LLVMSetSection(llval, buf.as_ptr());
            }
        },
        None => ()
    }
}

pub fn trans_item(ccx: &CrateContext, item: &hir::Item) {
    let _icx = push_ctxt("trans_item");

    let from_external = ccx.external_srcs().borrow().contains_key(&item.id);

    match item.node {
        hir::ItemFn(ref decl, _, _, abi, ref generics, ref body) => {
            if !generics.is_type_parameterized() {
                let trans_everywhere = attr::requests_inline(&item.attrs);
                // Ignore `trans_everywhere` for cross-crate inlined items
                // (`from_external`).  `trans_item` will be called once for each
                // compilation unit that references the item, so it will still get
                // translated everywhere it's needed.
                for (ref ccx, is_origin) in ccx.maybe_iter(!from_external && trans_everywhere) {
                    let llfn = get_item_val(ccx, item.id);
                    let empty_substs = ccx.tcx().mk_substs(Substs::trans_empty());
                    if abi != Rust {
                        foreign::trans_rust_fn_with_foreign_abi(ccx,
                                                                &**decl,
                                                                &**body,
                                                                &item.attrs,
                                                                llfn,
                                                                empty_substs,
                                                                item.id,
                                                                None);
                    } else {
                        trans_fn(ccx,
                                 &**decl,
                                 &**body,
                                 llfn,
                                 empty_substs,
                                 item.id,
                                 &item.attrs);
                    }
                    set_global_section(ccx, llfn, item);
                    update_linkage(ccx,
                                   llfn,
                                   Some(item.id),
                                   if is_origin {
                                       OriginalTranslation
                                   } else {
                                       InlinedCopy
                                   });

                    if is_entry_fn(ccx.sess(), item.id) {
                        create_entry_wrapper(ccx, item.span, llfn);
                        // check for the #[rustc_error] annotation, which forces an
                        // error in trans. This is used to write compile-fail tests
                        // that actually test that compilation succeeds without
                        // reporting an error.
                        let item_def_id = ccx.tcx().map.local_def_id(item.id);
                        if ccx.tcx().has_attr(item_def_id, "rustc_error") {
                            ccx.tcx().sess.span_fatal(item.span, "compilation successful");
                        }
                    }
                }
            }
        }
        hir::ItemImpl(_, _, ref generics, _, _, ref impl_items) => {
            meth::trans_impl(ccx, item.name, &impl_items[..], generics, item.id);
        }
        hir::ItemMod(_) => {
            // modules have no equivalent at runtime, they just affect
            // the mangled names of things contained within
        }
        hir::ItemEnum(ref enum_definition, ref gens) => {
            if gens.ty_params.is_empty() {
                // sizes only make sense for non-generic types

                enum_variant_size_lint(ccx, enum_definition, item.span, item.id);
            }
        }
        hir::ItemConst(..) => {}
        hir::ItemStatic(_, m, ref expr) => {
            let g = match consts::trans_static(ccx, m, expr, item.id, &item.attrs) {
                Ok(g) => g,
                Err(err) => ccx.tcx().sess.span_fatal(expr.span, &err.description()),
            };
            set_global_section(ccx, g, item);
            update_linkage(ccx, g, Some(item.id), OriginalTranslation);
        }
        hir::ItemForeignMod(ref foreign_mod) => {
            foreign::trans_foreign_mod(ccx, foreign_mod);
        }
        hir::ItemTrait(..) => {}
        _ => {
            // fall through
        }
    }
}

// only use this for foreign function ABIs and glue, use `register_fn` for Rust functions
pub fn register_fn_llvmty(ccx: &CrateContext,
                          sp: Span,
                          sym: String,
                          node_id: ast::NodeId,
                          cc: llvm::CallConv,
                          llfty: Type)
                          -> ValueRef {
    debug!("register_fn_llvmty id={} sym={}", node_id, sym);

    let llfn = declare::define_fn(ccx, &sym[..], cc, llfty,
                                   ty::FnConverging(ccx.tcx().mk_nil())).unwrap_or_else(||{
        ccx.sess().span_fatal(sp, &format!("symbol `{}` is already defined", sym));
    });
    finish_register_fn(ccx, sym, node_id);
    llfn
}

fn finish_register_fn(ccx: &CrateContext, sym: String, node_id: ast::NodeId) {
    ccx.item_symbols().borrow_mut().insert(node_id, sym);
}

fn register_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                         sp: Span,
                         sym: String,
                         node_id: ast::NodeId,
                         node_type: Ty<'tcx>)
                         -> ValueRef {
    if let ty::TyBareFn(_, ref f) = node_type.sty {
        if f.abi != Rust && f.abi != RustCall {
            ccx.sess().span_bug(sp,
                                &format!("only the `{}` or `{}` calling conventions are valid \
                                          for this function; `{}` was specified",
                                         Rust.name(),
                                         RustCall.name(),
                                         f.abi.name()));
        }
    } else {
        ccx.sess().span_bug(sp, "expected bare rust function")
    }

    let llfn = declare::define_rust_fn(ccx, &sym[..], node_type).unwrap_or_else(|| {
        ccx.sess().span_fatal(sp, &format!("symbol `{}` is already defined", sym));
    });
    finish_register_fn(ccx, sym, node_id);
    llfn
}

pub fn is_entry_fn(sess: &Session, node_id: ast::NodeId) -> bool {
    match *sess.entry_fn.borrow() {
        Some((entry_id, _)) => node_id == entry_id,
        None => false,
    }
}

/// Create the `main` function which will initialise the rust runtime and call users’ main
/// function.
pub fn create_entry_wrapper(ccx: &CrateContext, sp: Span, main_llfn: ValueRef) {
    let et = ccx.sess().entry_type.get().unwrap();
    match et {
        config::EntryMain => {
            create_entry_fn(ccx, sp, main_llfn, true);
        }
        config::EntryStart => create_entry_fn(ccx, sp, main_llfn, false),
        config::EntryNone => {}    // Do nothing.
    }

    fn create_entry_fn(ccx: &CrateContext,
                       sp: Span,
                       rust_main: ValueRef,
                       use_start_lang_item: bool) {
        let llfty = Type::func(&[ccx.int_type(), Type::i8p(ccx).ptr_to()], &ccx.int_type());

        let llfn = declare::define_cfn(ccx, "main", llfty, ccx.tcx().mk_nil()).unwrap_or_else(|| {
            ccx.sess().span_err(sp, "entry symbol `main` defined multiple times");
            // FIXME: We should be smart and show a better diagnostic here.
            ccx.sess().help("did you use #[no_mangle] on `fn main`? Use #[start] instead");
            ccx.sess().abort_if_errors();
            panic!();
        });

        let llbb = unsafe {
            llvm::LLVMAppendBasicBlockInContext(ccx.llcx(), llfn, "top\0".as_ptr() as *const _)
        };
        let bld = ccx.raw_builder();
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(bld, llbb);

            debuginfo::gdb::insert_reference_to_gdb_debug_scripts_section_global(ccx);

            let (start_fn, args) = if use_start_lang_item {
                let start_def_id = match ccx.tcx().lang_items.require(StartFnLangItem) {
                    Ok(id) => id,
                    Err(s) => {
                        ccx.sess().fatal(&s[..]);
                    }
                };
                let start_fn = if let Some(start_node_id) = ccx.tcx()
                                                               .map
                                                               .as_local_node_id(start_def_id) {
                    get_item_val(ccx, start_node_id)
                } else {
                    let start_fn_type = ccx.tcx().lookup_item_type(start_def_id).ty;
                    trans_external_path(ccx, start_def_id, start_fn_type)
                };
                let args = {
                    let opaque_rust_main =
                        llvm::LLVMBuildPointerCast(bld,
                                                   rust_main,
                                                   Type::i8p(ccx).to_ref(),
                                                   "rust_main\0".as_ptr() as *const _);

                    vec![opaque_rust_main, get_param(llfn, 0), get_param(llfn, 1)]
                };
                (start_fn, args)
            } else {
                debug!("using user-defined start fn");
                let args = vec![get_param(llfn, 0 as c_uint), get_param(llfn, 1 as c_uint)];

                (rust_main, args)
            };

            let result = llvm::LLVMBuildCall(bld,
                                             start_fn,
                                             args.as_ptr(),
                                             args.len() as c_uint,
                                             noname());

            llvm::LLVMBuildRet(bld, result);
        }
    }
}

fn exported_name<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                           id: ast::NodeId,
                           ty: Ty<'tcx>,
                           attrs: &[ast::Attribute])
                           -> String {
    match ccx.external_srcs().borrow().get(&id) {
        Some(&did) => {
            let sym = ccx.sess().cstore.item_symbol(did);
            debug!("found item {} in other crate...", sym);
            return sym;
        }
        None => {}
    }

    match attr::find_export_name_attr(ccx.sess().diagnostic(), attrs) {
        // Use provided name
        Some(name) => name.to_string(),
        _ => {
            let path = ccx.tcx().map.def_path_from_id(id);
            if attr::contains_name(attrs, "no_mangle") {
                // Don't mangle
                path.last().unwrap().data.to_string()
            } else {
                match weak_lang_items::link_name(attrs) {
                    Some(name) => name.to_string(),
                    None => {
                        // Usual name mangling
                        mangle_exported_name(ccx, path, ty, id)
                    }
                }
            }
        }
    }
}

fn contains_null(s: &str) -> bool {
    s.bytes().any(|b| b == 0)
}

pub fn get_item_val(ccx: &CrateContext, id: ast::NodeId) -> ValueRef {
    debug!("get_item_val(id=`{}`)", id);

    match ccx.item_vals().borrow().get(&id).cloned() {
        Some(v) => return v,
        None => {}
    }

    let item = ccx.tcx().map.get(id);
    debug!("get_item_val: id={} item={:?}", id, item);
    let val = match item {
        hir_map::NodeItem(i) => {
            let ty = ccx.tcx().node_id_to_type(i.id);
            let sym = || exported_name(ccx, id, ty, &i.attrs);

            let v = match i.node {
                hir::ItemStatic(..) => {
                    // If this static came from an external crate, then
                    // we need to get the symbol from metadata instead of
                    // using the current crate's name/version
                    // information in the hash of the symbol
                    let sym = sym();
                    debug!("making {}", sym);

                    // Create the global before evaluating the initializer;
                    // this is necessary to allow recursive statics.
                    let llty = type_of(ccx, ty);
                    let g = declare::define_global(ccx, &sym[..], llty).unwrap_or_else(|| {
                        ccx.sess()
                           .span_fatal(i.span, &format!("symbol `{}` is already defined", sym))
                    });

                    ccx.item_symbols().borrow_mut().insert(i.id, sym);
                    g
                }

                hir::ItemFn(_, _, _, abi, _, _) => {
                    let sym = sym();
                    let llfn = if abi == Rust {
                        register_fn(ccx, i.span, sym, i.id, ty)
                    } else {
                        foreign::register_rust_fn_with_foreign_abi(ccx, i.span, sym, i.id)
                    };
                    attributes::from_fn_attrs(ccx, &i.attrs, llfn);
                    llfn
                }

                _ => ccx.sess().bug("get_item_val: weird result in table"),
            };

            v
        }

        hir_map::NodeTraitItem(trait_item) => {
            debug!("get_item_val(): processing a NodeTraitItem");
            match trait_item.node {
                hir::MethodTraitItem(_, Some(_)) => {
                    register_method(ccx, id, &trait_item.attrs, trait_item.span)
                }
                _ => {
                    ccx.sess().span_bug(trait_item.span,
                                        "unexpected variant: trait item other than a provided \
                                         method in get_item_val()");
                }
            }
        }

        hir_map::NodeImplItem(impl_item) => {
            match impl_item.node {
                hir::ImplItemKind::Method(..) => {
                    register_method(ccx, id, &impl_item.attrs, impl_item.span)
                }
                _ => {
                    ccx.sess().span_bug(impl_item.span,
                                        "unexpected variant: non-method impl item in \
                                         get_item_val()");
                }
            }
        }

        hir_map::NodeForeignItem(ni) => {
            match ni.node {
                hir::ForeignItemFn(..) => {
                    let abi = ccx.tcx().map.get_foreign_abi(id);
                    let ty = ccx.tcx().node_id_to_type(ni.id);
                    let name = foreign::link_name(&*ni);
                    foreign::register_foreign_item_fn(ccx, abi, ty, &name, &ni.attrs)
                }
                hir::ForeignItemStatic(..) => {
                    foreign::register_static(ccx, &*ni)
                }
            }
        }

        hir_map::NodeVariant(ref v) => {
            let llfn;
            let fields = if v.node.data.is_struct() {
                ccx.sess().bug("struct variant kind unexpected in get_item_val")
            } else {
                v.node.data.fields()
            };
            assert!(!fields.is_empty());
            let ty = ccx.tcx().node_id_to_type(id);
            let parent = ccx.tcx().map.get_parent(id);
            let enm = ccx.tcx().map.expect_item(parent);
            let sym = exported_name(ccx, id, ty, &enm.attrs);

            llfn = match enm.node {
                hir::ItemEnum(_, _) => {
                    register_fn(ccx, (*v).span, sym, id, ty)
                }
                _ => ccx.sess().bug("NodeVariant, shouldn't happen"),
            };
            attributes::inline(llfn, attributes::InlineAttr::Hint);
            llfn
        }

        hir_map::NodeStructCtor(struct_def) => {
            // Only register the constructor if this is a tuple-like struct.
            let ctor_id = if struct_def.is_struct() {
                ccx.sess().bug("attempt to register a constructor of a non-tuple-like struct")
            } else {
                struct_def.id()
            };
            let parent = ccx.tcx().map.get_parent(id);
            let struct_item = ccx.tcx().map.expect_item(parent);
            let ty = ccx.tcx().node_id_to_type(ctor_id);
            let sym = exported_name(ccx, id, ty, &struct_item.attrs);
            let llfn = register_fn(ccx, struct_item.span, sym, ctor_id, ty);
            attributes::inline(llfn, attributes::InlineAttr::Hint);
            llfn
        }

        ref variant => {
            ccx.sess().bug(&format!("get_item_val(): unexpected variant: {:?}", variant))
        }
    };

    // All LLVM globals and functions are initially created as external-linkage
    // declarations.  If `trans_item`/`trans_fn` later turns the declaration
    // into a definition, it adjusts the linkage then (using `update_linkage`).
    //
    // The exception is foreign items, which have their linkage set inside the
    // call to `foreign::register_*` above.  We don't touch the linkage after
    // that (`foreign::trans_foreign_mod` doesn't adjust the linkage like the
    // other item translation functions do).

    ccx.item_vals().borrow_mut().insert(id, val);
    val
}

fn register_method(ccx: &CrateContext,
                   id: ast::NodeId,
                   attrs: &[ast::Attribute],
                   span: Span)
                   -> ValueRef {
    let mty = ccx.tcx().node_id_to_type(id);

    let sym = exported_name(ccx, id, mty, &attrs);

    if let ty::TyBareFn(_, ref f) = mty.sty {
        let llfn = if f.abi == Rust || f.abi == RustCall {
            register_fn(ccx, span, sym, id, mty)
        } else {
            foreign::register_rust_fn_with_foreign_abi(ccx, span, sym, id)
        };
        attributes::from_fn_attrs(ccx, &attrs, llfn);
        return llfn;
    } else {
        ccx.sess().span_bug(span, "expected bare rust function");
    }
}

pub fn write_metadata(cx: &SharedCrateContext, krate: &hir::Crate, reachable: &NodeSet) -> Vec<u8> {
    use flate;

    let any_library = cx.sess()
                        .crate_types
                        .borrow()
                        .iter()
                        .any(|ty| *ty != config::CrateTypeExecutable);
    if !any_library {
        return Vec::new();
    }

    let cstore = &cx.tcx().sess.cstore;
    let metadata = cstore.encode_metadata(
        cx.tcx(), cx.export_map(), cx.item_symbols(), cx.link_meta(), reachable,
        krate);
    let mut compressed = cstore.metadata_encoding_version().to_vec();
    compressed.extend_from_slice(&flate::deflate_bytes(&metadata));

    let llmeta = C_bytes_in_context(cx.metadata_llcx(), &compressed[..]);
    let llconst = C_struct_in_context(cx.metadata_llcx(), &[llmeta], false);
    let name = format!("rust_metadata_{}_{}",
                       cx.link_meta().crate_name,
                       cx.link_meta().crate_hash);
    let buf = CString::new(name).unwrap();
    let llglobal = unsafe {
        llvm::LLVMAddGlobal(cx.metadata_llmod(), val_ty(llconst).to_ref(), buf.as_ptr())
    };
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        let name =
            cx.tcx().sess.cstore.metadata_section_name(&cx.sess().target.target);
        let name = CString::new(name).unwrap();
        llvm::LLVMSetSection(llglobal, name.as_ptr())
    }
    return metadata;
}

/// Find any symbols that are defined in one compilation unit, but not declared
/// in any other compilation unit.  Give these symbols internal linkage.
fn internalize_symbols(cx: &SharedCrateContext, reachable: &HashSet<&str>) {
    unsafe {
        let mut declared = HashSet::new();

        // Collect all external declarations in all compilation units.
        for ccx in cx.iter() {
            for val in iter_globals(ccx.llmod()).chain(iter_functions(ccx.llmod())) {
                let linkage = llvm::LLVMGetLinkage(val);
                // We only care about external declarations (not definitions)
                // and available_externally definitions.
                if !(linkage == llvm::ExternalLinkage as c_uint &&
                     llvm::LLVMIsDeclaration(val) != 0) &&
                   !(linkage == llvm::AvailableExternallyLinkage as c_uint) {
                    continue;
                }

                let name = CStr::from_ptr(llvm::LLVMGetValueName(val))
                               .to_bytes()
                               .to_vec();
                declared.insert(name);
            }
        }

        // Examine each external definition.  If the definition is not used in
        // any other compilation unit, and is not reachable from other crates,
        // then give it internal linkage.
        for ccx in cx.iter() {
            for val in iter_globals(ccx.llmod()).chain(iter_functions(ccx.llmod())) {
                // We only care about external definitions.
                if !(llvm::LLVMGetLinkage(val) == llvm::ExternalLinkage as c_uint &&
                     llvm::LLVMIsDeclaration(val) == 0) {
                    continue;
                }

                let name = CStr::from_ptr(llvm::LLVMGetValueName(val))
                               .to_bytes()
                               .to_vec();
                if !declared.contains(&name) &&
                   !reachable.contains(str::from_utf8(&name).unwrap()) {
                    llvm::SetLinkage(val, llvm::InternalLinkage);
                    llvm::SetDLLStorageClass(val, llvm::DefaultStorageClass);
                }
            }
        }
    }
}

// Create a `__imp_<symbol> = &symbol` global for every public static `symbol`.
// This is required to satisfy `dllimport` references to static data in .rlibs
// when using MSVC linker.  We do this only for data, as linker can fix up
// code references on its own.
// See #26591, #27438
fn create_imps(cx: &SharedCrateContext) {
    // The x86 ABI seems to require that leading underscores are added to symbol
    // names, so we need an extra underscore on 32-bit. There's also a leading
    // '\x01' here which disables LLVM's symbol mangling (e.g. no extra
    // underscores added in front).
    let prefix = if cx.sess().target.target.target_pointer_width == "32" {
        "\x01__imp__"
    } else {
        "\x01__imp_"
    };
    unsafe {
        for ccx in cx.iter() {
            let exported: Vec<_> = iter_globals(ccx.llmod())
                                       .filter(|&val| {
                                           llvm::LLVMGetLinkage(val) ==
                                           llvm::ExternalLinkage as c_uint &&
                                           llvm::LLVMIsDeclaration(val) == 0
                                       })
                                       .collect();

            let i8p_ty = Type::i8p(&ccx);
            for val in exported {
                let name = CStr::from_ptr(llvm::LLVMGetValueName(val));
                let mut imp_name = prefix.as_bytes().to_vec();
                imp_name.extend(name.to_bytes());
                let imp_name = CString::new(imp_name).unwrap();
                let imp = llvm::LLVMAddGlobal(ccx.llmod(),
                                              i8p_ty.to_ref(),
                                              imp_name.as_ptr() as *const _);
                let init = llvm::LLVMConstBitCast(val, i8p_ty.to_ref());
                llvm::LLVMSetInitializer(imp, init);
                llvm::SetLinkage(imp, llvm::ExternalLinkage);
            }
        }
    }
}

struct ValueIter {
    cur: ValueRef,
    step: unsafe extern "C" fn(ValueRef) -> ValueRef,
}

impl Iterator for ValueIter {
    type Item = ValueRef;

    fn next(&mut self) -> Option<ValueRef> {
        let old = self.cur;
        if !old.is_null() {
            self.cur = unsafe { (self.step)(old) };
            Some(old)
        } else {
            None
        }
    }
}

fn iter_globals(llmod: llvm::ModuleRef) -> ValueIter {
    unsafe {
        ValueIter {
            cur: llvm::LLVMGetFirstGlobal(llmod),
            step: llvm::LLVMGetNextGlobal,
        }
    }
}

fn iter_functions(llmod: llvm::ModuleRef) -> ValueIter {
    unsafe {
        ValueIter {
            cur: llvm::LLVMGetFirstFunction(llmod),
            step: llvm::LLVMGetNextFunction,
        }
    }
}

/// The context provided lists a set of reachable ids as calculated by
/// middle::reachable, but this contains far more ids and symbols than we're
/// actually exposing from the object file. This function will filter the set in
/// the context to the set of ids which correspond to symbols that are exposed
/// from the object file being generated.
///
/// This list is later used by linkers to determine the set of symbols needed to
/// be exposed from a dynamic library and it's also encoded into the metadata.
pub fn filter_reachable_ids(ccx: &SharedCrateContext) -> NodeSet {
    ccx.reachable().iter().map(|x| *x).filter(|id| {
        // First, only worry about nodes which have a symbol name
        ccx.item_symbols().borrow().contains_key(id)
    }).filter(|&id| {
        // Next, we want to ignore some FFI functions that are not exposed from
        // this crate. Reachable FFI functions can be lumped into two
        // categories:
        //
        // 1. Those that are included statically via a static library
        // 2. Those included otherwise (e.g. dynamically or via a framework)
        //
        // Although our LLVM module is not literally emitting code for the
        // statically included symbols, it's an export of our library which
        // needs to be passed on to the linker and encoded in the metadata.
        //
        // As a result, if this id is an FFI item (foreign item) then we only
        // let it through if it's included statically.
        match ccx.tcx().map.get(id) {
            hir_map::NodeForeignItem(..) => {
                ccx.sess().cstore.is_statically_included_foreign_item(id)
            }
            _ => true,
        }
    }).collect()
}

pub fn trans_crate<'tcx>(tcx: &ty::ctxt<'tcx>,
                         mir_map: &MirMap<'tcx>,
                         analysis: ty::CrateAnalysis)
                         -> CrateTranslation {
    let ty::CrateAnalysis { export_map, reachable, name, .. } = analysis;
    let krate = tcx.map.krate();

    let check_overflow = if let Some(v) = tcx.sess.opts.debugging_opts.force_overflow_checks {
        v
    } else {
        tcx.sess.opts.debug_assertions
    };

    let check_dropflag = if let Some(v) = tcx.sess.opts.debugging_opts.force_dropflag_checks {
        v
    } else {
        tcx.sess.opts.debug_assertions
    };

    // Before we touch LLVM, make sure that multithreading is enabled.
    unsafe {
        use std::sync::Once;
        static INIT: Once = Once::new();
        static mut POISONED: bool = false;
        INIT.call_once(|| {
            if llvm::LLVMStartMultithreaded() != 1 {
                // use an extra bool to make sure that all future usage of LLVM
                // cannot proceed despite the Once not running more than once.
                POISONED = true;
            }

            ::back::write::configure_llvm(&tcx.sess);
        });

        if POISONED {
            tcx.sess.bug("couldn't enable multi-threaded LLVM");
        }
    }

    let link_meta = link::build_link_meta(&tcx.sess, krate, name);

    let codegen_units = tcx.sess.opts.cg.codegen_units;
    let shared_ccx = SharedCrateContext::new(&link_meta.crate_name,
                                             codegen_units,
                                             tcx,
                                             &mir_map,
                                             export_map,
                                             Sha256::new(),
                                             link_meta.clone(),
                                             reachable,
                                             check_overflow,
                                             check_dropflag);

    {
        let ccx = shared_ccx.get_ccx(0);

        // First, verify intrinsics.
        intrinsic::check_intrinsics(&ccx);

        // Next, translate all items. See `TransModVisitor` for
        // details on why we walk in this particular way.
        {
            let _icx = push_ctxt("text");
            intravisit::walk_mod(&mut TransItemsWithinModVisitor { ccx: &ccx }, &krate.module);
            krate.visit_all_items(&mut TransModVisitor { ccx: &ccx });
        }
    }

    for ccx in shared_ccx.iter() {
        if ccx.sess().opts.debuginfo != NoDebugInfo {
            debuginfo::finalize(&ccx);
        }
        for &(old_g, new_g) in ccx.statics_to_rauw().borrow().iter() {
            unsafe {
                let bitcast = llvm::LLVMConstPointerCast(new_g, llvm::LLVMTypeOf(old_g));
                llvm::LLVMReplaceAllUsesWith(old_g, bitcast);
                llvm::LLVMDeleteGlobal(old_g);
            }
        }
    }

    let reachable_symbol_ids = filter_reachable_ids(&shared_ccx);

    // Translate the metadata.
    let metadata = write_metadata(&shared_ccx, krate, &reachable_symbol_ids);

    if shared_ccx.sess().trans_stats() {
        let stats = shared_ccx.stats();
        println!("--- trans stats ---");
        println!("n_glues_created: {}", stats.n_glues_created.get());
        println!("n_null_glues: {}", stats.n_null_glues.get());
        println!("n_real_glues: {}", stats.n_real_glues.get());

        println!("n_fns: {}", stats.n_fns.get());
        println!("n_monos: {}", stats.n_monos.get());
        println!("n_inlines: {}", stats.n_inlines.get());
        println!("n_closures: {}", stats.n_closures.get());
        println!("fn stats:");
        stats.fn_stats.borrow_mut().sort_by(|&(_, insns_a), &(_, insns_b)| {
            insns_b.cmp(&insns_a)
        });
        for tuple in stats.fn_stats.borrow().iter() {
            match *tuple {
                (ref name, insns) => {
                    println!("{} insns, {}", insns, *name);
                }
            }
        }
    }
    if shared_ccx.sess().count_llvm_insns() {
        for (k, v) in shared_ccx.stats().llvm_insns.borrow().iter() {
            println!("{:7} {}", *v, *k);
        }
    }

    let modules = shared_ccx.iter()
        .map(|ccx| ModuleTranslation { llcx: ccx.llcx(), llmod: ccx.llmod() })
        .collect();

    let sess = shared_ccx.sess();
    let mut reachable_symbols = reachable_symbol_ids.iter().map(|id| {
        shared_ccx.item_symbols().borrow()[id].to_string()
    }).collect::<Vec<_>>();
    if sess.entry_fn.borrow().is_some() {
        reachable_symbols.push("main".to_string());
    }

    // For the purposes of LTO, we add to the reachable set all of the upstream
    // reachable extern fns. These functions are all part of the public ABI of
    // the final product, so LTO needs to preserve them.
    if sess.lto() {
        for cnum in sess.cstore.crates() {
            let syms = sess.cstore.reachable_ids(cnum);
            reachable_symbols.extend(syms.into_iter().filter(|did| {
                sess.cstore.is_extern_fn(shared_ccx.tcx(), *did) ||
                sess.cstore.is_static(*did)
            }).map(|did| {
                sess.cstore.item_symbol(did)
            }));
        }
    }

    if codegen_units > 1 {
        internalize_symbols(&shared_ccx,
                            &reachable_symbols.iter().map(|x| &x[..]).collect());
    }

    if sess.target.target.options.is_like_msvc &&
       sess.crate_types.borrow().iter().any(|ct| *ct == config::CrateTypeRlib) {
        create_imps(&shared_ccx);
    }

    let metadata_module = ModuleTranslation {
        llcx: shared_ccx.metadata_llcx(),
        llmod: shared_ccx.metadata_llmod(),
    };
    let no_builtins = attr::contains_name(&krate.attrs, "no_builtins");

    CrateTranslation {
        modules: modules,
        metadata_module: metadata_module,
        link: link_meta,
        metadata: metadata,
        reachable: reachable_symbols,
        no_builtins: no_builtins,
    }
}

/// We visit all the items in the krate and translate them.  We do
/// this in two walks. The first walk just finds module items. It then
/// walks the full contents of those module items and translates all
/// the items within. Note that this entire process is O(n). The
/// reason for this two phased walk is that each module is
/// (potentially) placed into a distinct codegen-unit. This walk also
/// ensures that the immediate contents of each module is processed
/// entirely before we proceed to find more modules, helping to ensure
/// an equitable distribution amongst codegen-units.
pub struct TransModVisitor<'a, 'tcx: 'a> {
    pub ccx: &'a CrateContext<'a, 'tcx>,
}

impl<'a, 'tcx, 'v> Visitor<'v> for TransModVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &hir::Item) {
        match i.node {
            hir::ItemMod(_) => {
                let item_ccx = self.ccx.rotate();
                intravisit::walk_item(&mut TransItemsWithinModVisitor { ccx: &item_ccx }, i);
            }
            _ => { }
        }
    }
}

/// Translates all the items within a given module. Expects owner to
/// invoke `walk_item` on a module item. Ignores nested modules.
pub struct TransItemsWithinModVisitor<'a, 'tcx: 'a> {
    pub ccx: &'a CrateContext<'a, 'tcx>,
}

impl<'a, 'tcx, 'v> Visitor<'v> for TransItemsWithinModVisitor<'a, 'tcx> {
    fn visit_nested_item(&mut self, item_id: hir::ItemId) {
        self.visit_item(self.ccx.tcx().map.expect_item(item_id.id));
    }

    fn visit_item(&mut self, i: &hir::Item) {
        match i.node {
            hir::ItemMod(..) => {
                // skip modules, they will be uncovered by the TransModVisitor
            }
            _ => {
                trans_item(self.ccx, i);
                intravisit::walk_item(self, i);
            }
        }
    }
}

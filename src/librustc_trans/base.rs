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
//! the result of the translation to LLVM -- while others, such as trans_fn
//! and trans_item, are called only for the side effect of adding a
//! particular definition to the LLVM IR output we're producing.
//!
//! Hopefully useful general knowledge about trans:
//!
//!   * There's no way to find out the Ty type of a ValueRef.  Doing so
//!     would be "trying to get the eggs out of an omelette" (credit:
//!     pcwalton).  You can, instead, find out its TypeRef by calling val_ty,
//!     but one TypeRef corresponds to many `Ty`s; for instance, tup(int, int,
//!     int) and rec(x=int, y=int, z=int) will have the same TypeRef.

use super::ModuleLlvm;
use super::ModuleSource;
use super::ModuleTranslation;
use super::ModuleKind;

use assert_module_sources;
use back::link;
use back::linker::LinkerInfo;
use back::symbol_export::{self, ExportedSymbols};
use back::write::{self, OngoingCrateTranslation};
use llvm::{ContextRef, Linkage, ModuleRef, ValueRef, Vector, get_param};
use llvm;
use metadata;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::middle::lang_items::StartFnLangItem;
use rustc::middle::cstore::{EncodedMetadata, EncodedMetadataHashes};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::dep_graph::AssertDepGraphSafe;
use rustc::middle::cstore::LinkMeta;
use rustc::hir::map as hir_map;
use rustc::util::common::{time, print_time_passes_entry};
use rustc::session::config::{self, NoDebugInfo, OutputFilenames, OutputType};
use rustc::session::Session;
use rustc_incremental::{self, IncrementalHashesMap};
use abi;
use allocator;
use mir::lvalue::LvalueRef;
use attributes;
use builder::Builder;
use callee;
use common::{C_bool, C_bytes_in_context, C_i32, C_uint};
use collector::{self, TransItemCollectionMode};
use common::{C_struct_in_context, C_u64, C_undef, C_array};
use common::CrateContext;
use common::{type_is_zero_size, val_ty};
use common;
use consts;
use context::{self, LocalCrateContext, SharedCrateContext, Stats};
use debuginfo;
use declare;
use machine;
use meth;
use mir;
use monomorphize::{self, Instance};
use partitioning::{self, PartitioningStrategy, CodegenUnit};
use symbol_names_test;
use time_graph;
use trans_item::{TransItem, DefPathBasedNames};
use type_::Type;
use type_of;
use value::Value;
use rustc::util::nodemap::{NodeSet, FxHashMap, FxHashSet};

use libc::c_uint;
use std::ffi::{CStr, CString};
use std::str;
use std::sync::Arc;
use std::time::{Instant, Duration};
use std::i32;
use syntax_pos::Span;
use syntax::attr;
use rustc::hir;
use syntax::ast;

use mir::lvalue::Alignment;

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
            self.ccx.stats().fn_stats.borrow_mut()
                .push((self.name.take().unwrap(), iend - self.istart));
            self.ccx.stats().n_fns.set(self.ccx.stats().n_fns.get() + 1);
            // Reset LLVM insn count to avoid compound costs.
            self.ccx.stats().n_llvm_insns.set(self.istart);
        }
    }
}

pub fn get_meta(bcx: &Builder, fat_ptr: ValueRef) -> ValueRef {
    bcx.struct_gep(fat_ptr, abi::FAT_PTR_EXTRA)
}

pub fn get_dataptr(bcx: &Builder, fat_ptr: ValueRef) -> ValueRef {
    bcx.struct_gep(fat_ptr, abi::FAT_PTR_ADDR)
}

pub fn bin_op_to_icmp_predicate(op: hir::BinOp_,
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
            bug!("comparison_op_to_icmp_predicate: expected comparison operator, \
                  found {:?}",
                 op)
        }
    }
}

pub fn bin_op_to_fcmp_predicate(op: hir::BinOp_) -> llvm::RealPredicate {
    match op {
        hir::BiEq => llvm::RealOEQ,
        hir::BiNe => llvm::RealUNE,
        hir::BiLt => llvm::RealOLT,
        hir::BiLe => llvm::RealOLE,
        hir::BiGt => llvm::RealOGT,
        hir::BiGe => llvm::RealOGE,
        op => {
            bug!("comparison_op_to_fcmp_predicate: expected comparison operator, \
                  found {:?}",
                 op);
        }
    }
}

pub fn compare_simd_types<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    lhs: ValueRef,
    rhs: ValueRef,
    t: Ty<'tcx>,
    ret_ty: Type,
    op: hir::BinOp_
) -> ValueRef {
    let signed = match t.sty {
        ty::TyFloat(_) => {
            let cmp = bin_op_to_fcmp_predicate(op);
            return bcx.sext(bcx.fcmp(cmp, lhs, rhs), ret_ty);
        },
        ty::TyUint(_) => false,
        ty::TyInt(_) => true,
        _ => bug!("compare_simd_types: invalid SIMD type"),
    };

    let cmp = bin_op_to_icmp_predicate(op, signed);
    // LLVM outputs an `< size x i1 >`, so we need to perform a sign extension
    // to get the correctly sized type. This will compile to a single instruction
    // once the IR is converted to assembly if the SIMD instruction is supported
    // by the target architecture.
    bcx.sext(bcx.icmp(cmp, lhs, rhs), ret_ty)
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
                                old_info: Option<ValueRef>)
                                -> ValueRef {
    let (source, target) = ccx.tcx().struct_lockstep_tails(source, target);
    match (&source.sty, &target.sty) {
        (&ty::TyArray(_, len), &ty::TySlice(_)) => C_uint(ccx, len),
        (&ty::TyDynamic(..), &ty::TyDynamic(..)) => {
            // For now, upcasts are limited to changes in marker
            // traits, and hence never actually require an actual
            // change to the vtable.
            old_info.expect("unsized_info: missing old info for trait upcast")
        }
        (_, &ty::TyDynamic(ref data, ..)) => {
            consts::ptrcast(meth::get_vtable(ccx, source, data.principal()),
                            Type::vtable_ptr(ccx))
        }
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}",
                                     source,
                                     target),
    }
}

/// Coerce `src` to `dst_ty`. `src_ty` must be a thin pointer.
pub fn unsize_thin_ptr<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    src: ValueRef,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>
) -> (ValueRef, ValueRef) {
    debug!("unsize_thin_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRef(_, ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) => {
            assert!(bcx.ccx.shared().type_is_sized(a));
            let ptr_ty = type_of::in_memory_type_of(bcx.ccx, b).ptr_to();
            (bcx.pointercast(src, ptr_ty), unsized_info(bcx.ccx, a, b, None))
        }
        (&ty::TyAdt(def_a, _), &ty::TyAdt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            let (a, b) = (src_ty.boxed_ty(), dst_ty.boxed_ty());
            assert!(bcx.ccx.shared().type_is_sized(a));
            let ptr_ty = type_of::in_memory_type_of(bcx.ccx, b).ptr_to();
            (bcx.pointercast(src, ptr_ty), unsized_info(bcx.ccx, a, b, None))
        }
        _ => bug!("unsize_thin_ptr: called on bad types"),
    }
}

/// Coerce `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty` and store the result in `dst`
pub fn coerce_unsized_into<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                                     src: &LvalueRef<'tcx>,
                                     dst: &LvalueRef<'tcx>) {
    let src_ty = src.ty.to_ty(bcx.tcx());
    let dst_ty = dst.ty.to_ty(bcx.tcx());
    let coerce_ptr = || {
        let (base, info) = if common::type_is_fat_ptr(bcx.ccx, src_ty) {
            // fat-ptr to fat-ptr unsize preserves the vtable
            // i.e. &'a fmt::Debug+Send => &'a fmt::Debug
            // So we need to pointercast the base to ensure
            // the types match up.
            let (base, info) = load_fat_ptr(bcx, src.llval, src.alignment, src_ty);
            let llcast_ty = type_of::fat_ptr_base_ty(bcx.ccx, dst_ty);
            let base = bcx.pointercast(base, llcast_ty);
            (base, info)
        } else {
            let base = load_ty(bcx, src.llval, src.alignment, src_ty);
            unsize_thin_ptr(bcx, base, src_ty, dst_ty)
        };
        store_fat_ptr(bcx, base, info, dst.llval, dst.alignment, dst_ty);
    };
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::TyRef(..), &ty::TyRef(..)) |
        (&ty::TyRef(..), &ty::TyRawPtr(..)) |
        (&ty::TyRawPtr(..), &ty::TyRawPtr(..)) => {
            coerce_ptr()
        }
        (&ty::TyAdt(def_a, _), &ty::TyAdt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            coerce_ptr()
        }

        (&ty::TyAdt(def_a, substs_a), &ty::TyAdt(def_b, substs_b)) => {
            assert_eq!(def_a, def_b);

            let src_fields = def_a.variants[0].fields.iter().map(|f| {
                monomorphize::field_ty(bcx.tcx(), substs_a, f)
            });
            let dst_fields = def_b.variants[0].fields.iter().map(|f| {
                monomorphize::field_ty(bcx.tcx(), substs_b, f)
            });

            let iter = src_fields.zip(dst_fields).enumerate();
            for (i, (src_fty, dst_fty)) in iter {
                if type_is_zero_size(bcx.ccx, dst_fty) {
                    continue;
                }

                let (src_f, src_f_align) = src.trans_field_ptr(bcx, i);
                let (dst_f, dst_f_align) = dst.trans_field_ptr(bcx, i);
                if src_fty == dst_fty {
                    memcpy_ty(bcx, dst_f, src_f, src_fty, None);
                } else {
                    coerce_unsized_into(
                        bcx,
                        &LvalueRef::new_sized_ty(src_f, src_fty, src_f_align),
                        &LvalueRef::new_sized_ty(dst_f, dst_fty, dst_f_align)
                    );
                }
            }
        }
        _ => bug!("coerce_unsized_into: invalid coercion {:?} -> {:?}",
                  src_ty,
                  dst_ty),
    }
}

pub fn cast_shift_expr_rhs(
    cx: &Builder, op: hir::BinOp_, lhs: ValueRef, rhs: ValueRef
) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs, |a, b| cx.trunc(a, b), |a, b| cx.zext(a, b))
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
    if op.is_shift() {
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

/// Returns whether this session's target will use SEH-based unwinding.
///
/// This is only true for MSVC targets, and even then the 64-bit MSVC target
/// currently uses SEH-ish unwinding with DWARF info tables to the side (same as
/// 64-bit MinGW) instead of "full SEH".
pub fn wants_msvc_seh(sess: &Session) -> bool {
    sess.target.target.options.is_like_msvc
}

pub fn call_assume<'a, 'tcx>(b: &Builder<'a, 'tcx>, val: ValueRef) {
    let assume_intrinsic = b.ccx.get_intrinsic("llvm.assume");
    b.call(assume_intrinsic, &[val], None);
}

/// Helper for loading values from memory. Does the necessary conversion if the in-memory type
/// differs from the type used for SSA values. Also handles various special cases where the type
/// gives us better information about what we are loading.
pub fn load_ty<'a, 'tcx>(b: &Builder<'a, 'tcx>, ptr: ValueRef,
                         alignment: Alignment, t: Ty<'tcx>) -> ValueRef {
    let ccx = b.ccx;
    if type_is_zero_size(ccx, t) {
        return C_undef(type_of::type_of(ccx, t));
    }

    unsafe {
        let global = llvm::LLVMIsAGlobalVariable(ptr);
        if !global.is_null() && llvm::LLVMIsGlobalConstant(global) == llvm::True {
            let val = llvm::LLVMGetInitializer(global);
            if !val.is_null() {
                if t.is_bool() {
                    return llvm::LLVMConstTrunc(val, Type::i1(ccx).to_ref());
                }
                return val;
            }
        }
    }

    if t.is_bool() {
        b.trunc(b.load_range_assert(ptr, 0, 2, llvm::False, alignment.to_align()),
                Type::i1(ccx))
    } else if t.is_char() {
        // a char is a Unicode codepoint, and so takes values from 0
        // to 0x10FFFF inclusive only.
        b.load_range_assert(ptr, 0, 0x10FFFF + 1, llvm::False, alignment.to_align())
    } else if (t.is_region_ptr() || t.is_box() || t.is_fn())
        && !common::type_is_fat_ptr(ccx, t)
    {
        b.load_nonnull(ptr, alignment.to_align())
    } else {
        b.load(ptr, alignment.to_align())
    }
}

/// Helper for storing values in memory. Does the necessary conversion if the in-memory type
/// differs from the type used for SSA values.
pub fn store_ty<'a, 'tcx>(cx: &Builder<'a, 'tcx>, v: ValueRef, dst: ValueRef,
                          dst_align: Alignment, t: Ty<'tcx>) {
    debug!("store_ty: {:?} : {:?} <- {:?}", Value(dst), t, Value(v));

    if common::type_is_fat_ptr(cx.ccx, t) {
        let lladdr = cx.extract_value(v, abi::FAT_PTR_ADDR);
        let llextra = cx.extract_value(v, abi::FAT_PTR_EXTRA);
        store_fat_ptr(cx, lladdr, llextra, dst, dst_align, t);
    } else {
        cx.store(from_immediate(cx, v), dst, dst_align.to_align());
    }
}

pub fn store_fat_ptr<'a, 'tcx>(cx: &Builder<'a, 'tcx>,
                               data: ValueRef,
                               extra: ValueRef,
                               dst: ValueRef,
                               dst_align: Alignment,
                               _ty: Ty<'tcx>) {
    // FIXME: emit metadata
    cx.store(data, get_dataptr(cx, dst), dst_align.to_align());
    cx.store(extra, get_meta(cx, dst), dst_align.to_align());
}

pub fn load_fat_ptr<'a, 'tcx>(
    b: &Builder<'a, 'tcx>, src: ValueRef, alignment: Alignment, t: Ty<'tcx>
) -> (ValueRef, ValueRef) {
    let ptr = get_dataptr(b, src);
    let ptr = if t.is_region_ptr() || t.is_box() {
        b.load_nonnull(ptr, alignment.to_align())
    } else {
        b.load(ptr, alignment.to_align())
    };

    let meta = get_meta(b, src);
    let meta_ty = val_ty(meta);
    // If the 'meta' field is a pointer, it's a vtable, so use load_nonnull
    // instead
    let meta = if meta_ty.element_type().kind() == llvm::TypeKind::Pointer {
        b.load_nonnull(meta, None)
    } else {
        b.load(meta, None)
    };

    (ptr, meta)
}

pub fn from_immediate(bcx: &Builder, val: ValueRef) -> ValueRef {
    if val_ty(val) == Type::i1(bcx.ccx) {
        bcx.zext(val, Type::i8(bcx.ccx))
    } else {
        val
    }
}

pub fn to_immediate(bcx: &Builder, val: ValueRef, ty: Ty) -> ValueRef {
    if ty.is_bool() {
        bcx.trunc(val, Type::i1(bcx.ccx))
    } else {
        val
    }
}

pub enum Lifetime { Start, End }

impl Lifetime {
    // If LLVM lifetime intrinsic support is enabled (i.e. optimizations
    // on), and `ptr` is nonzero-sized, then extracts the size of `ptr`
    // and the intrinsic for `lt` and passes them to `emit`, which is in
    // charge of generating code to call the passed intrinsic on whatever
    // block of generated code is targetted for the intrinsic.
    //
    // If LLVM lifetime intrinsic support is disabled (i.e.  optimizations
    // off) or `ptr` is zero-sized, then no-op (does not call `emit`).
    pub fn call(self, b: &Builder, ptr: ValueRef) {
        if b.ccx.sess().opts.optimize == config::OptLevel::No {
            return;
        }

        let size = machine::llsize_of_alloc(b.ccx, val_ty(ptr).element_type());
        if size == 0 {
            return;
        }

        let lifetime_intrinsic = b.ccx.get_intrinsic(match self {
            Lifetime::Start => "llvm.lifetime.start",
            Lifetime::End => "llvm.lifetime.end"
        });

        let ptr = b.pointercast(ptr, Type::i8p(b.ccx));
        b.call(lifetime_intrinsic, &[C_u64(b.ccx, size), ptr], None);
    }
}

pub fn call_memcpy<'a, 'tcx>(b: &Builder<'a, 'tcx>,
                               dst: ValueRef,
                               src: ValueRef,
                               n_bytes: ValueRef,
                               align: u32) {
    let ccx = b.ccx;
    let ptr_width = &ccx.sess().target.target.target_pointer_width;
    let key = format!("llvm.memcpy.p0i8.p0i8.i{}", ptr_width);
    let memcpy = ccx.get_intrinsic(&key);
    let src_ptr = b.pointercast(src, Type::i8p(ccx));
    let dst_ptr = b.pointercast(dst, Type::i8p(ccx));
    let size = b.intcast(n_bytes, ccx.int_type(), false);
    let align = C_i32(ccx, align as i32);
    let volatile = C_bool(ccx, false);
    b.call(memcpy, &[dst_ptr, src_ptr, size, align, volatile], None);
}

pub fn memcpy_ty<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    dst: ValueRef,
    src: ValueRef,
    t: Ty<'tcx>,
    align: Option<u32>,
) {
    let ccx = bcx.ccx;

    let size = ccx.size_of(t);
    if size == 0 {
        return;
    }

    let align = align.unwrap_or_else(|| ccx.align_of(t));
    call_memcpy(bcx, dst, src, C_uint(ccx, size), align);
}

pub fn call_memset<'a, 'tcx>(b: &Builder<'a, 'tcx>,
                             ptr: ValueRef,
                             fill_byte: ValueRef,
                             size: ValueRef,
                             align: ValueRef,
                             volatile: bool) -> ValueRef {
    let ptr_width = &b.ccx.sess().target.target.target_pointer_width;
    let intrinsic_key = format!("llvm.memset.p0i8.i{}", ptr_width);
    let llintrinsicfn = b.ccx.get_intrinsic(&intrinsic_key);
    let volatile = C_bool(b.ccx, volatile);
    b.call(llintrinsicfn, &[ptr, fill_byte, size, align, volatile], None)
}

pub fn trans_instance<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, instance: Instance<'tcx>) {
    let _s = if ccx.sess().trans_stats() {
        let mut instance_name = String::new();
        DefPathBasedNames::new(ccx.tcx(), true, true)
            .push_def_path(instance.def_id(), &mut instance_name);
        Some(StatRecorder::new(ccx, instance_name))
    } else {
        None
    };

    // this is an info! to allow collecting monomorphization statistics
    // and to allow finding the last function before LLVM aborts from
    // release builds.
    info!("trans_instance({})", instance);

    let fn_ty = common::instance_ty(ccx.shared(), &instance);
    let sig = common::ty_fn_sig(ccx, fn_ty);
    let sig = ccx.tcx().erase_late_bound_regions_and_normalize(&sig);

    let lldecl = match ccx.instances().borrow().get(&instance) {
        Some(&val) => val,
        None => bug!("Instance `{:?}` not already declared", instance)
    };

    ccx.stats().n_closures.set(ccx.stats().n_closures.get() + 1);

    // The `uwtable` attribute according to LLVM is:
    //
    //     This attribute indicates that the ABI being targeted requires that an
    //     unwind table entry be produced for this function even if we can show
    //     that no exceptions passes by it. This is normally the case for the
    //     ELF x86-64 abi, but it can be disabled for some compilation units.
    //
    // Typically when we're compiling with `-C panic=abort` (which implies this
    // `no_landing_pads` check) we don't need `uwtable` because we can't
    // generate any exceptions! On Windows, however, exceptions include other
    // events such as illegal instructions, segfaults, etc. This means that on
    // Windows we end up still needing the `uwtable` attribute even if the `-C
    // panic=abort` flag is passed.
    //
    // You can also find more info on why Windows is whitelisted here in:
    //      https://bugzilla.mozilla.org/show_bug.cgi?id=1302078
    if !ccx.sess().no_landing_pads() ||
       ccx.sess().target.target.options.is_like_windows {
        attributes::emit_uwtable(lldecl, true);
    }

    let mir = ccx.tcx().instance_mir(instance.def);
    mir::trans_mir(ccx, lldecl, &mir, instance, sig);
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
        "appending" => Some(llvm::Linkage::AppendingLinkage),
        "available_externally" => Some(llvm::Linkage::AvailableExternallyLinkage),
        "common" => Some(llvm::Linkage::CommonLinkage),
        "extern_weak" => Some(llvm::Linkage::ExternalWeakLinkage),
        "external" => Some(llvm::Linkage::ExternalLinkage),
        "internal" => Some(llvm::Linkage::InternalLinkage),
        "linkonce" => Some(llvm::Linkage::LinkOnceAnyLinkage),
        "linkonce_odr" => Some(llvm::Linkage::LinkOnceODRLinkage),
        "private" => Some(llvm::Linkage::PrivateLinkage),
        "weak" => Some(llvm::Linkage::WeakAnyLinkage),
        "weak_odr" => Some(llvm::Linkage::WeakODRLinkage),
        _ => None,
    }
}

pub fn set_link_section(ccx: &CrateContext,
                        llval: ValueRef,
                        attrs: &[ast::Attribute]) {
    if let Some(sect) = attr::first_attr_value_str_by_name(attrs, "link_section") {
        if contains_null(&sect.as_str()) {
            ccx.sess().fatal(&format!("Illegal null byte in link_section value: `{}`", &sect));
        }
        unsafe {
            let buf = CString::new(sect.as_str().as_bytes()).unwrap();
            llvm::LLVMSetSection(llval, buf.as_ptr());
        }
    }
}

// check for the #[rustc_error] annotation, which forces an
// error in trans. This is used to write compile-fail tests
// that actually test that compilation succeeds without
// reporting an error.
fn check_for_rustc_errors_attr(tcx: TyCtxt) {
    if let Some((id, span)) = *tcx.sess.entry_fn.borrow() {
        let main_def_id = tcx.hir.local_def_id(id);

        if tcx.has_attr(main_def_id, "rustc_error") {
            tcx.sess.span_fatal(span, "compilation successful");
        }
    }
}

/// Create the `main` function which will initialise the rust runtime and call
/// users main function.
fn maybe_create_entry_wrapper(ccx: &CrateContext) {
    let (main_def_id, span) = match *ccx.sess().entry_fn.borrow() {
        Some((id, span)) => {
            (ccx.tcx().hir.local_def_id(id), span)
        }
        None => return,
    };

    let instance = Instance::mono(ccx.tcx(), main_def_id);

    if !ccx.codegen_unit().contains_item(&TransItem::Fn(instance)) {
        // We want to create the wrapper in the same codegen unit as Rust's main
        // function.
        return;
    }

    let main_llfn = callee::get_fn(ccx, instance);

    let et = ccx.sess().entry_type.get().unwrap();
    match et {
        config::EntryMain => create_entry_fn(ccx, span, main_llfn, true),
        config::EntryStart => create_entry_fn(ccx, span, main_llfn, false),
        config::EntryNone => {}    // Do nothing.
    }

    fn create_entry_fn(ccx: &CrateContext,
                       sp: Span,
                       rust_main: ValueRef,
                       use_start_lang_item: bool) {
        let llfty = Type::func(&[ccx.int_type(), Type::i8p(ccx).ptr_to()], &ccx.int_type());

        if declare::get_defined_value(ccx, "main").is_some() {
            // FIXME: We should be smart and show a better diagnostic here.
            ccx.sess().struct_span_err(sp, "entry symbol `main` defined multiple times")
                      .help("did you use #[no_mangle] on `fn main`? Use #[start] instead")
                      .emit();
            ccx.sess().abort_if_errors();
            bug!();
        }
        let llfn = declare::declare_cfn(ccx, "main", llfty);

        // `main` should respect same config for frame pointer elimination as rest of code
        attributes::set_frame_pointer_elimination(ccx, llfn);

        let bld = Builder::new_block(ccx, llfn, "top");

        debuginfo::gdb::insert_reference_to_gdb_debug_scripts_section_global(ccx, &bld);

        let (start_fn, args) = if use_start_lang_item {
            let start_def_id = ccx.tcx().require_lang_item(StartFnLangItem);
            let start_instance = Instance::mono(ccx.tcx(), start_def_id);
            let start_fn = callee::get_fn(ccx, start_instance);
            (start_fn, vec![bld.pointercast(rust_main, Type::i8p(ccx).ptr_to()), get_param(llfn, 0),
                get_param(llfn, 1)])
        } else {
            debug!("using user-defined start fn");
            (rust_main, vec![get_param(llfn, 0 as c_uint), get_param(llfn, 1 as c_uint)])
        };

        let result = bld.call(start_fn, &args, None);
        bld.ret(result);
    }
}

fn contains_null(s: &str) -> bool {
    s.bytes().any(|b| b == 0)
}

fn write_metadata<'a, 'gcx>(tcx: TyCtxt<'a, 'gcx, 'gcx>,
                            link_meta: &LinkMeta,
                            exported_symbols: &NodeSet)
                            -> (ContextRef, ModuleRef,
                                EncodedMetadata, EncodedMetadataHashes) {
    use std::io::Write;
    use flate2::Compression;
    use flate2::write::DeflateEncoder;

    let (metadata_llcx, metadata_llmod) = unsafe {
        context::create_context_and_module(tcx.sess, "metadata")
    };

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum MetadataKind {
        None,
        Uncompressed,
        Compressed
    }

    let kind = tcx.sess.crate_types.borrow().iter().map(|ty| {
        match *ty {
            config::CrateTypeExecutable |
            config::CrateTypeStaticlib |
            config::CrateTypeCdylib => MetadataKind::None,

            config::CrateTypeRlib => MetadataKind::Uncompressed,

            config::CrateTypeDylib |
            config::CrateTypeProcMacro => MetadataKind::Compressed,
        }
    }).max().unwrap();

    if kind == MetadataKind::None {
        return (metadata_llcx,
                metadata_llmod,
                EncodedMetadata::new(),
                EncodedMetadataHashes::new());
    }

    let cstore = &tcx.sess.cstore;
    let (metadata, hashes) = cstore.encode_metadata(tcx,
                                                    &link_meta,
                                                    exported_symbols);
    if kind == MetadataKind::Uncompressed {
        return (metadata_llcx, metadata_llmod, metadata, hashes);
    }

    assert!(kind == MetadataKind::Compressed);
    let mut compressed = cstore.metadata_encoding_version().to_vec();
    DeflateEncoder::new(&mut compressed, Compression::Fast)
        .write_all(&metadata.raw_data).unwrap();

    let llmeta = C_bytes_in_context(metadata_llcx, &compressed);
    let llconst = C_struct_in_context(metadata_llcx, &[llmeta], false);
    let name = symbol_export::metadata_symbol_name(tcx);
    let buf = CString::new(name).unwrap();
    let llglobal = unsafe {
        llvm::LLVMAddGlobal(metadata_llmod, val_ty(llconst).to_ref(), buf.as_ptr())
    };
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        let section_name = metadata::metadata_section_name(&tcx.sess.target.target);
        let name = CString::new(section_name).unwrap();
        llvm::LLVMSetSection(llglobal, name.as_ptr());

        // Also generate a .section directive to force no
        // flags, at least for ELF outputs, so that the
        // metadata doesn't get loaded into memory.
        let directive = format!(".section {}", section_name);
        let directive = CString::new(directive).unwrap();
        llvm::LLVMSetModuleInlineAsm(metadata_llmod, directive.as_ptr())
    }
    return (metadata_llcx, metadata_llmod, metadata, hashes);
}

// Create a `__imp_<symbol> = &symbol` global for every public static `symbol`.
// This is required to satisfy `dllimport` references to static data in .rlibs
// when using MSVC linker.  We do this only for data, as linker can fix up
// code references on its own.
// See #26591, #27438
fn create_imps(sess: &Session,
               llvm_module: &ModuleLlvm) {
    // The x86 ABI seems to require that leading underscores are added to symbol
    // names, so we need an extra underscore on 32-bit. There's also a leading
    // '\x01' here which disables LLVM's symbol mangling (e.g. no extra
    // underscores added in front).
    let prefix = if sess.target.target.target_pointer_width == "32" {
        "\x01__imp__"
    } else {
        "\x01__imp_"
    };
    unsafe {
        let exported: Vec<_> = iter_globals(llvm_module.llmod)
                                   .filter(|&val| {
                                       llvm::LLVMRustGetLinkage(val) ==
                                       llvm::Linkage::ExternalLinkage &&
                                       llvm::LLVMIsDeclaration(val) == 0
                                   })
                                   .collect();

        let i8p_ty = Type::i8p_llcx(llvm_module.llcx);
        for val in exported {
            let name = CStr::from_ptr(llvm::LLVMGetValueName(val));
            let mut imp_name = prefix.as_bytes().to_vec();
            imp_name.extend(name.to_bytes());
            let imp_name = CString::new(imp_name).unwrap();
            let imp = llvm::LLVMAddGlobal(llvm_module.llmod,
                                          i8p_ty.to_ref(),
                                          imp_name.as_ptr() as *const _);
            llvm::LLVMSetInitializer(imp, consts::ptrcast(val, i8p_ty));
            llvm::LLVMRustSetLinkage(imp, llvm::Linkage::ExternalLinkage);
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

/// The context provided lists a set of reachable ids as calculated by
/// middle::reachable, but this contains far more ids and symbols than we're
/// actually exposing from the object file. This function will filter the set in
/// the context to the set of ids which correspond to symbols that are exposed
/// from the object file being generated.
///
/// This list is later used by linkers to determine the set of symbols needed to
/// be exposed from a dynamic library and it's also encoded into the metadata.
pub fn find_exported_symbols(tcx: TyCtxt, reachable: &NodeSet) -> NodeSet {
    reachable.iter().cloned().filter(|&id| {
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
        match tcx.hir.get(id) {
            hir_map::NodeForeignItem(..) => {
                let def_id = tcx.hir.local_def_id(id);
                tcx.sess.cstore.is_statically_included_foreign_item(def_id)
            }

            // Only consider nodes that actually have exported symbols.
            hir_map::NodeItem(&hir::Item {
                node: hir::ItemStatic(..), .. }) |
            hir_map::NodeItem(&hir::Item {
                node: hir::ItemFn(..), .. }) |
            hir_map::NodeImplItem(&hir::ImplItem {
                node: hir::ImplItemKind::Method(..), .. }) => {
                let def_id = tcx.hir.local_def_id(id);
                let generics = tcx.generics_of(def_id);
                let attributes = tcx.get_attrs(def_id);
                (generics.parent_types == 0 && generics.types.is_empty()) &&
                // Functions marked with #[inline] are only ever translated
                // with "internal" linkage and are never exported.
                !attr::requests_inline(&attributes)
            }

            _ => false
        }
    }).collect()
}

pub fn trans_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             analysis: ty::CrateAnalysis,
                             incremental_hashes_map: IncrementalHashesMap,
                             output_filenames: &OutputFilenames)
                             -> OngoingCrateTranslation {
    check_for_rustc_errors_attr(tcx);

    // Be careful with this krate: obviously it gives access to the
    // entire contents of the krate. So if you push any subtasks of
    // `TransCrate`, you need to be careful to register "reads" of the
    // particular items that will be processed.
    let krate = tcx.hir.krate();
    let ty::CrateAnalysis { reachable, .. } = analysis;
    let check_overflow = tcx.sess.overflow_checks();
    let link_meta = link::build_link_meta(&incremental_hashes_map);
    let exported_symbol_node_ids = find_exported_symbols(tcx, &reachable);

    let shared_ccx = SharedCrateContext::new(tcx,
                                             check_overflow,
                                             output_filenames);
    // Translate the metadata.
    let (metadata_llcx, metadata_llmod, metadata, metadata_incr_hashes) =
        time(tcx.sess.time_passes(), "write metadata", || {
            write_metadata(tcx, &link_meta, &exported_symbol_node_ids)
        });

    let metadata_module = ModuleTranslation {
        name: link::METADATA_MODULE_NAME.to_string(),
        symbol_name_hash: 0, // we always rebuild metadata, at least for now
        source: ModuleSource::Translated(ModuleLlvm {
            llcx: metadata_llcx,
            llmod: metadata_llmod,
        }),
        kind: ModuleKind::Metadata,
    };

    let no_builtins = attr::contains_name(&krate.attrs, "no_builtins");
    let time_graph = if tcx.sess.opts.debugging_opts.trans_time_graph {
        Some(time_graph::TimeGraph::new())
    } else {
        None
    };

    // Skip crate items and just output metadata in -Z no-trans mode.
    if tcx.sess.opts.debugging_opts.no_trans ||
       !tcx.sess.opts.output_types.should_trans() {
        let empty_exported_symbols = ExportedSymbols::empty();
        let linker_info = LinkerInfo::new(&shared_ccx, &empty_exported_symbols);
        let ongoing_translation = write::start_async_translation(
            tcx.sess,
            output_filenames,
            time_graph.clone(),
            tcx.crate_name(LOCAL_CRATE),
            link_meta,
            metadata,
            Arc::new(empty_exported_symbols),
            no_builtins,
            None,
            linker_info,
            false);

        ongoing_translation.submit_pre_translated_module_to_llvm(tcx.sess, metadata_module, true);

        assert_and_save_dep_graph(tcx,
                                  incremental_hashes_map,
                                  metadata_incr_hashes,
                                  link_meta);

        ongoing_translation.check_for_errors(tcx.sess);

        return ongoing_translation;
    }

    let exported_symbols = Arc::new(ExportedSymbols::compute(tcx,
                                                             &exported_symbol_node_ids));

    // Run the translation item collector and partition the collected items into
    // codegen units.
    let (translation_items, codegen_units) =
        collect_and_partition_translation_items(&shared_ccx, &exported_symbols);

    assert!(codegen_units.len() <= 1 || !tcx.sess.lto());

    let linker_info = LinkerInfo::new(&shared_ccx, &exported_symbols);
    let subsystem = attr::first_attr_value_str_by_name(&krate.attrs,
                                                       "windows_subsystem");
    let windows_subsystem = subsystem.map(|subsystem| {
        if subsystem != "windows" && subsystem != "console" {
            tcx.sess.fatal(&format!("invalid windows subsystem `{}`, only \
                                     `windows` and `console` are allowed",
                                    subsystem));
        }
        subsystem.to_string()
    });

    let no_integrated_as = tcx.sess.opts.cg.no_integrated_as ||
        (tcx.sess.target.target.options.no_integrated_as &&
         (output_filenames.outputs.contains_key(&OutputType::Object) ||
          output_filenames.outputs.contains_key(&OutputType::Exe)));

    let ongoing_translation = write::start_async_translation(
        tcx.sess,
        output_filenames,
        time_graph.clone(),
        tcx.crate_name(LOCAL_CRATE),
        link_meta,
        metadata,
        exported_symbols.clone(),
        no_builtins,
        windows_subsystem,
        linker_info,
        no_integrated_as);

    // Translate an allocator shim, if any
    //
    // If LTO is enabled and we've got some previous LLVM module we translated
    // above, then we can just translate directly into that LLVM module. If not,
    // however, we need to create a separate module and trans into that. Note
    // that the separate translation is critical for the standard library where
    // the rlib's object file doesn't have allocator functions but the dylib
    // links in an object file that has allocator functions. When we're
    // compiling a final LTO artifact, though, there's no need to worry about
    // this as we're not working with this dual "rlib/dylib" functionality.
    let allocator_module = if tcx.sess.lto() {
        None
    } else if let Some(kind) = tcx.sess.allocator_kind.get() {
        unsafe {
            let (llcx, llmod) =
                context::create_context_and_module(tcx.sess, "allocator");
            let modules = ModuleLlvm {
                llmod: llmod,
                llcx: llcx,
            };
            time(tcx.sess.time_passes(), "write allocator module", || {
                allocator::trans(tcx, &modules, kind)
            });

            Some(ModuleTranslation {
                name: link::ALLOCATOR_MODULE_NAME.to_string(),
                symbol_name_hash: 0, // we always rebuild allocator shims
                source: ModuleSource::Translated(modules),
                kind: ModuleKind::Allocator,
            })
        }
    } else {
        None
    };

    if let Some(allocator_module) = allocator_module {
        ongoing_translation.submit_pre_translated_module_to_llvm(tcx.sess, allocator_module, false);
    }

    let codegen_unit_count = codegen_units.len();
    ongoing_translation.submit_pre_translated_module_to_llvm(tcx.sess,
                                                             metadata_module,
                                                             codegen_unit_count == 0);

    let translation_items = Arc::new(translation_items);

    let mut all_stats = Stats::default();
    let mut module_dispositions = tcx.sess.opts.incremental.as_ref().map(|_| Vec::new());

    // We sort the codegen units by size. This way we can schedule work for LLVM
    // a bit more efficiently. Note that "size" is defined rather crudely at the
    // moment as it is just the number of TransItems in the CGU, not taking into
    // account the size of each TransItem.
    let codegen_units = {
        let mut codegen_units = codegen_units;
        codegen_units.sort_by_key(|cgu| -(cgu.items().len() as isize));
        codegen_units
    };

    let mut total_trans_time = Duration::new(0, 0);

    for (cgu_index, cgu) in codegen_units.into_iter().enumerate() {
        ongoing_translation.wait_for_signal_to_translate_item();
        ongoing_translation.check_for_errors(tcx.sess);

        let start_time = Instant::now();

        let module = {
            let _timing_guard = time_graph
                .as_ref()
                .map(|time_graph| time_graph.start(write::TRANS_WORKER_TIMELINE,
                                                   write::TRANS_WORK_PACKAGE_KIND));
            let dep_node = cgu.work_product_dep_node();
            let ((stats, module), _) =
                tcx.dep_graph.with_task(dep_node,
                                        AssertDepGraphSafe(&shared_ccx),
                                        AssertDepGraphSafe((cgu,
                                                            translation_items.clone(),
                                                            exported_symbols.clone())),
                                        module_translation);
            all_stats.extend(stats);

            if let Some(ref mut module_dispositions) = module_dispositions {
                module_dispositions.push(module.disposition());
            }

            module
        };

        let time_to_translate = Instant::now().duration_since(start_time);

        // We assume that the cost to run LLVM on a CGU is proportional to
        // the time we needed for translating it.
        let cost = time_to_translate.as_secs() * 1_000_000_000 +
                   time_to_translate.subsec_nanos() as u64;

        total_trans_time += time_to_translate;

        let is_last_cgu = (cgu_index + 1) == codegen_unit_count;

        ongoing_translation.submit_translated_module_to_llvm(tcx.sess,
                                                             module,
                                                             cost,
                                                             is_last_cgu);
        ongoing_translation.check_for_errors(tcx.sess);
    }

    // Since the main thread is sometimes blocked during trans, we keep track
    // -Ztime-passes output manually.
    print_time_passes_entry(tcx.sess.time_passes(),
                            "translate to LLVM IR",
                            total_trans_time);

    if let Some(module_dispositions) = module_dispositions {
        assert_module_sources::assert_module_sources(tcx, &module_dispositions);
    }

    fn module_translation<'a, 'tcx>(
        scx: AssertDepGraphSafe<&SharedCrateContext<'a, 'tcx>>,
        args: AssertDepGraphSafe<(CodegenUnit<'tcx>,
                                  Arc<FxHashSet<TransItem<'tcx>>>,
                                  Arc<ExportedSymbols>)>)
        -> (Stats, ModuleTranslation)
    {
        // FIXME(#40304): We ought to be using the id as a key and some queries, I think.
        let AssertDepGraphSafe(scx) = scx;
        let AssertDepGraphSafe((cgu, crate_trans_items, exported_symbols)) = args;

        let cgu_name = String::from(cgu.name());
        let cgu_id = cgu.work_product_id();
        let symbol_name_hash = cgu.compute_symbol_name_hash(scx, &exported_symbols);

        // Check whether there is a previous work-product we can
        // re-use.  Not only must the file exist, and the inputs not
        // be dirty, but the hash of the symbols we will generate must
        // be the same.
        let previous_work_product =
            scx.dep_graph().previous_work_product(&cgu_id).and_then(|work_product| {
                if work_product.input_hash == symbol_name_hash {
                    debug!("trans_reuse_previous_work_products: reusing {:?}", work_product);
                    Some(work_product)
                } else {
                    if scx.sess().opts.debugging_opts.incremental_info {
                        eprintln!("incremental: CGU `{}` invalidated because of \
                                   changed partitioning hash.",
                                   cgu.name());
                    }
                    debug!("trans_reuse_previous_work_products: \
                            not reusing {:?} because hash changed to {:?}",
                           work_product, symbol_name_hash);
                    None
                }
            });

        if let Some(buf) = previous_work_product {
            // Don't need to translate this module.
            let module = ModuleTranslation {
                name: cgu_name,
                symbol_name_hash,
                source: ModuleSource::Preexisting(buf.clone()),
                kind: ModuleKind::Regular,
            };
            return (Stats::default(), module);
        }

        // Instantiate translation items without filling out definitions yet...
        let lcx = LocalCrateContext::new(scx, cgu, crate_trans_items, exported_symbols);
        let module = {
            let ccx = CrateContext::new(scx, &lcx);
            let trans_items = ccx.codegen_unit()
                                 .items_in_deterministic_order(ccx.tcx());
            for &(trans_item, (linkage, visibility)) in &trans_items {
                trans_item.predefine(&ccx, linkage, visibility);
            }

            // ... and now that we have everything pre-defined, fill out those definitions.
            for &(trans_item, _) in &trans_items {
                trans_item.define(&ccx);
            }

            // If this codegen unit contains the main function, also create the
            // wrapper here
            maybe_create_entry_wrapper(&ccx);

            // Run replace-all-uses-with for statics that need it
            for &(old_g, new_g) in ccx.statics_to_rauw().borrow().iter() {
                unsafe {
                    let bitcast = llvm::LLVMConstPointerCast(new_g, llvm::LLVMTypeOf(old_g));
                    llvm::LLVMReplaceAllUsesWith(old_g, bitcast);
                    llvm::LLVMDeleteGlobal(old_g);
                }
            }

            // Create the llvm.used variable
            // This variable has type [N x i8*] and is stored in the llvm.metadata section
            if !ccx.used_statics().borrow().is_empty() {
                let name = CString::new("llvm.used").unwrap();
                let section = CString::new("llvm.metadata").unwrap();
                let array = C_array(Type::i8(&ccx).ptr_to(), &*ccx.used_statics().borrow());

                unsafe {
                    let g = llvm::LLVMAddGlobal(ccx.llmod(),
                                                val_ty(array).to_ref(),
                                                name.as_ptr());
                    llvm::LLVMSetInitializer(g, array);
                    llvm::LLVMRustSetLinkage(g, llvm::Linkage::AppendingLinkage);
                    llvm::LLVMSetSection(g, section.as_ptr());
                }
            }

            // Finalize debuginfo
            if ccx.sess().opts.debuginfo != NoDebugInfo {
                debuginfo::finalize(&ccx);
            }

            let llvm_module = ModuleLlvm {
                llcx: ccx.llcx(),
                llmod: ccx.llmod(),
            };

            // In LTO mode we inject the allocator shim into the existing
            // module.
            if ccx.sess().lto() {
                if let Some(kind) = ccx.sess().allocator_kind.get() {
                    time(ccx.sess().time_passes(), "write allocator module", || {
                        unsafe {
                            allocator::trans(ccx.tcx(), &llvm_module, kind);
                        }
                    });
                }
            }

            // Adjust exported symbols for MSVC dllimport
            if ccx.sess().target.target.options.is_like_msvc &&
               ccx.sess().crate_types.borrow().iter().any(|ct| *ct == config::CrateTypeRlib) {
                create_imps(ccx.sess(), &llvm_module);
            }

            ModuleTranslation {
                name: cgu_name,
                symbol_name_hash,
                source: ModuleSource::Translated(llvm_module),
                kind: ModuleKind::Regular,
            }
        };

        (lcx.into_stats(), module)
    }

    symbol_names_test::report_symbol_names(tcx);

    if shared_ccx.sess().trans_stats() {
        println!("--- trans stats ---");
        println!("n_glues_created: {}", all_stats.n_glues_created.get());
        println!("n_null_glues: {}", all_stats.n_null_glues.get());
        println!("n_real_glues: {}", all_stats.n_real_glues.get());

        println!("n_fns: {}", all_stats.n_fns.get());
        println!("n_inlines: {}", all_stats.n_inlines.get());
        println!("n_closures: {}", all_stats.n_closures.get());
        println!("fn stats:");
        all_stats.fn_stats.borrow_mut().sort_by(|&(_, insns_a), &(_, insns_b)| {
            insns_b.cmp(&insns_a)
        });
        for tuple in all_stats.fn_stats.borrow().iter() {
            match *tuple {
                (ref name, insns) => {
                    println!("{} insns, {}", insns, *name);
                }
            }
        }
    }

    if shared_ccx.sess().count_llvm_insns() {
        for (k, v) in all_stats.llvm_insns.borrow().iter() {
            println!("{:7} {}", *v, *k);
        }
    }

    ongoing_translation.check_for_errors(tcx.sess);

    assert_and_save_dep_graph(tcx,
                              incremental_hashes_map,
                              metadata_incr_hashes,
                              link_meta);
    ongoing_translation
}

fn assert_and_save_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       incremental_hashes_map: IncrementalHashesMap,
                                       metadata_incr_hashes: EncodedMetadataHashes,
                                       link_meta: LinkMeta) {
    time(tcx.sess.time_passes(),
         "assert dep graph",
         || rustc_incremental::assert_dep_graph(tcx));

    time(tcx.sess.time_passes(),
         "serialize dep graph",
         || rustc_incremental::save_dep_graph(tcx,
                                              incremental_hashes_map,
                                              &metadata_incr_hashes,
                                              link_meta.crate_hash));
}

#[inline(never)] // give this a place in the profiler
fn assert_symbols_are_distinct<'a, 'tcx, I>(tcx: TyCtxt<'a, 'tcx, 'tcx>, trans_items: I)
    where I: Iterator<Item=&'a TransItem<'tcx>>
{
    let mut symbols: Vec<_> = trans_items.map(|trans_item| {
        (trans_item, trans_item.symbol_name(tcx))
    }).collect();

    (&mut symbols[..]).sort_by(|&(_, ref sym1), &(_, ref sym2)|{
        sym1.cmp(sym2)
    });

    for pair in (&symbols[..]).windows(2) {
        let sym1 = &pair[0].1;
        let sym2 = &pair[1].1;

        if *sym1 == *sym2 {
            let trans_item1 = pair[0].0;
            let trans_item2 = pair[1].0;

            let span1 = trans_item1.local_span(tcx);
            let span2 = trans_item2.local_span(tcx);

            // Deterministically select one of the spans for error reporting
            let span = match (span1, span2) {
                (Some(span1), Some(span2)) => {
                    Some(if span1.lo.0 > span2.lo.0 {
                        span1
                    } else {
                        span2
                    })
                }
                (Some(span), None) |
                (None, Some(span)) => Some(span),
                _ => None
            };

            let error_message = format!("symbol `{}` is already defined", sym1);

            if let Some(span) = span {
                tcx.sess.span_fatal(span, &error_message)
            } else {
                tcx.sess.fatal(&error_message)
            }
        }
    }
}

fn collect_and_partition_translation_items<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                                     exported_symbols: &ExportedSymbols)
                                                     -> (FxHashSet<TransItem<'tcx>>,
                                                         Vec<CodegenUnit<'tcx>>) {
    let time_passes = scx.sess().time_passes();

    let collection_mode = match scx.sess().opts.debugging_opts.print_trans_items {
        Some(ref s) => {
            let mode_string = s.to_lowercase();
            let mode_string = mode_string.trim();
            if mode_string == "eager" {
                TransItemCollectionMode::Eager
            } else {
                if mode_string != "lazy" {
                    let message = format!("Unknown codegen-item collection mode '{}'. \
                                           Falling back to 'lazy' mode.",
                                           mode_string);
                    scx.sess().warn(&message);
                }

                TransItemCollectionMode::Lazy
            }
        }
        None => TransItemCollectionMode::Lazy
    };

    let (items, inlining_map) =
        time(time_passes, "translation item collection", || {
            collector::collect_crate_translation_items(&scx,
                                                       exported_symbols,
                                                       collection_mode)
    });

    assert_symbols_are_distinct(scx.tcx(), items.iter());

    let strategy = if scx.sess().opts.debugging_opts.incremental.is_some() {
        PartitioningStrategy::PerModule
    } else {
        PartitioningStrategy::FixedUnitCount(scx.sess().opts.cg.codegen_units)
    };

    let codegen_units = time(time_passes, "codegen unit partitioning", || {
        partitioning::partition(scx,
                                items.iter().cloned(),
                                strategy,
                                &inlining_map,
                                exported_symbols)
    });

    assert!(scx.tcx().sess.opts.cg.codegen_units == codegen_units.len() ||
            scx.tcx().sess.opts.debugging_opts.incremental.is_some());

    let translation_items: FxHashSet<TransItem<'tcx>> = items.iter().cloned().collect();

    if scx.sess().opts.debugging_opts.print_trans_items.is_some() {
        let mut item_to_cgus = FxHashMap();

        for cgu in &codegen_units {
            for (&trans_item, &linkage) in cgu.items() {
                item_to_cgus.entry(trans_item)
                            .or_insert(Vec::new())
                            .push((cgu.name().clone(), linkage));
            }
        }

        let mut item_keys: Vec<_> = items
            .iter()
            .map(|i| {
                let mut output = i.to_string(scx.tcx());
                output.push_str(" @@");
                let mut empty = Vec::new();
                let mut cgus = item_to_cgus.get_mut(i).unwrap_or(&mut empty);
                cgus.as_mut_slice().sort_by_key(|&(ref name, _)| name.clone());
                cgus.dedup();
                for &(ref cgu_name, (linkage, _)) in cgus.iter() {
                    output.push_str(" ");
                    output.push_str(&cgu_name);

                    let linkage_abbrev = match linkage {
                        llvm::Linkage::ExternalLinkage => "External",
                        llvm::Linkage::AvailableExternallyLinkage => "Available",
                        llvm::Linkage::LinkOnceAnyLinkage => "OnceAny",
                        llvm::Linkage::LinkOnceODRLinkage => "OnceODR",
                        llvm::Linkage::WeakAnyLinkage => "WeakAny",
                        llvm::Linkage::WeakODRLinkage => "WeakODR",
                        llvm::Linkage::AppendingLinkage => "Appending",
                        llvm::Linkage::InternalLinkage => "Internal",
                        llvm::Linkage::PrivateLinkage => "Private",
                        llvm::Linkage::ExternalWeakLinkage => "ExternalWeak",
                        llvm::Linkage::CommonLinkage => "Common",
                    };

                    output.push_str("[");
                    output.push_str(linkage_abbrev);
                    output.push_str("]");
                }
                output
            })
            .collect();

        item_keys.sort();

        for item in item_keys {
            println!("TRANS_ITEM {}", item);
        }
    }

    (translation_items, codegen_units)
}

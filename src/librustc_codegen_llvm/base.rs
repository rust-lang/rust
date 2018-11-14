// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Codegen the completed AST to the LLVM IR.
//!
//! Some functions here, such as codegen_block and codegen_expr, return a value --
//! the result of the codegen to LLVM -- while others, such as codegen_fn
//! and mono_item, are called only for the side effect of adding a
//! particular definition to the LLVM IR output we're producing.
//!
//! Hopefully useful general knowledge about codegen:
//!
//!   * There's no way to find out the Ty type of a Value.  Doing so
//!     would be "trying to get the eggs out of an omelette" (credit:
//!     pcwalton).  You can, instead, find out its llvm::Type by calling val_ty,
//!     but one llvm::Type corresponds to many `Ty`s; for instance, tup(int, int,
//!     int) and rec(x=int, y=int, z=int) will have the same llvm::Type.

use super::ModuleLlvm;
use super::ModuleCodegen;
use super::ModuleKind;
use super::CachedModuleCodegen;

use abi;
use back::write::{self, OngoingCodegen};
use llvm::{self, TypeKind, get_param};
use metadata;
use rustc::dep_graph::cgu_reuse_tracker::CguReuse;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::middle::lang_items::StartFnLangItem;
use rustc::middle::weak_lang_items;
use rustc::mir::mono::{Linkage, Visibility, Stats, CodegenUnitNameBuilder};
use rustc::middle::cstore::{EncodedMetadata};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::{self, Align, TyLayout, LayoutOf, VariantIdx};
use rustc::ty::query::Providers;
use rustc::middle::cstore::{self, LinkagePreference};
use rustc::middle::exported_symbols;
use rustc::util::common::{time, print_time_passes_entry};
use rustc::util::profiling::ProfileCategory;
use rustc::session::config::{self, DebugInfo, EntryFnType, Lto};
use rustc::session::Session;
use rustc_incremental;
use allocator;
use mir::place::PlaceRef;
use attributes;
use builder::{Builder, MemFlags};
use callee;
use common::{C_bool, C_bytes_in_context, C_usize};
use rustc_mir::monomorphize::item::DefPathBasedNames;
use common::{C_struct_in_context, C_array, val_ty};
use consts;
use context::CodegenCx;
use debuginfo;
use declare;
use meth;
use mir;
use monomorphize::Instance;
use monomorphize::partitioning::{CodegenUnit, CodegenUnitExt};
use rustc_codegen_utils::symbol_names_test;
use time_graph;
use mono_item::{MonoItem, MonoItemExt};
use type_::Type;
use type_of::LayoutLlvmExt;
use rustc::util::nodemap::FxHashMap;
use CrateInfo;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::indexed_vec::Idx;

use std::any::Any;
use std::cmp;
use std::ffi::CString;
use std::ops::{Deref, DerefMut};
use std::sync::mpsc;
use std::time::{Instant, Duration};
use syntax_pos::Span;
use syntax_pos::symbol::InternedString;
use syntax::attr;
use rustc::hir::{self, CodegenFnAttrs};

use value::Value;

use mir::operand::OperandValue;

use rustc_codegen_utils::check_for_rustc_errors_attr;

pub struct StatRecorder<'a, 'll: 'a, 'tcx: 'll> {
    cx: &'a CodegenCx<'ll, 'tcx>,
    name: Option<String>,
    istart: usize,
}

impl StatRecorder<'a, 'll, 'tcx> {
    pub fn new(cx: &'a CodegenCx<'ll, 'tcx>, name: String) -> Self {
        let istart = cx.stats.borrow().n_llvm_insns;
        StatRecorder {
            cx,
            name: Some(name),
            istart,
        }
    }
}

impl Drop for StatRecorder<'a, 'll, 'tcx> {
    fn drop(&mut self) {
        if self.cx.sess().codegen_stats() {
            let mut stats = self.cx.stats.borrow_mut();
            let iend = stats.n_llvm_insns;
            stats.fn_stats.push((self.name.take().unwrap(), iend - self.istart));
            stats.n_fns += 1;
            // Reset LLVM insn count to avoid compound costs.
            stats.n_llvm_insns = self.istart;
        }
    }
}

pub fn bin_op_to_icmp_predicate(op: hir::BinOpKind,
                                signed: bool)
                                -> llvm::IntPredicate {
    match op {
        hir::BinOpKind::Eq => llvm::IntEQ,
        hir::BinOpKind::Ne => llvm::IntNE,
        hir::BinOpKind::Lt => if signed { llvm::IntSLT } else { llvm::IntULT },
        hir::BinOpKind::Le => if signed { llvm::IntSLE } else { llvm::IntULE },
        hir::BinOpKind::Gt => if signed { llvm::IntSGT } else { llvm::IntUGT },
        hir::BinOpKind::Ge => if signed { llvm::IntSGE } else { llvm::IntUGE },
        op => {
            bug!("comparison_op_to_icmp_predicate: expected comparison operator, \
                  found {:?}",
                 op)
        }
    }
}

pub fn bin_op_to_fcmp_predicate(op: hir::BinOpKind) -> llvm::RealPredicate {
    match op {
        hir::BinOpKind::Eq => llvm::RealOEQ,
        hir::BinOpKind::Ne => llvm::RealUNE,
        hir::BinOpKind::Lt => llvm::RealOLT,
        hir::BinOpKind::Le => llvm::RealOLE,
        hir::BinOpKind::Gt => llvm::RealOGT,
        hir::BinOpKind::Ge => llvm::RealOGE,
        op => {
            bug!("comparison_op_to_fcmp_predicate: expected comparison operator, \
                  found {:?}",
                 op);
        }
    }
}

pub fn compare_simd_types(
    bx: &Builder<'a, 'll, 'tcx>,
    lhs: &'ll Value,
    rhs: &'ll Value,
    t: Ty<'tcx>,
    ret_ty: &'ll Type,
    op: hir::BinOpKind
) -> &'ll Value {
    let signed = match t.sty {
        ty::Float(_) => {
            let cmp = bin_op_to_fcmp_predicate(op);
            return bx.sext(bx.fcmp(cmp, lhs, rhs), ret_ty);
        },
        ty::Uint(_) => false,
        ty::Int(_) => true,
        _ => bug!("compare_simd_types: invalid SIMD type"),
    };

    let cmp = bin_op_to_icmp_predicate(op, signed);
    // LLVM outputs an `< size x i1 >`, so we need to perform a sign extension
    // to get the correctly sized type. This will compile to a single instruction
    // once the IR is converted to assembly if the SIMD instruction is supported
    // by the target architecture.
    bx.sext(bx.icmp(cmp, lhs, rhs), ret_ty)
}

/// Retrieve the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit funny. It is intended for use
/// in an upcast, where the new vtable for an object will be derived
/// from the old one.
pub fn unsized_info(
    cx: &CodegenCx<'ll, 'tcx>,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
    old_info: Option<&'ll Value>,
) -> &'ll Value {
    let (source, target) = cx.tcx.struct_lockstep_tails(source, target);
    match (&source.sty, &target.sty) {
        (&ty::Array(_, len), &ty::Slice(_)) => {
            C_usize(cx, len.unwrap_usize(cx.tcx))
        }
        (&ty::Dynamic(..), &ty::Dynamic(..)) => {
            // For now, upcasts are limited to changes in marker
            // traits, and hence never actually require an actual
            // change to the vtable.
            old_info.expect("unsized_info: missing old info for trait upcast")
        }
        (_, &ty::Dynamic(ref data, ..)) => {
            let vtable_ptr = cx.layout_of(cx.tcx.mk_mut_ptr(target))
                .field(cx, abi::FAT_PTR_EXTRA);
            consts::ptrcast(meth::get_vtable(cx, source, data.principal()),
                            vtable_ptr.llvm_type(cx))
        }
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}",
                  source,
                  target),
    }
}

/// Coerce `src` to `dst_ty`. `src_ty` must be a thin pointer.
pub fn unsize_thin_ptr(
    bx: &Builder<'a, 'll, 'tcx>,
    src: &'ll Value,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>
) -> (&'ll Value, &'ll Value) {
    debug!("unsize_thin_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::Ref(_, a, _),
         &ty::Ref(_, b, _)) |
        (&ty::Ref(_, a, _),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
            assert!(bx.cx.type_is_sized(a));
            let ptr_ty = bx.cx.layout_of(b).llvm_type(bx.cx).ptr_to();
            (bx.pointercast(src, ptr_ty), unsized_info(bx.cx, a, b, None))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            let (a, b) = (src_ty.boxed_ty(), dst_ty.boxed_ty());
            assert!(bx.cx.type_is_sized(a));
            let ptr_ty = bx.cx.layout_of(b).llvm_type(bx.cx).ptr_to();
            (bx.pointercast(src, ptr_ty), unsized_info(bx.cx, a, b, None))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b);

            let src_layout = bx.cx.layout_of(src_ty);
            let dst_layout = bx.cx.layout_of(dst_ty);
            let mut result = None;
            for i in 0..src_layout.fields.count() {
                let src_f = src_layout.field(bx.cx, i);
                assert_eq!(src_layout.fields.offset(i).bytes(), 0);
                assert_eq!(dst_layout.fields.offset(i).bytes(), 0);
                if src_f.is_zst() {
                    continue;
                }
                assert_eq!(src_layout.size, src_f.size);

                let dst_f = dst_layout.field(bx.cx, i);
                assert_ne!(src_f.ty, dst_f.ty);
                assert_eq!(result, None);
                result = Some(unsize_thin_ptr(bx, src, src_f.ty, dst_f.ty));
            }
            let (lldata, llextra) = result.unwrap();
            // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
            (bx.bitcast(lldata, dst_layout.scalar_pair_element_llvm_type(bx.cx, 0, true)),
             bx.bitcast(llextra, dst_layout.scalar_pair_element_llvm_type(bx.cx, 1, true)))
        }
        _ => bug!("unsize_thin_ptr: called on bad types"),
    }
}

/// Coerce `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty` and store the result in `dst`
pub fn coerce_unsized_into(
    bx: &Builder<'a, 'll, 'tcx>,
    src: PlaceRef<'ll, 'tcx>,
    dst: PlaceRef<'ll, 'tcx>
) {
    let src_ty = src.layout.ty;
    let dst_ty = dst.layout.ty;
    let coerce_ptr = || {
        let (base, info) = match src.load(bx).val {
            OperandValue::Pair(base, info) => {
                // fat-ptr to fat-ptr unsize preserves the vtable
                // i.e. &'a fmt::Debug+Send => &'a fmt::Debug
                // So we need to pointercast the base to ensure
                // the types match up.
                let thin_ptr = dst.layout.field(bx.cx, abi::FAT_PTR_ADDR);
                (bx.pointercast(base, thin_ptr.llvm_type(bx.cx)), info)
            }
            OperandValue::Immediate(base) => {
                unsize_thin_ptr(bx, base, src_ty, dst_ty)
            }
            OperandValue::Ref(..) => bug!()
        };
        OperandValue::Pair(base, info).store(bx, dst);
    };
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::Ref(..), &ty::Ref(..)) |
        (&ty::Ref(..), &ty::RawPtr(..)) |
        (&ty::RawPtr(..), &ty::RawPtr(..)) => {
            coerce_ptr()
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            coerce_ptr()
        }

        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b);

            for i in 0..def_a.variants[VariantIdx::new(0)].fields.len() {
                let src_f = src.project_field(bx, i);
                let dst_f = dst.project_field(bx, i);

                if dst_f.layout.is_zst() {
                    continue;
                }

                if src_f.layout.ty == dst_f.layout.ty {
                    memcpy_ty(bx, dst_f.llval, dst_f.align, src_f.llval, src_f.align,
                              src_f.layout, MemFlags::empty());
                } else {
                    coerce_unsized_into(bx, src_f, dst_f);
                }
            }
        }
        _ => bug!("coerce_unsized_into: invalid coercion {:?} -> {:?}",
                  src_ty,
                  dst_ty),
    }
}

pub fn cast_shift_expr_rhs(
    cx: &Builder<'_, 'll, '_>, op: hir::BinOpKind, lhs: &'ll Value, rhs: &'ll Value
) -> &'ll Value {
    cast_shift_rhs(op, lhs, rhs, |a, b| cx.trunc(a, b), |a, b| cx.zext(a, b))
}

fn cast_shift_rhs<'ll, F, G>(op: hir::BinOpKind,
                             lhs: &'ll Value,
                             rhs: &'ll Value,
                             trunc: F,
                             zext: G)
                             -> &'ll Value
    where F: FnOnce(&'ll Value, &'ll Type) -> &'ll Value,
          G: FnOnce(&'ll Value, &'ll Type) -> &'ll Value
{
    // Shifts may have any size int on the rhs
    if op.is_shift() {
        let mut rhs_llty = val_ty(rhs);
        let mut lhs_llty = val_ty(lhs);
        if rhs_llty.kind() == TypeKind::Vector {
            rhs_llty = rhs_llty.element_type()
        }
        if lhs_llty.kind() == TypeKind::Vector {
            lhs_llty = lhs_llty.element_type()
        }
        let rhs_sz = rhs_llty.int_width();
        let lhs_sz = lhs_llty.int_width();
        if lhs_sz < rhs_sz {
            trunc(rhs, lhs_llty)
        } else if lhs_sz > rhs_sz {
            // FIXME (#1877: If in the future shifting by negative
            // values is no longer undefined then this is wrong.
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

pub fn call_assume(bx: &Builder<'_, 'll, '_>, val: &'ll Value) {
    let assume_intrinsic = bx.cx.get_intrinsic("llvm.assume");
    bx.call(assume_intrinsic, &[val], None);
}

pub fn from_immediate(bx: &Builder<'_, 'll, '_>, val: &'ll Value) -> &'ll Value {
    if val_ty(val) == Type::i1(bx.cx) {
        bx.zext(val, Type::i8(bx.cx))
    } else {
        val
    }
}

pub fn to_immediate(
    bx: &Builder<'_, 'll, '_>,
    val: &'ll Value,
    layout: layout::TyLayout,
) -> &'ll Value {
    if let layout::Abi::Scalar(ref scalar) = layout.abi {
        return to_immediate_scalar(bx, val, scalar);
    }
    val
}

pub fn to_immediate_scalar(
    bx: &Builder<'_, 'll, '_>,
    val: &'ll Value,
    scalar: &layout::Scalar,
) -> &'ll Value {
    if scalar.is_bool() {
        return bx.trunc(val, Type::i1(bx.cx));
    }
    val
}

pub fn call_memcpy(
    bx: &Builder<'_, 'll, '_>,
    dst: &'ll Value,
    dst_align: Align,
    src: &'ll Value,
    src_align: Align,
    n_bytes: &'ll Value,
    flags: MemFlags,
) {
    if flags.contains(MemFlags::NONTEMPORAL) {
        // HACK(nox): This is inefficient but there is no nontemporal memcpy.
        let val = bx.load(src, src_align);
        let ptr = bx.pointercast(dst, val_ty(val).ptr_to());
        bx.store_with_flags(val, ptr, dst_align, flags);
        return;
    }
    let cx = bx.cx;
    let src_ptr = bx.pointercast(src, Type::i8p(cx));
    let dst_ptr = bx.pointercast(dst, Type::i8p(cx));
    let size = bx.intcast(n_bytes, cx.isize_ty, false);
    let volatile = flags.contains(MemFlags::VOLATILE);
    bx.memcpy(dst_ptr, dst_align.abi(), src_ptr, src_align.abi(), size, volatile);
}

pub fn memcpy_ty(
    bx: &Builder<'_, 'll, 'tcx>,
    dst: &'ll Value,
    dst_align: Align,
    src: &'ll Value,
    src_align: Align,
    layout: TyLayout<'tcx>,
    flags: MemFlags,
) {
    let size = layout.size.bytes();
    if size == 0 {
        return;
    }

    call_memcpy(bx, dst, dst_align, src, src_align, C_usize(bx.cx, size), flags);
}

pub fn call_memset(
    bx: &Builder<'_, 'll, '_>,
    ptr: &'ll Value,
    fill_byte: &'ll Value,
    size: &'ll Value,
    align: &'ll Value,
    volatile: bool,
) -> &'ll Value {
    let ptr_width = &bx.cx.sess().target.target.target_pointer_width;
    let intrinsic_key = format!("llvm.memset.p0i8.i{}", ptr_width);
    let llintrinsicfn = bx.cx.get_intrinsic(&intrinsic_key);
    let volatile = C_bool(bx.cx, volatile);
    bx.call(llintrinsicfn, &[ptr, fill_byte, size, align, volatile], None)
}

pub fn codegen_instance<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, instance: Instance<'tcx>) {
    let _s = if cx.sess().codegen_stats() {
        let mut instance_name = String::new();
        DefPathBasedNames::new(cx.tcx, true, true)
            .push_def_path(instance.def_id(), &mut instance_name);
        Some(StatRecorder::new(cx, instance_name))
    } else {
        None
    };

    // this is an info! to allow collecting monomorphization statistics
    // and to allow finding the last function before LLVM aborts from
    // release builds.
    info!("codegen_instance({})", instance);

    let sig = instance.fn_sig(cx.tcx);
    let sig = cx.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);

    let lldecl = cx.instances.borrow().get(&instance).cloned().unwrap_or_else(||
        bug!("Instance `{:?}` not already declared", instance));

    cx.stats.borrow_mut().n_closures += 1;

    let mir = cx.tcx.instance_mir(instance.def);
    mir::codegen_mir(cx, lldecl, &mir, instance, sig);
}

pub fn set_link_section(llval: &Value, attrs: &CodegenFnAttrs) {
    let sect = match attrs.link_section {
        Some(name) => name,
        None => return,
    };
    unsafe {
        let buf = SmallCStr::new(&sect.as_str());
        llvm::LLVMSetSection(llval, buf.as_ptr());
    }
}

/// Create the `main` function which will initialize the rust runtime and call
/// users main function.
fn maybe_create_entry_wrapper(cx: &CodegenCx) {
    let (main_def_id, span) = match *cx.sess().entry_fn.borrow() {
        Some((id, span, _)) => {
            (cx.tcx.hir.local_def_id(id), span)
        }
        None => return,
    };

    let instance = Instance::mono(cx.tcx, main_def_id);

    if !cx.codegen_unit.contains_item(&MonoItem::Fn(instance)) {
        // We want to create the wrapper in the same codegen unit as Rust's main
        // function.
        return;
    }

    let main_llfn = callee::get_fn(cx, instance);

    let et = cx.sess().entry_fn.get().map(|e| e.2);
    match et {
        Some(EntryFnType::Main) => create_entry_fn(cx, span, main_llfn, main_def_id, true),
        Some(EntryFnType::Start) => create_entry_fn(cx, span, main_llfn, main_def_id, false),
        None => {}    // Do nothing.
    }

    fn create_entry_fn(
        cx: &CodegenCx<'ll, '_>,
        sp: Span,
        rust_main: &'ll Value,
        rust_main_def_id: DefId,
        use_start_lang_item: bool,
    ) {
        let llfty = Type::func(&[Type::c_int(cx), Type::i8p(cx).ptr_to()], Type::c_int(cx));

        let main_ret_ty = cx.tcx.fn_sig(rust_main_def_id).output();
        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = cx.tcx.erase_regions(
            &main_ret_ty.no_bound_vars().unwrap(),
        );

        if declare::get_defined_value(cx, "main").is_some() {
            // FIXME: We should be smart and show a better diagnostic here.
            cx.sess().struct_span_err(sp, "entry symbol `main` defined multiple times")
                     .help("did you use #[no_mangle] on `fn main`? Use #[start] instead")
                     .emit();
            cx.sess().abort_if_errors();
            bug!();
        }
        let llfn = declare::declare_cfn(cx, "main", llfty);

        // `main` should respect same config for frame pointer elimination as rest of code
        attributes::set_frame_pointer_elimination(cx, llfn);
        attributes::apply_target_cpu_attr(cx, llfn);

        let bx = Builder::new_block(cx, llfn, "top");

        debuginfo::gdb::insert_reference_to_gdb_debug_scripts_section_global(&bx);

        // Params from native main() used as args for rust start function
        let param_argc = get_param(llfn, 0);
        let param_argv = get_param(llfn, 1);
        let arg_argc = bx.intcast(param_argc, cx.isize_ty, true);
        let arg_argv = param_argv;

        let (start_fn, args) = if use_start_lang_item {
            let start_def_id = cx.tcx.require_lang_item(StartFnLangItem);
            let start_fn = callee::resolve_and_get_fn(
                cx,
                start_def_id,
                cx.tcx.intern_substs(&[main_ret_ty.into()]),
            );
            (start_fn, vec![bx.pointercast(rust_main, Type::i8p(cx).ptr_to()),
                            arg_argc, arg_argv])
        } else {
            debug!("using user-defined start fn");
            (rust_main, vec![arg_argc, arg_argv])
        };

        let result = bx.call(start_fn, &args, None);
        bx.ret(bx.intcast(result, Type::c_int(cx), true));
    }
}

fn write_metadata<'a, 'gcx>(tcx: TyCtxt<'a, 'gcx, 'gcx>,
                            llvm_module: &ModuleLlvm)
                            -> EncodedMetadata {
    use std::io::Write;
    use flate2::Compression;
    use flate2::write::DeflateEncoder;

    let (metadata_llcx, metadata_llmod) = (&*llvm_module.llcx, llvm_module.llmod());

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum MetadataKind {
        None,
        Uncompressed,
        Compressed
    }

    let kind = tcx.sess.crate_types.borrow().iter().map(|ty| {
        match *ty {
            config::CrateType::Executable |
            config::CrateType::Staticlib |
            config::CrateType::Cdylib => MetadataKind::None,

            config::CrateType::Rlib => MetadataKind::Uncompressed,

            config::CrateType::Dylib |
            config::CrateType::ProcMacro => MetadataKind::Compressed,
        }
    }).max().unwrap_or(MetadataKind::None);

    if kind == MetadataKind::None {
        return EncodedMetadata::new();
    }

    let metadata = tcx.encode_metadata();
    if kind == MetadataKind::Uncompressed {
        return metadata;
    }

    assert!(kind == MetadataKind::Compressed);
    let mut compressed = tcx.metadata_encoding_version();
    DeflateEncoder::new(&mut compressed, Compression::fast())
        .write_all(&metadata.raw_data).unwrap();

    let llmeta = C_bytes_in_context(metadata_llcx, &compressed);
    let llconst = C_struct_in_context(metadata_llcx, &[llmeta], false);
    let name = exported_symbols::metadata_symbol_name(tcx);
    let buf = CString::new(name).unwrap();
    let llglobal = unsafe {
        llvm::LLVMAddGlobal(metadata_llmod, val_ty(llconst), buf.as_ptr())
    };
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        let section_name = metadata::metadata_section_name(&tcx.sess.target.target);
        let name = SmallCStr::new(section_name);
        llvm::LLVMSetSection(llglobal, name.as_ptr());

        // Also generate a .section directive to force no
        // flags, at least for ELF outputs, so that the
        // metadata doesn't get loaded into memory.
        let directive = format!(".section {}", section_name);
        let directive = CString::new(directive).unwrap();
        llvm::LLVMSetModuleInlineAsm(metadata_llmod, directive.as_ptr())
    }
    return metadata;
}

pub struct ValueIter<'ll> {
    cur: Option<&'ll Value>,
    step: unsafe extern "C" fn(&'ll Value) -> Option<&'ll Value>,
}

impl Iterator for ValueIter<'ll> {
    type Item = &'ll Value;

    fn next(&mut self) -> Option<&'ll Value> {
        let old = self.cur;
        if let Some(old) = old {
            self.cur = unsafe { (self.step)(old) };
        }
        old
    }
}

pub fn iter_globals(llmod: &'ll llvm::Module) -> ValueIter<'ll> {
    unsafe {
        ValueIter {
            cur: llvm::LLVMGetFirstGlobal(llmod),
            step: llvm::LLVMGetNextGlobal,
        }
    }
}

fn determine_cgu_reuse<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 cgu: &CodegenUnit<'tcx>)
                                 -> CguReuse {
    if !tcx.dep_graph.is_fully_enabled() {
        return CguReuse::No
    }

    let work_product_id = &cgu.work_product_id();
    if tcx.dep_graph.previous_work_product(work_product_id).is_none() {
        // We don't have anything cached for this CGU. This can happen
        // if the CGU did not exist in the previous session.
        return CguReuse::No
    }

    // Try to mark the CGU as green. If it we can do so, it means that nothing
    // affecting the LLVM module has changed and we can re-use a cached version.
    // If we compile with any kind of LTO, this means we can re-use the bitcode
    // of the Pre-LTO stage (possibly also the Post-LTO version but we'll only
    // know that later). If we are not doing LTO, there is only one optimized
    // version of each module, so we re-use that.
    let dep_node = cgu.codegen_dep_node(tcx);
    assert!(!tcx.dep_graph.dep_node_exists(&dep_node),
        "CompileCodegenUnit dep-node for CGU `{}` already exists before marking.",
        cgu.name());

    if tcx.dep_graph.try_mark_green(tcx, &dep_node).is_some() {
        // We can re-use either the pre- or the post-thinlto state
        if tcx.sess.lto() != Lto::No {
            CguReuse::PreLto
        } else {
            CguReuse::PostLto
        }
    } else {
        CguReuse::No
    }
}

pub fn codegen_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               rx: mpsc::Receiver<Box<dyn Any + Send>>)
                               -> OngoingCodegen
{
    check_for_rustc_errors_attr(tcx);

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);

    // Codegen the metadata.
    tcx.sess.profiler(|p| p.start_activity(ProfileCategory::Codegen));

    let metadata_cgu_name = cgu_name_builder.build_cgu_name(LOCAL_CRATE,
                                                            &["crate"],
                                                            Some("metadata")).as_str()
                                                                             .to_string();
    let metadata_llvm_module = ModuleLlvm::new(tcx.sess, &metadata_cgu_name);
    let metadata = time(tcx.sess, "write metadata", || {
        write_metadata(tcx, &metadata_llvm_module)
    });
    tcx.sess.profiler(|p| p.end_activity(ProfileCategory::Codegen));

    let metadata_module = ModuleCodegen {
        name: metadata_cgu_name,
        module_llvm: metadata_llvm_module,
        kind: ModuleKind::Metadata,
    };

    let time_graph = if tcx.sess.opts.debugging_opts.codegen_time_graph {
        Some(time_graph::TimeGraph::new())
    } else {
        None
    };

    // Skip crate items and just output metadata in -Z no-codegen mode.
    if tcx.sess.opts.debugging_opts.no_codegen ||
       !tcx.sess.opts.output_types.should_codegen() {
        let ongoing_codegen = write::start_async_codegen(
            tcx,
            time_graph,
            metadata,
            rx,
            1);

        ongoing_codegen.submit_pre_codegened_module_to_llvm(tcx, metadata_module);
        ongoing_codegen.codegen_finished(tcx);

        assert_and_save_dep_graph(tcx);

        ongoing_codegen.check_for_errors(tcx.sess);

        return ongoing_codegen;
    }

    // Run the monomorphization collector and partition the collected items into
    // codegen units.
    let codegen_units = tcx.collect_and_partition_mono_items(LOCAL_CRATE).1;
    let codegen_units = (*codegen_units).clone();

    // Force all codegen_unit queries so they are already either red or green
    // when compile_codegen_unit accesses them. We are not able to re-execute
    // the codegen_unit query from just the DepNode, so an unknown color would
    // lead to having to re-execute compile_codegen_unit, possibly
    // unnecessarily.
    if tcx.dep_graph.is_fully_enabled() {
        for cgu in &codegen_units {
            tcx.codegen_unit(cgu.name().clone());
        }
    }

    let ongoing_codegen = write::start_async_codegen(
        tcx,
        time_graph.clone(),
        metadata,
        rx,
        codegen_units.len());
    let ongoing_codegen = AbortCodegenOnDrop(Some(ongoing_codegen));

    // Codegen an allocator shim, if necessary.
    //
    // If the crate doesn't have an `allocator_kind` set then there's definitely
    // no shim to generate. Otherwise we also check our dependency graph for all
    // our output crate types. If anything there looks like its a `Dynamic`
    // linkage, then it's already got an allocator shim and we'll be using that
    // one instead. If nothing exists then it's our job to generate the
    // allocator!
    let any_dynamic_crate = tcx.sess.dependency_formats.borrow()
        .iter()
        .any(|(_, list)| {
            use rustc::middle::dependency_format::Linkage;
            list.iter().any(|&linkage| linkage == Linkage::Dynamic)
        });
    let allocator_module = if any_dynamic_crate {
        None
    } else if let Some(kind) = *tcx.sess.allocator_kind.get() {
        let llmod_id = cgu_name_builder.build_cgu_name(LOCAL_CRATE,
                                                       &["crate"],
                                                       Some("allocator")).as_str()
                                                                         .to_string();
        let modules = ModuleLlvm::new(tcx.sess, &llmod_id);
        time(tcx.sess, "write allocator module", || {
            unsafe {
                allocator::codegen(tcx, &modules, kind)
            }
        });

        Some(ModuleCodegen {
            name: llmod_id,
            module_llvm: modules,
            kind: ModuleKind::Allocator,
        })
    } else {
        None
    };

    if let Some(allocator_module) = allocator_module {
        ongoing_codegen.submit_pre_codegened_module_to_llvm(tcx, allocator_module);
    }

    ongoing_codegen.submit_pre_codegened_module_to_llvm(tcx, metadata_module);

    // We sort the codegen units by size. This way we can schedule work for LLVM
    // a bit more efficiently.
    let codegen_units = {
        let mut codegen_units = codegen_units;
        codegen_units.sort_by_cached_key(|cgu| cmp::Reverse(cgu.size_estimate()));
        codegen_units
    };

    let mut total_codegen_time = Duration::new(0, 0);
    let mut all_stats = Stats::default();

    for cgu in codegen_units.into_iter() {
        ongoing_codegen.wait_for_signal_to_codegen_item();
        ongoing_codegen.check_for_errors(tcx.sess);

        let cgu_reuse = determine_cgu_reuse(tcx, &cgu);
        tcx.sess.cgu_reuse_tracker.set_actual_reuse(&cgu.name().as_str(), cgu_reuse);

        match cgu_reuse {
            CguReuse::No => {
                let _timing_guard = time_graph.as_ref().map(|time_graph| {
                    time_graph.start(write::CODEGEN_WORKER_TIMELINE,
                                     write::CODEGEN_WORK_PACKAGE_KIND,
                                     &format!("codegen {}", cgu.name()))
                });
                let start_time = Instant::now();
                let stats = compile_codegen_unit(tcx, *cgu.name());
                all_stats.extend(stats);
                total_codegen_time += start_time.elapsed();
                false
            }
            CguReuse::PreLto => {
                write::submit_pre_lto_module_to_llvm(tcx, CachedModuleCodegen {
                    name: cgu.name().to_string(),
                    source: cgu.work_product(tcx),
                });
                true
            }
            CguReuse::PostLto => {
                write::submit_post_lto_module_to_llvm(tcx, CachedModuleCodegen {
                    name: cgu.name().to_string(),
                    source: cgu.work_product(tcx),
                });
                true
            }
        };
    }

    ongoing_codegen.codegen_finished(tcx);

    // Since the main thread is sometimes blocked during codegen, we keep track
    // -Ztime-passes output manually.
    print_time_passes_entry(tcx.sess.time_passes(),
                            "codegen to LLVM IR",
                            total_codegen_time);

    rustc_incremental::assert_module_sources::assert_module_sources(tcx);

    symbol_names_test::report_symbol_names(tcx);

    if tcx.sess.codegen_stats() {
        println!("--- codegen stats ---");
        println!("n_glues_created: {}", all_stats.n_glues_created);
        println!("n_null_glues: {}", all_stats.n_null_glues);
        println!("n_real_glues: {}", all_stats.n_real_glues);

        println!("n_fns: {}", all_stats.n_fns);
        println!("n_inlines: {}", all_stats.n_inlines);
        println!("n_closures: {}", all_stats.n_closures);
        println!("fn stats:");
        all_stats.fn_stats.sort_by_key(|&(_, insns)| insns);
        for &(ref name, insns) in all_stats.fn_stats.iter() {
            println!("{} insns, {}", insns, *name);
        }
    }

    if tcx.sess.count_llvm_insns() {
        for (k, v) in all_stats.llvm_insns.iter() {
            println!("{:7} {}", *v, *k);
        }
    }

    ongoing_codegen.check_for_errors(tcx.sess);

    assert_and_save_dep_graph(tcx);
    ongoing_codegen.into_inner()
}

/// A curious wrapper structure whose only purpose is to call `codegen_aborted`
/// when it's dropped abnormally.
///
/// In the process of working on rust-lang/rust#55238 a mysterious segfault was
/// stumbled upon. The segfault was never reproduced locally, but it was
/// suspected to be related to the fact that codegen worker threads were
/// sticking around by the time the main thread was exiting, causing issues.
///
/// This structure is an attempt to fix that issue where the `codegen_aborted`
/// message will block until all workers have finished. This should ensure that
/// even if the main codegen thread panics we'll wait for pending work to
/// complete before returning from the main thread, hopefully avoiding
/// segfaults.
///
/// If you see this comment in the code, then it means that this workaround
/// worked! We may yet one day track down the mysterious cause of that
/// segfault...
struct AbortCodegenOnDrop(Option<OngoingCodegen>);

impl AbortCodegenOnDrop {
    fn into_inner(mut self) -> OngoingCodegen {
        self.0.take().unwrap()
    }
}

impl Deref for AbortCodegenOnDrop {
    type Target = OngoingCodegen;

    fn deref(&self) -> &OngoingCodegen {
        self.0.as_ref().unwrap()
    }
}

impl DerefMut for AbortCodegenOnDrop {
    fn deref_mut(&mut self) -> &mut OngoingCodegen {
        self.0.as_mut().unwrap()
    }
}

impl Drop for AbortCodegenOnDrop {
    fn drop(&mut self) {
        if let Some(codegen) = self.0.take() {
            codegen.codegen_aborted();
        }
    }
}

fn assert_and_save_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    time(tcx.sess,
         "assert dep graph",
         || rustc_incremental::assert_dep_graph(tcx));

    time(tcx.sess,
         "serialize dep graph",
         || rustc_incremental::save_dep_graph(tcx));
}

impl CrateInfo {
    pub fn new(tcx: TyCtxt) -> CrateInfo {
        let mut info = CrateInfo {
            panic_runtime: None,
            compiler_builtins: None,
            profiler_runtime: None,
            sanitizer_runtime: None,
            is_no_builtins: Default::default(),
            native_libraries: Default::default(),
            used_libraries: tcx.native_libraries(LOCAL_CRATE),
            link_args: tcx.link_args(LOCAL_CRATE),
            crate_name: Default::default(),
            used_crates_dynamic: cstore::used_crates(tcx, LinkagePreference::RequireDynamic),
            used_crates_static: cstore::used_crates(tcx, LinkagePreference::RequireStatic),
            used_crate_source: Default::default(),
            wasm_imports: Default::default(),
            lang_item_to_crate: Default::default(),
            missing_lang_items: Default::default(),
        };
        let lang_items = tcx.lang_items();

        let load_wasm_items = tcx.sess.crate_types.borrow()
            .iter()
            .any(|c| *c != config::CrateType::Rlib) &&
            tcx.sess.opts.target_triple.triple() == "wasm32-unknown-unknown";

        if load_wasm_items {
            info.load_wasm_imports(tcx, LOCAL_CRATE);
        }

        let crates = tcx.crates();

        let n_crates = crates.len();
        info.native_libraries.reserve(n_crates);
        info.crate_name.reserve(n_crates);
        info.used_crate_source.reserve(n_crates);
        info.missing_lang_items.reserve(n_crates);

        for &cnum in crates.iter() {
            info.native_libraries.insert(cnum, tcx.native_libraries(cnum));
            info.crate_name.insert(cnum, tcx.crate_name(cnum).to_string());
            info.used_crate_source.insert(cnum, tcx.used_crate_source(cnum));
            if tcx.is_panic_runtime(cnum) {
                info.panic_runtime = Some(cnum);
            }
            if tcx.is_compiler_builtins(cnum) {
                info.compiler_builtins = Some(cnum);
            }
            if tcx.is_profiler_runtime(cnum) {
                info.profiler_runtime = Some(cnum);
            }
            if tcx.is_sanitizer_runtime(cnum) {
                info.sanitizer_runtime = Some(cnum);
            }
            if tcx.is_no_builtins(cnum) {
                info.is_no_builtins.insert(cnum);
            }
            if load_wasm_items {
                info.load_wasm_imports(tcx, cnum);
            }
            let missing = tcx.missing_lang_items(cnum);
            for &item in missing.iter() {
                if let Ok(id) = lang_items.require(item) {
                    info.lang_item_to_crate.insert(item, id.krate);
                }
            }

            // No need to look for lang items that are whitelisted and don't
            // actually need to exist.
            let missing = missing.iter()
                .cloned()
                .filter(|&l| !weak_lang_items::whitelisted(tcx, l))
                .collect();
            info.missing_lang_items.insert(cnum, missing);
        }

        return info
    }

    fn load_wasm_imports(&mut self, tcx: TyCtxt, cnum: CrateNum) {
        self.wasm_imports.extend(tcx.wasm_import_module_map(cnum).iter().map(|(&id, module)| {
            let instance = Instance::mono(tcx, id);
            let import_name = tcx.symbol_name(instance);

            (import_name.to_string(), module.clone())
        }));
    }
}

fn compile_codegen_unit<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  cgu_name: InternedString)
                                  -> Stats {
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let ((stats, module), _) = tcx.dep_graph.with_task(dep_node,
                                                       tcx,
                                                       cgu_name,
                                                       module_codegen);
    let time_to_codegen = start_time.elapsed();

    // We assume that the cost to run LLVM on a CGU is proportional to
    // the time we needed for codegenning it.
    let cost = time_to_codegen.as_secs() * 1_000_000_000 +
               time_to_codegen.subsec_nanos() as u64;

    write::submit_codegened_module_to_llvm(tcx,
                                           module,
                                           cost);
    return stats;

    fn module_codegen<'a, 'tcx>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        cgu_name: InternedString)
        -> (Stats, ModuleCodegen)
    {
        let cgu = tcx.codegen_unit(cgu_name);

        // Instantiate monomorphizations without filling out definitions yet...
        let llvm_module = ModuleLlvm::new(tcx.sess, &cgu_name.as_str());
        let stats = {
            let cx = CodegenCx::new(tcx, cgu, &llvm_module);
            let mono_items = cx.codegen_unit
                               .items_in_deterministic_order(cx.tcx);
            for &(mono_item, (linkage, visibility)) in &mono_items {
                mono_item.predefine(&cx, linkage, visibility);
            }

            // ... and now that we have everything pre-defined, fill out those definitions.
            for &(mono_item, _) in &mono_items {
                mono_item.define(&cx);
            }

            // If this codegen unit contains the main function, also create the
            // wrapper here
            maybe_create_entry_wrapper(&cx);

            // Run replace-all-uses-with for statics that need it
            for &(old_g, new_g) in cx.statics_to_rauw.borrow().iter() {
                unsafe {
                    let bitcast = llvm::LLVMConstPointerCast(new_g, val_ty(old_g));
                    llvm::LLVMReplaceAllUsesWith(old_g, bitcast);
                    llvm::LLVMDeleteGlobal(old_g);
                }
            }

            // Create the llvm.used variable
            // This variable has type [N x i8*] and is stored in the llvm.metadata section
            if !cx.used_statics.borrow().is_empty() {
                let name = const_cstr!("llvm.used");
                let section = const_cstr!("llvm.metadata");
                let array = C_array(Type::i8(&cx).ptr_to(), &*cx.used_statics.borrow());

                unsafe {
                    let g = llvm::LLVMAddGlobal(cx.llmod,
                                                val_ty(array),
                                                name.as_ptr());
                    llvm::LLVMSetInitializer(g, array);
                    llvm::LLVMRustSetLinkage(g, llvm::Linkage::AppendingLinkage);
                    llvm::LLVMSetSection(g, section.as_ptr());
                }
            }

            // Finalize debuginfo
            if cx.sess().opts.debuginfo != DebugInfo::None {
                debuginfo::finalize(&cx);
            }

            cx.stats.into_inner()
        };

        (stats, ModuleCodegen {
            name: cgu_name.to_string(),
            module_llvm: llvm_module,
            kind: ModuleKind::Regular,
        })
    }
}

pub fn provide_both(providers: &mut Providers) {
    providers.dllimport_foreign_items = |tcx, krate| {
        let module_map = tcx.foreign_modules(krate);
        let module_map = module_map.iter()
            .map(|lib| (lib.def_id, lib))
            .collect::<FxHashMap<_, _>>();

        let dllimports = tcx.native_libraries(krate)
            .iter()
            .filter(|lib| {
                if lib.kind != cstore::NativeLibraryKind::NativeUnknown {
                    return false
                }
                let cfg = match lib.cfg {
                    Some(ref cfg) => cfg,
                    None => return true,
                };
                attr::cfg_matches(cfg, &tcx.sess.parse_sess, None)
            })
            .filter_map(|lib| lib.foreign_module)
            .map(|id| &module_map[&id])
            .flat_map(|module| module.foreign_items.iter().cloned())
            .collect();
        Lrc::new(dllimports)
    };

    providers.is_dllimport_foreign_item = |tcx, def_id| {
        tcx.dllimport_foreign_items(def_id.krate).contains(&def_id)
    };
}

pub fn linkage_to_llvm(linkage: Linkage) -> llvm::Linkage {
    match linkage {
        Linkage::External => llvm::Linkage::ExternalLinkage,
        Linkage::AvailableExternally => llvm::Linkage::AvailableExternallyLinkage,
        Linkage::LinkOnceAny => llvm::Linkage::LinkOnceAnyLinkage,
        Linkage::LinkOnceODR => llvm::Linkage::LinkOnceODRLinkage,
        Linkage::WeakAny => llvm::Linkage::WeakAnyLinkage,
        Linkage::WeakODR => llvm::Linkage::WeakODRLinkage,
        Linkage::Appending => llvm::Linkage::AppendingLinkage,
        Linkage::Internal => llvm::Linkage::InternalLinkage,
        Linkage::Private => llvm::Linkage::PrivateLinkage,
        Linkage::ExternalWeak => llvm::Linkage::ExternalWeakLinkage,
        Linkage::Common => llvm::Linkage::CommonLinkage,
    }
}

pub fn visibility_to_llvm(linkage: Visibility) -> llvm::Visibility {
    match linkage {
        Visibility::Default => llvm::Visibility::Default,
        Visibility::Hidden => llvm::Visibility::Hidden,
        Visibility::Protected => llvm::Visibility::Protected,
    }
}

// FIXME(mw): Anything that is produced via DepGraph::with_task() must implement
//            the HashStable trait. Normally DepGraph::with_task() calls are
//            hidden behind queries, but CGU creation is a special case in two
//            ways: (1) it's not a query and (2) CGU are output nodes, so their
//            Fingerprints are not actually needed. It remains to be clarified
//            how exactly this case will be handled in the red/green system but
//            for now we content ourselves with providing a no-op HashStable
//            implementation for CGUs.
mod temp_stable_hash_impls {
    use rustc_data_structures::stable_hasher::{StableHasherResult, StableHasher,
                                               HashStable};
    use ModuleCodegen;

    impl<HCX> HashStable<HCX> for ModuleCodegen {
        fn hash_stable<W: StableHasherResult>(&self,
                                              _: &mut HCX,
                                              _: &mut StableHasher<W>) {
            // do nothing
        }
    }
}

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

use {ModuleCodegen, ModuleKind, CachedModuleCodegen};

use rustc::dep_graph::cgu_reuse_tracker::CguReuse;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::middle::lang_items::StartFnLangItem;
use rustc::middle::weak_lang_items;
use rustc::mir::mono::{Linkage, Stats, CodegenUnitNameBuilder};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::{self, Align, TyLayout, LayoutOf, HasTyCtxt};
use rustc::ty::query::Providers;
use rustc::middle::cstore::{self, LinkagePreference};
use rustc::util::common::{time, print_time_passes_entry};
use rustc::util::profiling::ProfileCategory;
use rustc::session::config::{self, EntryFnType, Lto};
use rustc::session::Session;
use mir::place::PlaceRef;
use {MemFlags, CrateInfo};
use callee;
use rustc_mir::monomorphize::collector::{self, MonoItemCollectionMode};
use rustc_mir::monomorphize::item::DefPathBasedNames;
use common::{self, RealPredicate, TypeKind, IntPredicate};
use meth;
use mir;
use rustc::util::time_graph;
use rustc_mir::monomorphize::Instance;
use rustc_mir::monomorphize::partitioning::{self, PartitioningStrategy,
    CodegenUnit, CodegenUnitExt};
use mono_item::{MonoItem, BaseMonoItemExt};
use rustc::util::nodemap::{FxHashMap, FxHashSet, DefIdSet};
use rustc_data_structures::sync::Lrc;
use rustc_codegen_utils::{symbol_names_test, check_for_rustc_errors_attr};
use rustc::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};

use interfaces::*;

use std::any::Any;
use std::sync::Arc;
use std::time::{Instant, Duration};
use std::cmp;
use std::sync::mpsc;
use syntax_pos::Span;
use syntax::attr;
use rustc::hir;

use mir::operand::OperandValue;

use std::marker::PhantomData;


pub struct StatRecorder<'a, 'll: 'a, 'tcx: 'll, Cx: 'a + CodegenMethods<'ll, 'tcx>> {
    cx: &'a Cx,
    name: Option<String>,
    istart: usize,
    phantom: PhantomData<(&'ll (), &'tcx ())>
}

impl<'a, 'll: 'a, 'tcx: 'll, Cx: 'a + CodegenMethods<'ll, 'tcx>> StatRecorder<'a, 'll, 'tcx, Cx> {
    pub fn new(cx: &'a Cx, name: String) -> Self {
        let istart = cx.stats().borrow().n_llvm_insns;
        StatRecorder {
            cx,
            name: Some(name),
            istart,
            phantom: PhantomData
        }
    }
}

impl<'a, 'll: 'a, 'tcx: 'll, Cx: 'a + CodegenMethods<'ll, 'tcx>> Drop for
    StatRecorder<'a, 'll, 'tcx, Cx>
{
    fn drop(&mut self) {
        if self.cx.sess().codegen_stats() {
            let mut stats = self.cx.stats().borrow_mut();
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
                                -> IntPredicate {
    match op {
        hir::BinOpKind::Eq => IntPredicate::IntEQ,
        hir::BinOpKind::Ne => IntPredicate::IntNE,
        hir::BinOpKind::Lt => if signed { IntPredicate::IntSLT } else { IntPredicate::IntULT },
        hir::BinOpKind::Le => if signed { IntPredicate::IntSLE } else { IntPredicate::IntULE },
        hir::BinOpKind::Gt => if signed { IntPredicate::IntSGT } else { IntPredicate::IntUGT },
        hir::BinOpKind::Ge => if signed { IntPredicate::IntSGE } else { IntPredicate::IntUGE },
        op => {
            bug!("comparison_op_to_icmp_predicate: expected comparison operator, \
                  found {:?}",
                 op)
        }
    }
}

pub fn bin_op_to_fcmp_predicate(op: hir::BinOpKind) -> RealPredicate {
    match op {
        hir::BinOpKind::Eq => RealPredicate::RealOEQ,
        hir::BinOpKind::Ne => RealPredicate::RealUNE,
        hir::BinOpKind::Lt => RealPredicate::RealOLT,
        hir::BinOpKind::Le => RealPredicate::RealOLE,
        hir::BinOpKind::Gt => RealPredicate::RealOGT,
        hir::BinOpKind::Ge => RealPredicate::RealOGE,
        op => {
            bug!("comparison_op_to_fcmp_predicate: expected comparison operator, \
                  found {:?}",
                 op);
        }
    }
}

pub fn compare_simd_types<'a, 'll:'a, 'tcx:'ll, Bx : BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    lhs: <Bx::CodegenCx as Backend<'ll>>::Value,
    rhs: <Bx::CodegenCx as Backend<'ll>>::Value,
    t: Ty<'tcx>,
    ret_ty: <Bx::CodegenCx as Backend<'ll>>::Type,
    op: hir::BinOpKind
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    let signed = match t.sty {
        ty::Float(_) => {
            let cmp = bin_op_to_fcmp_predicate(op);
            let cmp = bx.fcmp(cmp, lhs, rhs);
            return bx.sext(cmp, ret_ty);
        },
        ty::Uint(_) => false,
        ty::Int(_) => true,
        _ => bug!("compare_simd_types: invalid SIMD type"),
    };

    let cmp = bin_op_to_icmp_predicate(op, signed);
    let cmp = bx.icmp(cmp, lhs, rhs);
    // LLVM outputs an `< size x i1 >`, so we need to perform a sign extension
    // to get the correctly sized type. This will compile to a single instruction
    // once the IR is converted to assembly if the SIMD instruction is supported
    // by the target architecture.
    bx.sext(cmp, ret_ty)
}

/// Retrieve the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit funny. It is intended for use
/// in an upcast, where the new vtable for an object will be derived
/// from the old one.
pub fn unsized_info<'a, 'll: 'a, 'tcx: 'll, Cx: 'a + CodegenMethods<'ll, 'tcx>>(
    cx: &'a Cx,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
    old_info: Option<Cx::Value>,
) -> Cx::Value where &'a Cx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx> {
    let (source, target) = cx.tcx().struct_lockstep_tails(source, target);
    match (&source.sty, &target.sty) {
        (&ty::Array(_, len), &ty::Slice(_)) => {
            cx.const_usize(len.unwrap_usize(*cx.tcx()))
        }
        (&ty::Dynamic(..), &ty::Dynamic(..)) => {
            // For now, upcasts are limited to changes in marker
            // traits, and hence never actually require an actual
            // change to the vtable.
            old_info.expect("unsized_info: missing old info for trait upcast")
        }
        (_, &ty::Dynamic(ref data, ..)) => {
            let vtable_ptr = cx.layout_of(cx.tcx().mk_mut_ptr(target))
                .field(cx, FAT_PTR_EXTRA);
            cx.static_ptrcast(meth::get_vtable(cx, source, data.principal()),
                            cx.backend_type(&vtable_ptr))
        }
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}",
                                     source,
                                     target),
    }
}

/// Coerce `src` to `dst_ty`. `src_ty` must be a thin pointer.
pub fn unsize_thin_ptr<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    src: <Bx::CodegenCx as Backend<'ll>>::Value,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>
) -> (<Bx::CodegenCx as Backend<'ll>>::Value, <Bx::CodegenCx as Backend<'ll>>::Value) where
    &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
{
    debug!("unsize_thin_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::Ref(_, a, _),
         &ty::Ref(_, b, _)) |
        (&ty::Ref(_, a, _),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
            assert!(bx.cx().type_is_sized(a));
            let ptr_ty = bx.cx().type_ptr_to(bx.cx().backend_type(&bx.cx().layout_of(b)));
            (bx.pointercast(src, ptr_ty), unsized_info(bx.cx(), a, b, None))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            let (a, b) = (src_ty.boxed_ty(), dst_ty.boxed_ty());
            assert!(bx.cx().type_is_sized(a));
            let ptr_ty = bx.cx().type_ptr_to(bx.cx().backend_type(&bx.cx().layout_of(b)));
            (bx.pointercast(src, ptr_ty), unsized_info(bx.cx(), a, b, None))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b);

            let src_layout = bx.cx().layout_of(src_ty);
            let dst_layout = bx.cx().layout_of(dst_ty);
            let mut result = None;
            for i in 0..src_layout.fields.count() {
                let src_f = src_layout.field(bx.cx(), i);
                assert_eq!(src_layout.fields.offset(i).bytes(), 0);
                assert_eq!(dst_layout.fields.offset(i).bytes(), 0);
                if src_f.is_zst() {
                    continue;
                }
                assert_eq!(src_layout.size, src_f.size);

                let dst_f = dst_layout.field(bx.cx(), i);
                assert_ne!(src_f.ty, dst_f.ty);
                assert_eq!(result, None);
                result = Some(unsize_thin_ptr(bx, src, src_f.ty, dst_f.ty));
            }
            let (lldata, llextra) = result.unwrap();
            // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
            (bx.bitcast(lldata, bx.cx().scalar_pair_element_backend_type(&dst_layout, 0, true)),
             bx.bitcast(llextra, bx.cx().scalar_pair_element_backend_type(&dst_layout, 1, true)))
        }
        _ => bug!("unsize_thin_ptr: called on bad types"),
    }
}

/// Coerce `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty` and store the result in `dst`
pub fn coerce_unsized_into<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    src: PlaceRef<'tcx, <Bx::CodegenCx as Backend<'ll>>::Value>,
    dst: PlaceRef<'tcx, <Bx::CodegenCx as Backend<'ll>>::Value>
) where &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
{
    let src_ty = src.layout.ty;
    let dst_ty = dst.layout.ty;
    let mut coerce_ptr = || {
        let (base, info) = match bx.load_ref(&src).val {
            OperandValue::Pair(base, info) => {
                // fat-ptr to fat-ptr unsize preserves the vtable
                // i.e. &'a fmt::Debug+Send => &'a fmt::Debug
                // So we need to pointercast the base to ensure
                // the types match up.
                let thin_ptr = dst.layout.field(bx.cx(), FAT_PTR_ADDR);
                (bx.pointercast(base, bx.cx().backend_type(&thin_ptr)), info)
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

            for i in 0..def_a.variants[0].fields.len() {
                let src_f = src.project_field(bx, i);
                let dst_f = dst.project_field(bx, i);

                if dst_f.layout.is_zst() {
                    continue;
                }

                if src_f.layout.ty == dst_f.layout.ty {
                    memcpy_ty(bx, dst_f.llval, src_f.llval, src_f.layout,
                              src_f.align.min(dst_f.align), MemFlags::empty());
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

pub fn cast_shift_expr_rhs<'a, 'll: 'a, 'tcx: 'll, Bx : BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    op: hir::BinOpKind,
    lhs: <Bx::CodegenCx as Backend<'ll>>::Value,
    rhs: <Bx::CodegenCx as Backend<'ll>>::Value
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    cast_shift_rhs(bx, op, lhs, rhs)
}

fn cast_shift_rhs<'a, 'll :'a, 'tcx : 'll, Bx : BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    op: hir::BinOpKind,
    lhs: <Bx::CodegenCx as Backend<'ll>>::Value,
    rhs: <Bx::CodegenCx as Backend<'ll>>::Value,
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    // Shifts may have any size int on the rhs
    if op.is_shift() {
        let mut rhs_llty = bx.cx().val_ty(rhs);
        let mut lhs_llty = bx.cx().val_ty(lhs);
        if bx.cx().type_kind(rhs_llty) == TypeKind::Vector {
            rhs_llty = bx.cx().element_type(rhs_llty)
        }
        if bx.cx().type_kind(lhs_llty) == TypeKind::Vector {
            lhs_llty = bx.cx().element_type(lhs_llty)
        }
        let rhs_sz = bx.cx().int_width(rhs_llty);
        let lhs_sz = bx.cx().int_width(lhs_llty);
        if lhs_sz < rhs_sz {
            bx.trunc(rhs, lhs_llty)
        } else if lhs_sz > rhs_sz {
            // FIXME (#1877: If shifting by negative
            // values becomes not undefined then this is wrong.
            bx.zext(rhs, lhs_llty)
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

pub fn call_assume<'a, 'll: 'a, 'tcx: 'll, Bx : BuilderMethods<'a, 'll ,'tcx>>(
    bx: &mut Bx,
    val: <Bx::CodegenCx as Backend<'ll>>::Value
) {
    let assume_intrinsic = bx.cx().get_intrinsic("llvm.assume");
    bx.call(assume_intrinsic, &[val], None);
}

pub fn from_immediate<'a, 'll: 'a, 'tcx: 'll, Bx : BuilderMethods<'a, 'll ,'tcx>>(
    bx: &mut Bx,
    val: <Bx::CodegenCx as Backend<'ll>>::Value
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    if bx.cx().val_ty(val) == bx.cx().type_i1() {
        bx.zext(val, bx.cx().type_i8())
    } else {
        val
    }
}

pub fn to_immediate<'a, 'll: 'a, 'tcx: 'll, Bx : BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    val: <Bx::CodegenCx as Backend<'ll>>::Value,
    layout: layout::TyLayout,
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    if let layout::Abi::Scalar(ref scalar) = layout.abi {
        return to_immediate_scalar(bx, val, scalar);
    }
    val
}

pub fn to_immediate_scalar<'a, 'll :'a, 'tcx :'ll, Bx : BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    val: <Bx::CodegenCx as Backend<'ll>>::Value,
    scalar: &layout::Scalar,
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    if scalar.is_bool() {
        return bx.trunc(val, bx.cx().type_i1());
    }
    val
}

pub fn memcpy_ty<'a, 'll: 'a, 'tcx: 'll, Bx : BuilderMethods<'a, 'll, 'tcx>>(
    bx: &mut Bx,
    dst: <Bx::CodegenCx as Backend<'ll>>::Value,
    src: <Bx::CodegenCx as Backend<'ll>>::Value,
    layout: TyLayout<'tcx>,
    align: Align,
    flags: MemFlags,
) {
    let size = layout.size.bytes();
    if size == 0 {
        return;
    }

    bx.call_memcpy(dst, src, bx.cx().const_usize(size), align, flags);
}

pub fn codegen_instance<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    instance: Instance<'tcx>
) where &'a Bx::CodegenCx : LayoutOf<Ty = Ty<'tcx>, TyLayout=TyLayout<'tcx>> + HasTyCtxt<'tcx> {
    let _s = if cx.sess().codegen_stats() {
        let mut instance_name = String::new();
        DefPathBasedNames::new(*cx.tcx(), true, true)
            .push_def_path(instance.def_id(), &mut instance_name);
        Some(StatRecorder::new(cx, instance_name))
    } else {
        None
    };

    // this is an info! to allow collecting monomorphization statistics
    // and to allow finding the last function before LLVM aborts from
    // release builds.
    info!("codegen_instance({})", instance);

    let fn_ty = instance.ty(*cx.tcx());
    let sig = common::ty_fn_sig(cx, fn_ty);
    let sig = cx.tcx().normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);

    let lldecl = match cx.instances().borrow().get(&instance) {
        Some(&val) => val,
        None => bug!("Instance `{:?}` not already declared", instance)
    };

    cx.stats().borrow_mut().n_closures += 1;

    let mir = cx.tcx().instance_mir(instance.def);
    mir::codegen_mir::<'a, 'll, 'tcx, Bx>(
        cx, lldecl, &mir, instance, sig
    );
}

/// Create the `main` function which will initialize the rust runtime and call
/// users main function.
pub fn maybe_create_entry_wrapper<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    cx: &'a Bx::CodegenCx
) {
    let (main_def_id, span) = match *cx.sess().entry_fn.borrow() {
        Some((id, span, _)) => {
            (cx.tcx().hir.local_def_id(id), span)
        }
        None => return,
    };

    let instance = Instance::mono(*cx.tcx(), main_def_id);

    if !cx.codegen_unit().contains_item(&MonoItem::Fn(instance)) {
        // We want to create the wrapper in the same codegen unit as Rust's main
        // function.
        return;
    }

    let main_llfn = cx.get_fn(instance);

    let et = cx.sess().entry_fn.get().map(|e| e.2);
    match et {
        Some(EntryFnType::Main) => create_entry_fn::<Bx>(cx, span, main_llfn, main_def_id, true),
        Some(EntryFnType::Start) => create_entry_fn::<Bx>(cx, span, main_llfn, main_def_id, false),
        None => {}    // Do nothing.
    }

    fn create_entry_fn<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
        cx: &'a Bx::CodegenCx,
        sp: Span,
        rust_main: <Bx::CodegenCx as Backend<'ll>>::Value,
        rust_main_def_id: DefId,
        use_start_lang_item: bool,
    ) {
        let llfty =
            cx.type_func(&[cx.type_int(), cx.type_ptr_to(cx.type_i8p())], cx.type_int());

        let main_ret_ty = cx.tcx().fn_sig(rust_main_def_id).output();
        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = cx.tcx().erase_regions(
            &main_ret_ty.no_late_bound_regions().unwrap(),
        );

        if cx.get_defined_value("main").is_some() {
            // FIXME: We should be smart and show a better diagnostic here.
            cx.sess().struct_span_err(sp, "entry symbol `main` defined multiple times")
                      .help("did you use #[no_mangle] on `fn main`? Use #[start] instead")
                      .emit();
            cx.sess().abort_if_errors();
            bug!();
        }
        let llfn = cx.declare_cfn("main", llfty);

        // `main` should respect same config for frame pointer elimination as rest of code
        cx.set_frame_pointer_elimination(llfn);
        cx.apply_target_cpu_attr(llfn);

        let mut bx = Bx::new_block(&cx, llfn, "top");

        bx.insert_reference_to_gdb_debug_scripts_section_global();

        // Params from native main() used as args for rust start function
        let param_argc = cx.get_param(llfn, 0);
        let param_argv = cx.get_param(llfn, 1);
        let arg_argc = bx.intcast(param_argc, cx.type_isize(), true);
        let arg_argv = param_argv;

        let (start_fn, args) = if use_start_lang_item {
            let start_def_id = cx.tcx().require_lang_item(StartFnLangItem);
            let start_fn = callee::resolve_and_get_fn(
                cx,
                start_def_id,
                cx.tcx().intern_substs(&[main_ret_ty.into()]),
            );
            (start_fn, vec![bx.pointercast(rust_main, cx.type_ptr_to(cx.type_i8p())),
                            arg_argc, arg_argv])
        } else {
            debug!("using user-defined start fn");
            (rust_main, vec![arg_argc, arg_argv])
        };

        let result = bx.call(start_fn, &args, None);
        let cast = bx.intcast(result, cx.type_int(), true); 
        bx.ret(cast);
    }
}

pub const CODEGEN_WORKER_ID: usize = ::std::usize::MAX;
pub const CODEGEN_WORKER_TIMELINE: time_graph::TimelineId =
    time_graph::TimelineId(CODEGEN_WORKER_ID);
pub const CODEGEN_WORK_PACKAGE_KIND: time_graph::WorkPackageKind =
    time_graph::WorkPackageKind(&["#DE9597", "#FED1D3", "#FDC5C7", "#B46668", "#88494B"]);


pub fn codegen_crate<B : ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'ll, 'tcx, 'tcx>,
    rx: mpsc::Receiver<Box<dyn Any + Send>>
) -> B::OngoingCodegen {

    check_for_rustc_errors_attr(tcx);

    if let Some(true) = tcx.sess.opts.debugging_opts.thinlto {
        if backend.thin_lto_available() {
            tcx.sess.fatal("this compiler's LLVM does not support ThinLTO");
        }
    }

    if (tcx.sess.opts.debugging_opts.pgo_gen.is_some() ||
        !tcx.sess.opts.debugging_opts.pgo_use.is_empty()) &&
        backend.pgo_available()
    {
        tcx.sess.fatal("this compiler's LLVM does not support PGO");
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);

    // Codegen the metadata.
    tcx.sess.profiler(|p| p.start_activity(ProfileCategory::Codegen));

    let metadata_cgu_name = cgu_name_builder.build_cgu_name(LOCAL_CRATE,
                                                            &["crate"],
                                                            Some("metadata")).as_str()
                                                                             .to_string();
    let metadata_llvm_module = backend.new_metadata(tcx.sess, &metadata_cgu_name);
    let metadata = time(tcx.sess, "write metadata", || {
        backend.write_metadata(tcx, &metadata_llvm_module)
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
        let ongoing_codegen = backend.start_async_codegen(
            tcx,
            time_graph.clone(),
            metadata,
            rx,
            1);

        backend.submit_pre_codegened_module_to_llvm(&ongoing_codegen, tcx, metadata_module);
        backend.codegen_finished(&ongoing_codegen, tcx);

        assert_and_save_dep_graph(tcx);

        backend.check_for_errors(&ongoing_codegen, tcx.sess);

        return ongoing_codegen;
    }

    // Run the monomorphization collector and partition the collected items into
    // codegen units.
    let codegen_units =
        tcx.collect_and_partition_mono_items(LOCAL_CRATE).1;
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

    let ongoing_codegen = backend.start_async_codegen(
        tcx,
        time_graph.clone(),
        metadata,
        rx,
        codegen_units.len());

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
            list.iter().any(|linkage| {
                match linkage {
                    Linkage::Dynamic => true,
                    _ => false,
                }
            })
        });
    let allocator_module = if any_dynamic_crate {
        None
    } else if let Some(kind) = *tcx.sess.allocator_kind.get() {
        let llmod_id = cgu_name_builder.build_cgu_name(LOCAL_CRATE,
                                                       &["crate"],
                                                       Some("allocator")).as_str()
                                                                         .to_string();
        let modules = backend.new_metadata(tcx.sess, &llmod_id);
        time(tcx.sess, "write allocator module", || {
            backend.codegen_allocator(tcx, &modules, kind)
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
        backend.submit_pre_codegened_module_to_llvm(&ongoing_codegen, tcx, allocator_module);
    }

    backend.submit_pre_codegened_module_to_llvm(&ongoing_codegen, tcx, metadata_module);

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
        backend.wait_for_signal_to_codegen_item(&ongoing_codegen);
        backend.check_for_errors(&ongoing_codegen, tcx.sess);

        let cgu_reuse = determine_cgu_reuse(tcx, &cgu);
        tcx.sess.cgu_reuse_tracker.set_actual_reuse(&cgu.name().as_str(), cgu_reuse);

        match cgu_reuse {
            CguReuse::No => {
                let _timing_guard = time_graph.as_ref().map(|time_graph| {
                    time_graph.start(CODEGEN_WORKER_TIMELINE,
                                     CODEGEN_WORK_PACKAGE_KIND,
                                     &format!("codegen {}", cgu.name()))
                });
                let start_time = Instant::now();
                let stats = backend.compile_codegen_unit(tcx, *cgu.name());
                all_stats.extend(stats);
                total_codegen_time += start_time.elapsed();
                false
            }
            CguReuse::PreLto => {
                backend.submit_pre_lto_module_to_llvm(tcx, CachedModuleCodegen {
                    name: cgu.name().to_string(),
                    source: cgu.work_product(tcx),
                });
                true
            }
            CguReuse::PostLto => {
                backend.submit_post_lto_module_to_llvm(tcx, CachedModuleCodegen {
                    name: cgu.name().to_string(),
                    source: cgu.work_product(tcx),
                });
                true
            }
        };
    }

    backend.codegen_finished(&ongoing_codegen, tcx);

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

    backend.check_for_errors(&ongoing_codegen, tcx.sess);

    assert_and_save_dep_graph(tcx);
    ongoing_codegen
}

fn assert_and_save_dep_graph<'ll, 'tcx>(tcx: TyCtxt<'ll, 'tcx, 'tcx>) {
    time(tcx.sess,
         "assert dep graph",
         || rustc_incremental::assert_dep_graph(tcx));

    time(tcx.sess,
         "serialize dep graph",
         || rustc_incremental::save_dep_graph(tcx));
}

fn collect_and_partition_mono_items<'ll, 'tcx>(
    tcx: TyCtxt<'ll, 'tcx, 'tcx>,
    cnum: CrateNum,
) -> (Arc<DefIdSet>, Arc<Vec<Arc<CodegenUnit<'tcx>>>>)
{
    assert_eq!(cnum, LOCAL_CRATE);

    let collection_mode = match tcx.sess.opts.debugging_opts.print_mono_items {
        Some(ref s) => {
            let mode_string = s.to_lowercase();
            let mode_string = mode_string.trim();
            if mode_string == "eager" {
                MonoItemCollectionMode::Eager
            } else {
                if mode_string != "lazy" {
                    let message = format!("Unknown codegen-item collection mode '{}'. \
                                           Falling back to 'lazy' mode.",
                                           mode_string);
                    tcx.sess.warn(&message);
                }

                MonoItemCollectionMode::Lazy
            }
        }
        None => {
            if tcx.sess.opts.cg.link_dead_code {
                MonoItemCollectionMode::Eager
            } else {
                MonoItemCollectionMode::Lazy
            }
        }
    };

    let (items, inlining_map) =
        time(tcx.sess, "monomorphization collection", || {
            collector::collect_crate_mono_items(tcx, collection_mode)
    });

    tcx.sess.abort_if_errors();

    ::rustc_mir::monomorphize::assert_symbols_are_distinct(tcx, items.iter());

    let strategy = if tcx.sess.opts.incremental.is_some() {
        PartitioningStrategy::PerModule
    } else {
        PartitioningStrategy::FixedUnitCount(tcx.sess.codegen_units())
    };

    let codegen_units = time(tcx.sess, "codegen unit partitioning", || {
        partitioning::partition(tcx,
                                items.iter().cloned(),
                                strategy,
                                &inlining_map)
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>()
    });

    let mono_items: DefIdSet = items.iter().filter_map(|mono_item| {
        match *mono_item {
            MonoItem::Fn(ref instance) => Some(instance.def_id()),
            MonoItem::Static(def_id) => Some(def_id),
            _ => None,
        }
    }).collect();

    if tcx.sess.opts.debugging_opts.print_mono_items.is_some() {
        let mut item_to_cgus: FxHashMap<_, Vec<_>> = FxHashMap();

        for cgu in &codegen_units {
            for (&mono_item, &linkage) in cgu.items() {
                item_to_cgus.entry(mono_item)
                            .or_default()
                            .push((cgu.name().clone(), linkage));
            }
        }

        let mut item_keys: Vec<_> = items
            .iter()
            .map(|i| {
                let mut output = i.to_string(tcx);
                output.push_str(" @@");
                let mut empty = Vec::new();
                let cgus = item_to_cgus.get_mut(i).unwrap_or(&mut empty);
                cgus.as_mut_slice().sort_by_key(|&(ref name, _)| name.clone());
                cgus.dedup();
                for &(ref cgu_name, (linkage, _)) in cgus.iter() {
                    output.push_str(" ");
                    output.push_str(&cgu_name.as_str());

                    let linkage_abbrev = match linkage {
                        Linkage::External => "External",
                        Linkage::AvailableExternally => "Available",
                        Linkage::LinkOnceAny => "OnceAny",
                        Linkage::LinkOnceODR => "OnceODR",
                        Linkage::WeakAny => "WeakAny",
                        Linkage::WeakODR => "WeakODR",
                        Linkage::Appending => "Appending",
                        Linkage::Internal => "Internal",
                        Linkage::Private => "Private",
                        Linkage::ExternalWeak => "ExternalWeak",
                        Linkage::Common => "Common",
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
            println!("MONO_ITEM {}", item);
        }
    }

    (Arc::new(mono_items), Arc::new(codegen_units))
}


impl CrateInfo {
    pub fn new(tcx: TyCtxt) -> CrateInfo {
        let mut info = CrateInfo {
            panic_runtime: None,
            compiler_builtins: None,
            profiler_runtime: None,
            sanitizer_runtime: None,
            is_no_builtins: FxHashSet(),
            native_libraries: FxHashMap(),
            used_libraries: tcx.native_libraries(LOCAL_CRATE),
            link_args: tcx.link_args(LOCAL_CRATE),
            crate_name: FxHashMap(),
            used_crates_dynamic: cstore::used_crates(tcx, LinkagePreference::RequireDynamic),
            used_crates_static: cstore::used_crates(tcx, LinkagePreference::RequireStatic),
            used_crate_source: FxHashMap(),
            wasm_imports: FxHashMap(),
            lang_item_to_crate: FxHashMap(),
            missing_lang_items: FxHashMap(),
        };
        let lang_items = tcx.lang_items();

        let load_wasm_items = tcx.sess.crate_types.borrow()
            .iter()
            .any(|c| *c != config::CrateType::Rlib) &&
            tcx.sess.opts.target_triple.triple() == "wasm32-unknown-unknown";

        if load_wasm_items {
            info.load_wasm_imports(tcx, LOCAL_CRATE);
        }

        for &cnum in tcx.crates().iter() {
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
        for (&id, module) in tcx.wasm_import_module_map(cnum).iter() {
            let instance = Instance::mono(tcx, id);
            let import_name = tcx.symbol_name(instance);
            self.wasm_imports.insert(import_name.to_string(), module.clone());
        }
    }
}

fn is_codegened_item(tcx: TyCtxt, id: DefId) -> bool {
    let (all_mono_items, _) =
        tcx.collect_and_partition_mono_items(LOCAL_CRATE);
    all_mono_items.contains(&id)
}


pub fn provide(providers: &mut Providers) {
    providers.collect_and_partition_mono_items =
        collect_and_partition_mono_items;

    providers.is_codegened_item = is_codegened_item;

    providers.codegen_unit = |tcx, name| {
        let (_, all) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);
        all.iter()
            .find(|cgu| *cgu.name() == name)
            .cloned()
            .unwrap_or_else(|| panic!("failed to find cgu with name {:?}", name))
    };

    provide_extern(providers);
}

pub fn provide_extern(providers: &mut Providers) {
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

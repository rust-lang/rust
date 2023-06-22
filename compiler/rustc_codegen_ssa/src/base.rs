use crate::back::link::are_upstream_rust_objects_already_included;
use crate::back::metadata::create_compressed_metadata_file;
use crate::back::write::{
    compute_per_cgu_lto_type, start_async_codegen, submit_codegened_module_to_llvm,
    submit_post_lto_module_to_llvm, submit_pre_lto_module_to_llvm, ComputedLtoType, OngoingCodegen,
};
use crate::common::{IntPredicate, RealPredicate, TypeKind};
use crate::errors;
use crate::meth;
use crate::mir;
use crate::mir::operand::OperandValue;
use crate::mir::place::PlaceRef;
use crate::traits::*;
use crate::{CachedModuleCodegen, CompiledModule, CrateInfo, MemFlags, ModuleCodegen, ModuleKind};

use rustc_ast::expand::allocator::{global_fn_name, AllocatorKind, ALLOCATOR_METHODS};
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::profiling::{get_resident_set_size, print_time_passes_entry};
use rustc_data_structures::sync::par_map;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::lang_items::LangItem;
use rustc_metadata::EncodedMetadata;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::middle::debugger_visualizer::{DebuggerVisualizerFile, DebuggerVisualizerType};
use rustc_middle::middle::exported_symbols;
use rustc_middle::middle::exported_symbols::SymbolExportKind;
use rustc_middle::middle::lang_items;
use rustc_middle::mir::mono::{CodegenUnit, CodegenUnitNameBuilder, MonoItem};
use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_session::cgu_reuse_tracker::CguReuse;
use rustc_session::config::{self, CrateType, EntryFnType, OutputType};
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::Symbol;
use rustc_target::abi::{Align, FIRST_VARIANT};

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use itertools::Itertools;

pub fn bin_op_to_icmp_predicate(op: hir::BinOpKind, signed: bool) -> IntPredicate {
    match op {
        hir::BinOpKind::Eq => IntPredicate::IntEQ,
        hir::BinOpKind::Ne => IntPredicate::IntNE,
        hir::BinOpKind::Lt => {
            if signed {
                IntPredicate::IntSLT
            } else {
                IntPredicate::IntULT
            }
        }
        hir::BinOpKind::Le => {
            if signed {
                IntPredicate::IntSLE
            } else {
                IntPredicate::IntULE
            }
        }
        hir::BinOpKind::Gt => {
            if signed {
                IntPredicate::IntSGT
            } else {
                IntPredicate::IntUGT
            }
        }
        hir::BinOpKind::Ge => {
            if signed {
                IntPredicate::IntSGE
            } else {
                IntPredicate::IntUGE
            }
        }
        op => bug!(
            "comparison_op_to_icmp_predicate: expected comparison operator, \
             found {:?}",
            op
        ),
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
            bug!(
                "comparison_op_to_fcmp_predicate: expected comparison operator, \
                 found {:?}",
                op
            );
        }
    }
}

pub fn compare_simd_types<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    lhs: Bx::Value,
    rhs: Bx::Value,
    t: Ty<'tcx>,
    ret_ty: Bx::Type,
    op: hir::BinOpKind,
) -> Bx::Value {
    let signed = match t.kind() {
        ty::Float(_) => {
            let cmp = bin_op_to_fcmp_predicate(op);
            let cmp = bx.fcmp(cmp, lhs, rhs);
            return bx.sext(cmp, ret_ty);
        }
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

/// Retrieves the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit odd. It is intended for use in an upcast,
/// where the new vtable for an object will be derived from the old one.
pub fn unsized_info<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
    old_info: Option<Bx::Value>,
) -> Bx::Value {
    let cx = bx.cx();
    let (source, target) =
        cx.tcx().struct_lockstep_tails_erasing_lifetimes(source, target, bx.param_env());
    match (source.kind(), target.kind()) {
        (&ty::Array(_, len), &ty::Slice(_)) => {
            cx.const_usize(len.eval_target_usize(cx.tcx(), ty::ParamEnv::reveal_all()))
        }
        (
            &ty::Dynamic(ref data_a, _, src_dyn_kind),
            &ty::Dynamic(ref data_b, _, target_dyn_kind),
        ) if src_dyn_kind == target_dyn_kind => {
            let old_info =
                old_info.expect("unsized_info: missing old info for trait upcasting coercion");
            if data_a.principal_def_id() == data_b.principal_def_id() {
                // A NOP cast that doesn't actually change anything, should be allowed even with invalid vtables.
                return old_info;
            }

            // trait upcasting coercion

            let vptr_entry_idx =
                cx.tcx().vtable_trait_upcasting_coercion_new_vptr_slot((source, target));

            if let Some(entry_idx) = vptr_entry_idx {
                let ptr_ty = cx.type_i8p();
                let ptr_align = cx.tcx().data_layout.pointer_align.abi;
                let vtable_ptr_ty = vtable_ptr_ty(cx, target, target_dyn_kind);
                let llvtable = bx.pointercast(old_info, bx.type_ptr_to(ptr_ty));
                let gep = bx.inbounds_gep(
                    ptr_ty,
                    llvtable,
                    &[bx.const_usize(u64::try_from(entry_idx).unwrap())],
                );
                let new_vptr = bx.load(ptr_ty, gep, ptr_align);
                bx.nonnull_metadata(new_vptr);
                // VTable loads are invariant.
                bx.set_invariant_load(new_vptr);
                bx.pointercast(new_vptr, vtable_ptr_ty)
            } else {
                old_info
            }
        }
        (_, &ty::Dynamic(ref data, _, target_dyn_kind)) => {
            let vtable_ptr_ty = vtable_ptr_ty(cx, target, target_dyn_kind);
            cx.const_ptrcast(meth::get_vtable(cx, source, data.principal()), vtable_ptr_ty)
        }
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}", source, target),
    }
}

// Returns the vtable pointer type of a `dyn` or `dyn*` type
fn vtable_ptr_ty<'tcx, Cx: CodegenMethods<'tcx>>(
    cx: &Cx,
    target: Ty<'tcx>,
    kind: ty::DynKind,
) -> <Cx as BackendTypes>::Type {
    cx.scalar_pair_element_backend_type(
        cx.layout_of(match kind {
            // vtable is the second field of `*mut dyn Trait`
            ty::Dyn => cx.tcx().mk_mut_ptr(target),
            // vtable is the second field of `dyn* Trait`
            ty::DynStar => target,
        }),
        1,
        true,
    )
}

/// Coerces `src` to `dst_ty`. `src_ty` must be a pointer.
pub fn unsize_ptr<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    src: Bx::Value,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>,
    old_info: Option<Bx::Value>,
) -> (Bx::Value, Bx::Value) {
    debug!("unsize_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (src_ty.kind(), dst_ty.kind()) {
        (&ty::Ref(_, a, _), &ty::Ref(_, b, _) | &ty::RawPtr(ty::TypeAndMut { ty: b, .. }))
        | (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }), &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
            assert_eq!(bx.cx().type_is_sized(a), old_info.is_none());
            let ptr_ty = bx.cx().type_ptr_to(bx.cx().backend_type(bx.cx().layout_of(b)));
            (bx.pointercast(src, ptr_ty), unsized_info(bx, a, b, old_info))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b);
            let src_layout = bx.cx().layout_of(src_ty);
            let dst_layout = bx.cx().layout_of(dst_ty);
            if src_ty == dst_ty {
                return (src, old_info.unwrap());
            }
            let mut result = None;
            for i in 0..src_layout.fields.count() {
                let src_f = src_layout.field(bx.cx(), i);
                if src_f.is_zst() {
                    continue;
                }

                assert_eq!(src_layout.fields.offset(i).bytes(), 0);
                assert_eq!(dst_layout.fields.offset(i).bytes(), 0);
                assert_eq!(src_layout.size, src_f.size);

                let dst_f = dst_layout.field(bx.cx(), i);
                assert_ne!(src_f.ty, dst_f.ty);
                assert_eq!(result, None);
                result = Some(unsize_ptr(bx, src, src_f.ty, dst_f.ty, old_info));
            }
            let (lldata, llextra) = result.unwrap();
            let lldata_ty = bx.cx().scalar_pair_element_backend_type(dst_layout, 0, true);
            let llextra_ty = bx.cx().scalar_pair_element_backend_type(dst_layout, 1, true);
            // HACK(eddyb) have to bitcast pointers until LLVM removes pointee types.
            (bx.bitcast(lldata, lldata_ty), bx.bitcast(llextra, llextra_ty))
        }
        _ => bug!("unsize_ptr: called on bad types"),
    }
}

/// Coerces `src` to `dst_ty` which is guaranteed to be a `dyn*` type.
pub fn cast_to_dyn_star<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    src: Bx::Value,
    src_ty_and_layout: TyAndLayout<'tcx>,
    dst_ty: Ty<'tcx>,
    old_info: Option<Bx::Value>,
) -> (Bx::Value, Bx::Value) {
    debug!("cast_to_dyn_star: {:?} => {:?}", src_ty_and_layout.ty, dst_ty);
    assert!(
        matches!(dst_ty.kind(), ty::Dynamic(_, _, ty::DynStar)),
        "destination type must be a dyn*"
    );
    // FIXME(dyn-star): We can remove this when all supported LLVMs use opaque ptrs only.
    let unit_ptr = bx.cx().type_ptr_to(bx.cx().type_struct(&[], false));
    let src = match bx.cx().type_kind(bx.cx().backend_type(src_ty_and_layout)) {
        TypeKind::Pointer => bx.pointercast(src, unit_ptr),
        TypeKind::Integer => bx.inttoptr(src, unit_ptr),
        // FIXME(dyn-star): We probably have to do a bitcast first, then inttoptr.
        kind => bug!("unexpected TypeKind for left-hand side of `dyn*` cast: {kind:?}"),
    };
    (src, unsized_info(bx, src_ty_and_layout.ty, dst_ty, old_info))
}

/// Coerces `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty`, and stores the result in `dst`.
pub fn coerce_unsized_into<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    src: PlaceRef<'tcx, Bx::Value>,
    dst: PlaceRef<'tcx, Bx::Value>,
) {
    let src_ty = src.layout.ty;
    let dst_ty = dst.layout.ty;
    match (src_ty.kind(), dst_ty.kind()) {
        (&ty::Ref(..), &ty::Ref(..) | &ty::RawPtr(..)) | (&ty::RawPtr(..), &ty::RawPtr(..)) => {
            let (base, info) = match bx.load_operand(src).val {
                OperandValue::Pair(base, info) => unsize_ptr(bx, base, src_ty, dst_ty, Some(info)),
                OperandValue::Immediate(base) => unsize_ptr(bx, base, src_ty, dst_ty, None),
                OperandValue::Ref(..) | OperandValue::ZeroSized => bug!(),
            };
            OperandValue::Pair(base, info).store(bx, dst);
        }

        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b);

            for i in def_a.variant(FIRST_VARIANT).fields.indices() {
                let src_f = src.project_field(bx, i.as_usize());
                let dst_f = dst.project_field(bx, i.as_usize());

                if dst_f.layout.is_zst() {
                    continue;
                }

                if src_f.layout.ty == dst_f.layout.ty {
                    memcpy_ty(
                        bx,
                        dst_f.llval,
                        dst_f.align,
                        src_f.llval,
                        src_f.align,
                        src_f.layout,
                        MemFlags::empty(),
                    );
                } else {
                    coerce_unsized_into(bx, src_f, dst_f);
                }
            }
        }
        _ => bug!("coerce_unsized_into: invalid coercion {:?} -> {:?}", src_ty, dst_ty,),
    }
}

pub fn cast_shift_expr_rhs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    lhs: Bx::Value,
    rhs: Bx::Value,
) -> Bx::Value {
    // Shifts may have any size int on the rhs
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
        // FIXME (#1877: If in the future shifting by negative
        // values is no longer undefined then this is wrong.
        bx.zext(rhs, lhs_llty)
    } else {
        rhs
    }
}

/// Returns `true` if this session's target will use SEH-based unwinding.
///
/// This is only true for MSVC targets, and even then the 64-bit MSVC target
/// currently uses SEH-ish unwinding with DWARF info tables to the side (same as
/// 64-bit MinGW) instead of "full SEH".
pub fn wants_msvc_seh(sess: &Session) -> bool {
    sess.target.is_like_msvc
}

pub fn memcpy_ty<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    dst: Bx::Value,
    dst_align: Align,
    src: Bx::Value,
    src_align: Align,
    layout: TyAndLayout<'tcx>,
    flags: MemFlags,
) {
    let size = layout.size.bytes();
    if size == 0 {
        return;
    }

    if flags == MemFlags::empty()
        && let Some(bty) = bx.cx().scalar_copy_backend_type(layout)
    {
        // I look forward to only supporting opaque pointers
        let pty = bx.type_ptr_to(bty);
        let src = bx.pointercast(src, pty);
        let dst = bx.pointercast(dst, pty);

        let temp = bx.load(bty, src, src_align);
        bx.store(temp, dst, dst_align);
    } else {
        bx.memcpy(dst, dst_align, src, src_align, bx.cx().const_usize(size), flags);
    }
}

pub fn codegen_instance<'a, 'tcx: 'a, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    instance: Instance<'tcx>,
) {
    // this is an info! to allow collecting monomorphization statistics
    // and to allow finding the last function before LLVM aborts from
    // release builds.
    info!("codegen_instance({})", instance);

    mir::codegen_mir::<Bx>(cx, instance);
}

/// Creates the `main` function which will initialize the rust runtime and call
/// users main function.
pub fn maybe_create_entry_wrapper<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
) -> Option<Bx::Function> {
    let (main_def_id, entry_type) = cx.tcx().entry_fn(())?;
    let main_is_local = main_def_id.is_local();
    let instance = Instance::mono(cx.tcx(), main_def_id);

    if main_is_local {
        // We want to create the wrapper in the same codegen unit as Rust's main
        // function.
        if !cx.codegen_unit().contains_item(&MonoItem::Fn(instance)) {
            return None;
        }
    } else if !cx.codegen_unit().is_primary() {
        // We want to create the wrapper only when the codegen unit is the primary one
        return None;
    }

    let main_llfn = cx.get_fn_addr(instance);

    let entry_fn = create_entry_fn::<Bx>(cx, main_llfn, main_def_id, entry_type);
    return Some(entry_fn);

    fn create_entry_fn<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        cx: &'a Bx::CodegenCx,
        rust_main: Bx::Value,
        rust_main_def_id: DefId,
        entry_type: EntryFnType,
    ) -> Bx::Function {
        // The entry function is either `int main(void)` or `int main(int argc, char **argv)`,
        // depending on whether the target needs `argc` and `argv` to be passed in.
        let llfty = if cx.sess().target.main_needs_argc_argv {
            cx.type_func(&[cx.type_int(), cx.type_ptr_to(cx.type_i8p())], cx.type_int())
        } else {
            cx.type_func(&[], cx.type_int())
        };

        let main_ret_ty = cx.tcx().fn_sig(rust_main_def_id).no_bound_vars().unwrap().output();
        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = cx.tcx().normalize_erasing_regions(
            ty::ParamEnv::reveal_all(),
            main_ret_ty.no_bound_vars().unwrap(),
        );

        let Some(llfn) = cx.declare_c_main(llfty) else {
            // FIXME: We should be smart and show a better diagnostic here.
            let span = cx.tcx().def_span(rust_main_def_id);
            cx.sess().emit_err(errors::MultipleMainFunctions { span });
            cx.sess().abort_if_errors();
            bug!();
        };

        // `main` should respect same config for frame pointer elimination as rest of code
        cx.set_frame_pointer_type(llfn);
        cx.apply_target_cpu_attr(llfn);

        let llbb = Bx::append_block(&cx, llfn, "top");
        let mut bx = Bx::build(&cx, llbb);

        bx.insert_reference_to_gdb_debug_scripts_section_global();

        let isize_ty = cx.type_isize();
        let i8pp_ty = cx.type_ptr_to(cx.type_i8p());
        let (arg_argc, arg_argv) = get_argc_argv(cx, &mut bx);

        let (start_fn, start_ty, args) = if let EntryFnType::Main { sigpipe } = entry_type {
            let start_def_id = cx.tcx().require_lang_item(LangItem::Start, None);
            let start_fn = cx.get_fn_addr(
                ty::Instance::resolve(
                    cx.tcx(),
                    ty::ParamEnv::reveal_all(),
                    start_def_id,
                    cx.tcx().mk_substs(&[main_ret_ty.into()]),
                )
                .unwrap()
                .unwrap(),
            );

            let i8_ty = cx.type_i8();
            let arg_sigpipe = bx.const_u8(sigpipe);

            let start_ty =
                cx.type_func(&[cx.val_ty(rust_main), isize_ty, i8pp_ty, i8_ty], isize_ty);
            (start_fn, start_ty, vec![rust_main, arg_argc, arg_argv, arg_sigpipe])
        } else {
            debug!("using user-defined start fn");
            let start_ty = cx.type_func(&[isize_ty, i8pp_ty], isize_ty);
            (rust_main, start_ty, vec![arg_argc, arg_argv])
        };

        let result = bx.call(start_ty, None, None, start_fn, &args, None);
        let cast = bx.intcast(result, cx.type_int(), true);
        bx.ret(cast);

        llfn
    }
}

/// Obtain the `argc` and `argv` values to pass to the rust start function.
fn get_argc_argv<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    bx: &mut Bx,
) -> (Bx::Value, Bx::Value) {
    if cx.sess().target.main_needs_argc_argv {
        // Params from native `main()` used as args for rust start function
        let param_argc = bx.get_param(0);
        let param_argv = bx.get_param(1);
        let arg_argc = bx.intcast(param_argc, cx.type_isize(), true);
        let arg_argv = param_argv;
        (arg_argc, arg_argv)
    } else {
        // The Rust start function doesn't need `argc` and `argv`, so just pass zeros.
        let arg_argc = bx.const_int(cx.type_int(), 0);
        let arg_argv = bx.const_null(cx.type_ptr_to(cx.type_i8p()));
        (arg_argc, arg_argv)
    }
}

/// This function returns all of the debugger visualizers specified for the
/// current crate as well as all upstream crates transitively that match the
/// `visualizer_type` specified.
pub fn collect_debugger_visualizers_transitive(
    tcx: TyCtxt<'_>,
    visualizer_type: DebuggerVisualizerType,
) -> BTreeSet<DebuggerVisualizerFile> {
    tcx.debugger_visualizers(LOCAL_CRATE)
        .iter()
        .chain(
            tcx.crates(())
                .iter()
                .filter(|&cnum| {
                    let used_crate_source = tcx.used_crate_source(*cnum);
                    used_crate_source.rlib.is_some() || used_crate_source.rmeta.is_some()
                })
                .flat_map(|&cnum| tcx.debugger_visualizers(cnum)),
        )
        .filter(|visualizer| visualizer.visualizer_type == visualizer_type)
        .cloned()
        .collect::<BTreeSet<_>>()
}

/// Decide allocator kind to codegen. If `Some(_)` this will be the same as
/// `tcx.allocator_kind`, but it may be `None` in more cases (e.g. if using
/// allocator definitions from a dylib dependency).
pub fn allocator_kind_for_codegen(tcx: TyCtxt<'_>) -> Option<AllocatorKind> {
    // If the crate doesn't have an `allocator_kind` set then there's definitely
    // no shim to generate. Otherwise we also check our dependency graph for all
    // our output crate types. If anything there looks like its a `Dynamic`
    // linkage, then it's already got an allocator shim and we'll be using that
    // one instead. If nothing exists then it's our job to generate the
    // allocator!
    let any_dynamic_crate = tcx.dependency_formats(()).iter().any(|(_, list)| {
        use rustc_middle::middle::dependency_format::Linkage;
        list.iter().any(|&linkage| linkage == Linkage::Dynamic)
    });
    if any_dynamic_crate { None } else { tcx.allocator_kind(()) }
}

pub fn codegen_crate<B: ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'_>,
    target_cpu: String,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> OngoingCodegen<B> {
    // Skip crate items and just output metadata in -Z no-codegen mode.
    if tcx.sess.opts.unstable_opts.no_codegen || !tcx.sess.opts.output_types.should_codegen() {
        let ongoing_codegen = start_async_codegen(backend, tcx, target_cpu, metadata, None);

        ongoing_codegen.codegen_finished(tcx);

        ongoing_codegen.check_for_errors(tcx.sess);

        return ongoing_codegen;
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);

    // Run the monomorphization collector and partition the collected items into
    // codegen units.
    let codegen_units = tcx.collect_and_partition_mono_items(()).1;

    // Force all codegen_unit queries so they are already either red or green
    // when compile_codegen_unit accesses them. We are not able to re-execute
    // the codegen_unit query from just the DepNode, so an unknown color would
    // lead to having to re-execute compile_codegen_unit, possibly
    // unnecessarily.
    if tcx.dep_graph.is_fully_enabled() {
        for cgu in codegen_units {
            tcx.ensure().codegen_unit(cgu.name());
        }
    }

    let metadata_module = need_metadata_module.then(|| {
        // Emit compressed metadata object.
        let metadata_cgu_name =
            cgu_name_builder.build_cgu_name(LOCAL_CRATE, &["crate"], Some("metadata")).to_string();
        tcx.sess.time("write_compressed_metadata", || {
            let file_name =
                tcx.output_filenames(()).temp_path(OutputType::Metadata, Some(&metadata_cgu_name));
            let data = create_compressed_metadata_file(
                tcx.sess,
                &metadata,
                &exported_symbols::metadata_symbol_name(tcx),
            );
            if let Err(error) = std::fs::write(&file_name, data) {
                tcx.sess.emit_fatal(errors::MetadataObjectFileWrite { error });
            }
            CompiledModule {
                name: metadata_cgu_name,
                kind: ModuleKind::Metadata,
                object: Some(file_name),
                dwarf_object: None,
                bytecode: None,
            }
        })
    });

    let ongoing_codegen =
        start_async_codegen(backend.clone(), tcx, target_cpu, metadata, metadata_module);

    // Codegen an allocator shim, if necessary.
    if let Some(kind) = allocator_kind_for_codegen(tcx) {
        let llmod_id =
            cgu_name_builder.build_cgu_name(LOCAL_CRATE, &["crate"], Some("allocator")).to_string();
        let module_llvm = tcx.sess.time("write_allocator_module", || {
            backend.codegen_allocator(
                tcx,
                &llmod_id,
                kind,
                // If allocator_kind is Some then alloc_error_handler_kind must
                // also be Some.
                tcx.alloc_error_handler_kind(()).unwrap(),
            )
        });

        ongoing_codegen.submit_pre_codegened_module_to_llvm(
            tcx,
            ModuleCodegen { name: llmod_id, module_llvm, kind: ModuleKind::Allocator },
        );
    }

    // For better throughput during parallel processing by LLVM, we used to sort
    // CGUs largest to smallest. This would lead to better thread utilization
    // by, for example, preventing a large CGU from being processed last and
    // having only one LLVM thread working while the rest remained idle.
    //
    // However, this strategy would lead to high memory usage, as it meant the
    // LLVM-IR for all of the largest CGUs would be resident in memory at once.
    //
    // Instead, we can compromise by ordering CGUs such that the largest and
    // smallest are first, second largest and smallest are next, etc. If there
    // are large size variations, this can reduce memory usage significantly.
    let codegen_units: Vec<_> = {
        let mut sorted_cgus = codegen_units.iter().collect::<Vec<_>>();
        sorted_cgus.sort_by_cached_key(|cgu| cgu.size_estimate());

        let (first_half, second_half) = sorted_cgus.split_at(sorted_cgus.len() / 2);
        second_half.iter().rev().interleave(first_half).copied().collect()
    };

    // Calculate the CGU reuse
    let cgu_reuse = tcx.sess.time("find_cgu_reuse", || {
        codegen_units.iter().map(|cgu| determine_cgu_reuse(tcx, &cgu)).collect::<Vec<_>>()
    });

    let mut total_codegen_time = Duration::new(0, 0);
    let start_rss = tcx.sess.opts.unstable_opts.time_passes.then(|| get_resident_set_size());

    // The non-parallel compiler can only translate codegen units to LLVM IR
    // on a single thread, leading to a staircase effect where the N LLVM
    // threads have to wait on the single codegen threads to generate work
    // for them. The parallel compiler does not have this restriction, so
    // we can pre-load the LLVM queue in parallel before handing off
    // coordination to the OnGoingCodegen scheduler.
    //
    // This likely is a temporary measure. Once we don't have to support the
    // non-parallel compiler anymore, we can compile CGUs end-to-end in
    // parallel and get rid of the complicated scheduling logic.
    let mut pre_compiled_cgus = if tcx.sess.threads() > 1 {
        tcx.sess.time("compile_first_CGU_batch", || {
            // Try to find one CGU to compile per thread.
            let cgus: Vec<_> = cgu_reuse
                .iter()
                .enumerate()
                .filter(|&(_, reuse)| reuse == &CguReuse::No)
                .take(tcx.sess.threads())
                .collect();

            // Compile the found CGUs in parallel.
            let start_time = Instant::now();

            let pre_compiled_cgus = par_map(cgus, |(i, _)| {
                let module = backend.compile_codegen_unit(tcx, codegen_units[i].name());
                (i, module)
            });

            total_codegen_time += start_time.elapsed();

            pre_compiled_cgus
        })
    } else {
        FxHashMap::default()
    };

    for (i, cgu) in codegen_units.iter().enumerate() {
        ongoing_codegen.wait_for_signal_to_codegen_item();
        ongoing_codegen.check_for_errors(tcx.sess);

        let cgu_reuse = cgu_reuse[i];
        tcx.sess.cgu_reuse_tracker.set_actual_reuse(cgu.name().as_str(), cgu_reuse);

        match cgu_reuse {
            CguReuse::No => {
                let (module, cost) = if let Some(cgu) = pre_compiled_cgus.remove(&i) {
                    cgu
                } else {
                    let start_time = Instant::now();
                    let module = backend.compile_codegen_unit(tcx, cgu.name());
                    total_codegen_time += start_time.elapsed();
                    module
                };
                // This will unwind if there are errors, which triggers our `AbortCodegenOnDrop`
                // guard. Unfortunately, just skipping the `submit_codegened_module_to_llvm` makes
                // compilation hang on post-monomorphization errors.
                tcx.sess.abort_if_errors();

                submit_codegened_module_to_llvm(
                    &backend,
                    &ongoing_codegen.coordinator.sender,
                    module,
                    cost,
                );
                false
            }
            CguReuse::PreLto => {
                submit_pre_lto_module_to_llvm(
                    &backend,
                    tcx,
                    &ongoing_codegen.coordinator.sender,
                    CachedModuleCodegen {
                        name: cgu.name().to_string(),
                        source: cgu.previous_work_product(tcx),
                    },
                );
                true
            }
            CguReuse::PostLto => {
                submit_post_lto_module_to_llvm(
                    &backend,
                    &ongoing_codegen.coordinator.sender,
                    CachedModuleCodegen {
                        name: cgu.name().to_string(),
                        source: cgu.previous_work_product(tcx),
                    },
                );
                true
            }
        };
    }

    ongoing_codegen.codegen_finished(tcx);

    // Since the main thread is sometimes blocked during codegen, we keep track
    // -Ztime-passes output manually.
    if tcx.sess.opts.unstable_opts.time_passes {
        let end_rss = get_resident_set_size();

        print_time_passes_entry(
            "codegen_to_LLVM_IR",
            total_codegen_time,
            start_rss.unwrap(),
            end_rss,
            tcx.sess.opts.unstable_opts.time_passes_format,
        );
    }

    ongoing_codegen.check_for_errors(tcx.sess);
    ongoing_codegen
}

impl CrateInfo {
    pub fn new(tcx: TyCtxt<'_>, target_cpu: String) -> CrateInfo {
        let exported_symbols = tcx
            .sess
            .crate_types()
            .iter()
            .map(|&c| (c, crate::back::linker::exported_symbols(tcx, c)))
            .collect();
        let linked_symbols = tcx
            .sess
            .crate_types()
            .iter()
            .map(|&c| (c, crate::back::linker::linked_symbols(tcx, c)))
            .collect();
        let local_crate_name = tcx.crate_name(LOCAL_CRATE);
        let crate_attrs = tcx.hir().attrs(rustc_hir::CRATE_HIR_ID);
        let subsystem = attr::first_attr_value_str_by_name(crate_attrs, sym::windows_subsystem);
        let windows_subsystem = subsystem.map(|subsystem| {
            if subsystem != sym::windows && subsystem != sym::console {
                tcx.sess.emit_fatal(errors::InvalidWindowsSubsystem { subsystem });
            }
            subsystem.to_string()
        });

        // This list is used when generating the command line to pass through to
        // system linker. The linker expects undefined symbols on the left of the
        // command line to be defined in libraries on the right, not the other way
        // around. For more info, see some comments in the add_used_library function
        // below.
        //
        // In order to get this left-to-right dependency ordering, we use the reverse
        // postorder of all crates putting the leaves at the right-most positions.
        let mut compiler_builtins = None;
        let mut used_crates: Vec<_> = tcx
            .postorder_cnums(())
            .iter()
            .rev()
            .copied()
            .filter(|&cnum| {
                let link = !tcx.dep_kind(cnum).macros_only();
                if link && tcx.is_compiler_builtins(cnum) {
                    compiler_builtins = Some(cnum);
                    return false;
                }
                link
            })
            .collect();
        // `compiler_builtins` are always placed last to ensure that they're linked correctly.
        used_crates.extend(compiler_builtins);

        let mut info = CrateInfo {
            target_cpu,
            exported_symbols,
            linked_symbols,
            local_crate_name,
            compiler_builtins,
            profiler_runtime: None,
            is_no_builtins: Default::default(),
            native_libraries: Default::default(),
            used_libraries: tcx.native_libraries(LOCAL_CRATE).iter().map(Into::into).collect(),
            crate_name: Default::default(),
            used_crates,
            used_crate_source: Default::default(),
            dependency_formats: tcx.dependency_formats(()).clone(),
            windows_subsystem,
            natvis_debugger_visualizers: Default::default(),
            feature_packed_bundled_libs: tcx.features().packed_bundled_libs,
        };
        let crates = tcx.crates(());

        let n_crates = crates.len();
        info.native_libraries.reserve(n_crates);
        info.crate_name.reserve(n_crates);
        info.used_crate_source.reserve(n_crates);

        for &cnum in crates.iter() {
            info.native_libraries
                .insert(cnum, tcx.native_libraries(cnum).iter().map(Into::into).collect());
            info.crate_name.insert(cnum, tcx.crate_name(cnum));

            let used_crate_source = tcx.used_crate_source(cnum);
            info.used_crate_source.insert(cnum, used_crate_source.clone());
            if tcx.is_profiler_runtime(cnum) {
                info.profiler_runtime = Some(cnum);
            }
            if tcx.is_no_builtins(cnum) {
                info.is_no_builtins.insert(cnum);
            }
        }

        // Handle circular dependencies in the standard library.
        // See comment before `add_linked_symbol_object` function for the details.
        // If global LTO is enabled then almost everything (*) is glued into a single object file,
        // so this logic is not necessary and can cause issues on some targets (due to weak lang
        // item symbols being "privatized" to that object file), so we disable it.
        // (*) Native libs, and `#[compiler_builtins]` and `#[no_builtins]` crates are not glued,
        // and we assume that they cannot define weak lang items. This is not currently enforced
        // by the compiler, but that's ok because all this stuff is unstable anyway.
        let target = &tcx.sess.target;
        if !are_upstream_rust_objects_already_included(tcx.sess) {
            let missing_weak_lang_items: FxHashSet<Symbol> = info
                .used_crates
                .iter()
                .flat_map(|&cnum| tcx.missing_lang_items(cnum))
                .filter(|l| l.is_weak())
                .filter_map(|&l| {
                    let name = l.link_name()?;
                    lang_items::required(tcx, l).then_some(name)
                })
                .collect();
            let prefix = if target.is_like_windows && target.arch == "x86" { "_" } else { "" };
            info.linked_symbols
                .iter_mut()
                .filter(|(crate_type, _)| {
                    !matches!(crate_type, CrateType::Rlib | CrateType::Staticlib)
                })
                .for_each(|(_, linked_symbols)| {
                    linked_symbols.extend(
                        missing_weak_lang_items
                            .iter()
                            .map(|item| (format!("{prefix}{item}"), SymbolExportKind::Text)),
                    );
                    if tcx.allocator_kind(()).is_some() {
                        // At least one crate needs a global allocator. This crate may be placed
                        // after the crate that defines it in the linker order, in which case some
                        // linkers return an error. By adding the global allocator shim methods to
                        // the linked_symbols list, linking the generated symbols.o will ensure that
                        // circular dependencies involving the global allocator don't lead to linker
                        // errors.
                        linked_symbols.extend(ALLOCATOR_METHODS.iter().map(|method| {
                            (
                                format!("{prefix}{}", global_fn_name(method.name).as_str()),
                                SymbolExportKind::Text,
                            )
                        }));
                    }
                });
        }

        let embed_visualizers = tcx.sess.crate_types().iter().any(|&crate_type| match crate_type {
            CrateType::Executable | CrateType::Dylib | CrateType::Cdylib => {
                // These are crate types for which we invoke the linker and can embed
                // NatVis visualizers.
                true
            }
            CrateType::ProcMacro => {
                // We could embed NatVis for proc macro crates too (to improve the debugging
                // experience for them) but it does not seem like a good default, since
                // this is a rare use case and we don't want to slow down the common case.
                false
            }
            CrateType::Staticlib | CrateType::Rlib => {
                // We don't invoke the linker for these, so we don't need to collect the NatVis for them.
                false
            }
        });

        if target.is_like_msvc && embed_visualizers {
            info.natvis_debugger_visualizers =
                collect_debugger_visualizers_transitive(tcx, DebuggerVisualizerType::Natvis);
        }

        info
    }
}

pub fn provide(providers: &mut Providers) {
    providers.backend_optimization_level = |tcx, cratenum| {
        let for_speed = match tcx.sess.opts.optimize {
            // If globally no optimisation is done, #[optimize] has no effect.
            //
            // This is done because if we ended up "upgrading" to `-O2` here, we’d populate the
            // pass manager and it is likely that some module-wide passes (such as inliner or
            // cross-function constant propagation) would ignore the `optnone` annotation we put
            // on the functions, thus necessarily involving these functions into optimisations.
            config::OptLevel::No => return config::OptLevel::No,
            // If globally optimise-speed is already specified, just use that level.
            config::OptLevel::Less => return config::OptLevel::Less,
            config::OptLevel::Default => return config::OptLevel::Default,
            config::OptLevel::Aggressive => return config::OptLevel::Aggressive,
            // If globally optimize-for-size has been requested, use -O2 instead (if optimize(size)
            // are present).
            config::OptLevel::Size => config::OptLevel::Default,
            config::OptLevel::SizeMin => config::OptLevel::Default,
        };

        let (defids, _) = tcx.collect_and_partition_mono_items(cratenum);

        let any_for_speed = defids.items().any(|id| {
            let CodegenFnAttrs { optimize, .. } = tcx.codegen_fn_attrs(*id);
            match optimize {
                attr::OptimizeAttr::None | attr::OptimizeAttr::Size => false,
                attr::OptimizeAttr::Speed => true,
            }
        });

        if any_for_speed {
            return for_speed;
        }

        tcx.sess.opts.optimize
    };
}

fn determine_cgu_reuse<'tcx>(tcx: TyCtxt<'tcx>, cgu: &CodegenUnit<'tcx>) -> CguReuse {
    if !tcx.dep_graph.is_fully_enabled() {
        return CguReuse::No;
    }

    let work_product_id = &cgu.work_product_id();
    if tcx.dep_graph.previous_work_product(work_product_id).is_none() {
        // We don't have anything cached for this CGU. This can happen
        // if the CGU did not exist in the previous session.
        return CguReuse::No;
    }

    // Try to mark the CGU as green. If it we can do so, it means that nothing
    // affecting the LLVM module has changed and we can re-use a cached version.
    // If we compile with any kind of LTO, this means we can re-use the bitcode
    // of the Pre-LTO stage (possibly also the Post-LTO version but we'll only
    // know that later). If we are not doing LTO, there is only one optimized
    // version of each module, so we re-use that.
    let dep_node = cgu.codegen_dep_node(tcx);
    assert!(
        !tcx.dep_graph.dep_node_exists(&dep_node),
        "CompileCodegenUnit dep-node for CGU `{}` already exists before marking.",
        cgu.name()
    );

    if tcx.try_mark_green(&dep_node) {
        // We can re-use either the pre- or the post-thinlto state. If no LTO is
        // being performed then we can use post-LTO artifacts, otherwise we must
        // reuse pre-LTO artifacts
        match compute_per_cgu_lto_type(
            &tcx.sess.lto(),
            &tcx.sess.opts,
            &tcx.sess.crate_types(),
            ModuleKind::Regular,
        ) {
            ComputedLtoType::No => CguReuse::PostLto,
            _ => CguReuse::PreLto,
        }
    } else {
        CguReuse::No
    }
}

use std::cmp;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use itertools::Itertools;
use rustc_abi::FIRST_VARIANT;
use rustc_ast as ast;
use rustc_ast::expand::allocator::{ALLOCATOR_METHODS, AllocatorKind, global_fn_name};
use rustc_attr_data_structures::OptimizeAttr;
use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::profiling::{get_resident_set_size, print_time_passes_entry};
use rustc_data_structures::sync::{IntoDynSyncSend, par_map};
use rustc_data_structures::unord::UnordMap;
use rustc_hir::ItemId;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::lang_items::LangItem;
use rustc_metadata::EncodedMetadata;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::middle::debugger_visualizer::{DebuggerVisualizerFile, DebuggerVisualizerType};
use rustc_middle::middle::exported_symbols::SymbolExportKind;
use rustc_middle::middle::{exported_symbols, lang_items};
use rustc_middle::mir::BinOp;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::mir::mono::{CodegenUnit, CodegenUnitNameBuilder, MonoItem, MonoItemPartitions};
use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::Session;
use rustc_session::config::{self, CrateType, EntryFnType, OutputType};
use rustc_span::{DUMMY_SP, Symbol, sym};
use rustc_symbol_mangling::mangle_internal_symbol;
use rustc_trait_selection::infer::{BoundRegionConversionTime, TyCtxtInferExt};
use rustc_trait_selection::traits::{ObligationCause, ObligationCtxt};
use tracing::{debug, info};

use crate::assert_module_sources::CguReuse;
use crate::back::link::are_upstream_rust_objects_already_included;
use crate::back::metadata::create_compressed_metadata_file;
use crate::back::write::{
    ComputedLtoType, OngoingCodegen, compute_per_cgu_lto_type, start_async_codegen,
    submit_codegened_module_to_llvm, submit_post_lto_module_to_llvm, submit_pre_lto_module_to_llvm,
};
use crate::common::{self, IntPredicate, RealPredicate, TypeKind};
use crate::meth::load_vtable;
use crate::mir::operand::OperandValue;
use crate::mir::place::PlaceRef;
use crate::traits::*;
use crate::{
    CachedModuleCodegen, CodegenLintLevels, CompiledModule, CrateInfo, ModuleCodegen, ModuleKind,
    errors, meth, mir,
};

pub(crate) fn bin_op_to_icmp_predicate(op: BinOp, signed: bool) -> IntPredicate {
    match (op, signed) {
        (BinOp::Eq, _) => IntPredicate::IntEQ,
        (BinOp::Ne, _) => IntPredicate::IntNE,
        (BinOp::Lt, true) => IntPredicate::IntSLT,
        (BinOp::Lt, false) => IntPredicate::IntULT,
        (BinOp::Le, true) => IntPredicate::IntSLE,
        (BinOp::Le, false) => IntPredicate::IntULE,
        (BinOp::Gt, true) => IntPredicate::IntSGT,
        (BinOp::Gt, false) => IntPredicate::IntUGT,
        (BinOp::Ge, true) => IntPredicate::IntSGE,
        (BinOp::Ge, false) => IntPredicate::IntUGE,
        op => bug!("bin_op_to_icmp_predicate: expected comparison operator, found {:?}", op),
    }
}

pub(crate) fn bin_op_to_fcmp_predicate(op: BinOp) -> RealPredicate {
    match op {
        BinOp::Eq => RealPredicate::RealOEQ,
        BinOp::Ne => RealPredicate::RealUNE,
        BinOp::Lt => RealPredicate::RealOLT,
        BinOp::Le => RealPredicate::RealOLE,
        BinOp::Gt => RealPredicate::RealOGT,
        BinOp::Ge => RealPredicate::RealOGE,
        op => bug!("bin_op_to_fcmp_predicate: expected comparison operator, found {:?}", op),
    }
}

pub fn compare_simd_types<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    lhs: Bx::Value,
    rhs: Bx::Value,
    t: Ty<'tcx>,
    ret_ty: Bx::Type,
    op: BinOp,
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

/// Codegen takes advantage of the additional assumption, where if the
/// principal trait def id of what's being casted doesn't change,
/// then we don't need to adjust the vtable at all. This
/// corresponds to the fact that `dyn Tr<A>: Unsize<dyn Tr<B>>`
/// requires that `A = B`; we don't allow *upcasting* objects
/// between the same trait with different args. If we, for
/// some reason, were to relax the `Unsize` trait, it could become
/// unsound, so let's validate here that the trait refs are subtypes.
pub fn validate_trivial_unsize<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    target_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> bool {
    match (source_data.principal(), target_data.principal()) {
        (Some(hr_source_principal), Some(hr_target_principal)) => {
            let (infcx, param_env) =
                tcx.infer_ctxt().build_with_typing_env(ty::TypingEnv::fully_monomorphized());
            let universe = infcx.universe();
            let ocx = ObligationCtxt::new(&infcx);
            infcx.enter_forall(hr_target_principal, |target_principal| {
                let source_principal = infcx.instantiate_binder_with_fresh_vars(
                    DUMMY_SP,
                    BoundRegionConversionTime::HigherRankedType,
                    hr_source_principal,
                );
                let Ok(()) = ocx.eq(
                    &ObligationCause::dummy(),
                    param_env,
                    target_principal,
                    source_principal,
                ) else {
                    return false;
                };
                if !ocx.select_all_or_error().is_empty() {
                    return false;
                }
                infcx.leak_check(universe, None).is_ok()
            })
        }
        (_, None) => true,
        _ => false,
    }
}

/// Retrieves the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit odd. It is intended for use in an upcast,
/// where the new vtable for an object will be derived from the old one.
fn unsized_info<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
    old_info: Option<Bx::Value>,
) -> Bx::Value {
    let cx = bx.cx();
    let (source, target) =
        cx.tcx().struct_lockstep_tails_for_codegen(source, target, bx.typing_env());
    match (source.kind(), target.kind()) {
        (&ty::Array(_, len), &ty::Slice(_)) => cx.const_usize(
            len.try_to_target_usize(cx.tcx()).expect("expected monomorphic const in codegen"),
        ),
        (&ty::Dynamic(data_a, _, src_dyn_kind), &ty::Dynamic(data_b, _, target_dyn_kind))
            if src_dyn_kind == target_dyn_kind =>
        {
            let old_info =
                old_info.expect("unsized_info: missing old info for trait upcasting coercion");
            let b_principal_def_id = data_b.principal_def_id();
            if data_a.principal_def_id() == b_principal_def_id || b_principal_def_id.is_none() {
                // Codegen takes advantage of the additional assumption, where if the
                // principal trait def id of what's being casted doesn't change,
                // then we don't need to adjust the vtable at all. This
                // corresponds to the fact that `dyn Tr<A>: Unsize<dyn Tr<B>>`
                // requires that `A = B`; we don't allow *upcasting* objects
                // between the same trait with different args. If we, for
                // some reason, were to relax the `Unsize` trait, it could become
                // unsound, so let's assert here that the trait refs are *equal*.
                debug_assert!(
                    validate_trivial_unsize(cx.tcx(), data_a, data_b),
                    "NOP unsize vtable changed principal trait ref: {data_a} -> {data_b}"
                );

                // A NOP cast that doesn't actually change anything, let's avoid any
                // unnecessary work. This relies on the assumption that if the principal
                // traits are equal, then the associated type bounds (`dyn Trait<Assoc=T>`)
                // are also equal, which is ensured by the fact that normalization is
                // a function and we do not allow overlapping impls.
                return old_info;
            }

            // trait upcasting coercion

            let vptr_entry_idx = cx.tcx().supertrait_vtable_slot((source, target));

            if let Some(entry_idx) = vptr_entry_idx {
                let ptr_size = bx.data_layout().pointer_size;
                let vtable_byte_offset = u64::try_from(entry_idx).unwrap() * ptr_size.bytes();
                load_vtable(bx, old_info, bx.type_ptr(), vtable_byte_offset, source, true)
            } else {
                old_info
            }
        }
        (_, ty::Dynamic(data, _, _)) => meth::get_vtable(
            cx,
            source,
            data.principal()
                .map(|principal| bx.tcx().instantiate_bound_regions_with_erased(principal)),
        ),
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}", source, target),
    }
}

/// Coerces `src` to `dst_ty`. `src_ty` must be a pointer.
pub(crate) fn unsize_ptr<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    src: Bx::Value,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>,
    old_info: Option<Bx::Value>,
) -> (Bx::Value, Bx::Value) {
    debug!("unsize_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (src_ty.kind(), dst_ty.kind()) {
        (&ty::Ref(_, a, _), &ty::Ref(_, b, _) | &ty::RawPtr(b, _))
        | (&ty::RawPtr(a, _), &ty::RawPtr(b, _)) => {
            assert_eq!(bx.cx().type_is_sized(a), old_info.is_none());
            (src, unsized_info(bx, a, b, old_info))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b); // implies same number of fields
            let src_layout = bx.cx().layout_of(src_ty);
            let dst_layout = bx.cx().layout_of(dst_ty);
            if src_ty == dst_ty {
                return (src, old_info.unwrap());
            }
            let mut result = None;
            for i in 0..src_layout.fields.count() {
                let src_f = src_layout.field(bx.cx(), i);
                if src_f.is_1zst() {
                    // We are looking for the one non-1-ZST field; this is not it.
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
            result.unwrap()
        }
        _ => bug!("unsize_ptr: called on bad types"),
    }
}

/// Coerces `src` to `dst_ty` which is guaranteed to be a `dyn*` type.
pub(crate) fn cast_to_dyn_star<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
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
    let src = match bx.cx().type_kind(bx.cx().backend_type(src_ty_and_layout)) {
        TypeKind::Pointer => src,
        TypeKind::Integer => bx.inttoptr(src, bx.type_ptr()),
        // FIXME(dyn-star): We probably have to do a bitcast first, then inttoptr.
        kind => bug!("unexpected TypeKind for left-hand side of `dyn*` cast: {kind:?}"),
    };
    (src, unsized_info(bx, src_ty_and_layout.ty, dst_ty, old_info))
}

/// Coerces `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty`, and stores the result in `dst`.
pub(crate) fn coerce_unsized_into<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
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
            assert_eq!(def_a, def_b); // implies same number of fields

            for i in def_a.variant(FIRST_VARIANT).fields.indices() {
                let src_f = src.project_field(bx, i.as_usize());
                let dst_f = dst.project_field(bx, i.as_usize());

                if dst_f.layout.is_zst() {
                    // No data here, nothing to copy/coerce.
                    continue;
                }

                if src_f.layout.ty == dst_f.layout.ty {
                    bx.typed_place_copy(dst_f.val, src_f.val, src_f.layout);
                } else {
                    coerce_unsized_into(bx, src_f, dst_f);
                }
            }
        }
        _ => bug!("coerce_unsized_into: invalid coercion {:?} -> {:?}", src_ty, dst_ty,),
    }
}

/// Returns `rhs` sufficiently masked, truncated, and/or extended so that it can be used to shift
/// `lhs`: it has the same size as `lhs`, and the value, when interpreted unsigned (no matter its
/// type), will not exceed the size of `lhs`.
///
/// Shifts in MIR are all allowed to have mismatched LHS & RHS types, and signed RHS.
/// The shift methods in `BuilderMethods`, however, are fully homogeneous
/// (both parameters and the return type are all the same size) and assume an unsigned RHS.
///
/// If `is_unchecked` is false, this masks the RHS to ensure it stays in-bounds,
/// as the `BuilderMethods` shifts are UB for out-of-bounds shift amounts.
/// For 32- and 64-bit types, this matches the semantics
/// of Java. (See related discussion on #1877 and #10183.)
///
/// If `is_unchecked` is true, this does no masking, and adds sufficient `assume`
/// calls or operation flags to preserve as much freedom to optimize as possible.
pub(crate) fn build_shift_expr_rhs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    lhs: Bx::Value,
    mut rhs: Bx::Value,
    is_unchecked: bool,
) -> Bx::Value {
    // Shifts may have any size int on the rhs
    let mut rhs_llty = bx.cx().val_ty(rhs);
    let mut lhs_llty = bx.cx().val_ty(lhs);

    let mask = common::shift_mask_val(bx, lhs_llty, rhs_llty, false);
    if !is_unchecked {
        rhs = bx.and(rhs, mask);
    }

    if bx.cx().type_kind(rhs_llty) == TypeKind::Vector {
        rhs_llty = bx.cx().element_type(rhs_llty)
    }
    if bx.cx().type_kind(lhs_llty) == TypeKind::Vector {
        lhs_llty = bx.cx().element_type(lhs_llty)
    }
    let rhs_sz = bx.cx().int_width(rhs_llty);
    let lhs_sz = bx.cx().int_width(lhs_llty);
    if lhs_sz < rhs_sz {
        if is_unchecked { bx.unchecked_utrunc(rhs, lhs_llty) } else { bx.trunc(rhs, lhs_llty) }
    } else if lhs_sz > rhs_sz {
        // We zero-extend even if the RHS is signed. So e.g. `(x: i32) << -1i8` will zero-extend the
        // RHS to `255i32`. But then we mask the shift amount to be within the size of the LHS
        // anyway so the result is `31` as it should be. All the extra bits introduced by zext
        // are masked off so their value does not matter.
        // FIXME: if we ever support 512bit integers, this will be wrong! For such large integers,
        // the extra bits introduced by zext are *not* all masked away any more.
        assert!(lhs_sz <= 256);
        bx.zext(rhs, lhs_llty)
    } else {
        rhs
    }
}

// Returns `true` if this session's target will use native wasm
// exceptions. This means that the VM does the unwinding for
// us
pub fn wants_wasm_eh(sess: &Session) -> bool {
    sess.target.is_like_wasm
        && (sess.target.os != "emscripten" || sess.opts.unstable_opts.emscripten_wasm_eh)
}

/// Returns `true` if this session's target will use SEH-based unwinding.
///
/// This is only true for MSVC targets, and even then the 64-bit MSVC target
/// currently uses SEH-ish unwinding with DWARF info tables to the side (same as
/// 64-bit MinGW) instead of "full SEH".
pub fn wants_msvc_seh(sess: &Session) -> bool {
    sess.target.is_like_msvc
}

/// Returns `true` if this session's target requires the new exception
/// handling LLVM IR instructions (catchpad / cleanuppad / ... instead
/// of landingpad)
pub(crate) fn wants_new_eh_instructions(sess: &Session) -> bool {
    wants_wasm_eh(sess) || wants_msvc_seh(sess)
}

pub(crate) fn codegen_instance<'a, 'tcx: 'a, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    instance: Instance<'tcx>,
) {
    // this is an info! to allow collecting monomorphization statistics
    // and to allow finding the last function before LLVM aborts from
    // release builds.
    info!("codegen_instance({})", instance);

    mir::codegen_mir::<Bx>(cx, instance);
}

pub fn codegen_global_asm<'tcx, Cx>(cx: &mut Cx, item_id: ItemId)
where
    Cx: LayoutOf<'tcx, LayoutOfResult = TyAndLayout<'tcx>> + AsmCodegenMethods<'tcx>,
{
    let item = cx.tcx().hir_item(item_id);
    if let rustc_hir::ItemKind::GlobalAsm { asm, .. } = item.kind {
        let operands: Vec<_> = asm
            .operands
            .iter()
            .map(|(op, op_sp)| match *op {
                rustc_hir::InlineAsmOperand::Const { ref anon_const } => {
                    match cx.tcx().const_eval_poly(anon_const.def_id.to_def_id()) {
                        Ok(const_value) => {
                            let ty =
                                cx.tcx().typeck_body(anon_const.body).node_type(anon_const.hir_id);
                            let string = common::asm_const_to_str(
                                cx.tcx(),
                                *op_sp,
                                const_value,
                                cx.layout_of(ty),
                            );
                            GlobalAsmOperandRef::Const { string }
                        }
                        Err(ErrorHandled::Reported { .. }) => {
                            // An error has already been reported and
                            // compilation is guaranteed to fail if execution
                            // hits this path. So an empty string instead of
                            // a stringified constant value will suffice.
                            GlobalAsmOperandRef::Const { string: String::new() }
                        }
                        Err(ErrorHandled::TooGeneric(_)) => {
                            span_bug!(*op_sp, "asm const cannot be resolved; too generic")
                        }
                    }
                }
                rustc_hir::InlineAsmOperand::SymFn { expr } => {
                    let ty = cx.tcx().typeck(item_id.owner_id).expr_ty(expr);
                    let instance = match ty.kind() {
                        &ty::FnDef(def_id, args) => Instance::expect_resolve(
                            cx.tcx(),
                            ty::TypingEnv::fully_monomorphized(),
                            def_id,
                            args,
                            expr.span,
                        ),
                        _ => span_bug!(*op_sp, "asm sym is not a function"),
                    };

                    GlobalAsmOperandRef::SymFn { instance }
                }
                rustc_hir::InlineAsmOperand::SymStatic { path: _, def_id } => {
                    GlobalAsmOperandRef::SymStatic { def_id }
                }
                rustc_hir::InlineAsmOperand::In { .. }
                | rustc_hir::InlineAsmOperand::Out { .. }
                | rustc_hir::InlineAsmOperand::InOut { .. }
                | rustc_hir::InlineAsmOperand::SplitInOut { .. }
                | rustc_hir::InlineAsmOperand::Label { .. } => {
                    span_bug!(*op_sp, "invalid operand type for global_asm!")
                }
            })
            .collect();

        cx.codegen_global_asm(asm.template, &operands, asm.options, asm.line_spans);
    } else {
        span_bug!(item.span, "Mismatch between hir::Item type and MonoItem type")
    }
}

/// Creates the `main` function which will initialize the rust runtime and call
/// users main function.
pub fn maybe_create_entry_wrapper<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    cgu: &CodegenUnit<'tcx>,
) -> Option<Bx::Function> {
    let (main_def_id, entry_type) = cx.tcx().entry_fn(())?;
    let main_is_local = main_def_id.is_local();
    let instance = Instance::mono(cx.tcx(), main_def_id);

    if main_is_local {
        // We want to create the wrapper in the same codegen unit as Rust's main
        // function.
        if !cgu.contains_item(&MonoItem::Fn(instance)) {
            return None;
        }
    } else if !cgu.is_primary() {
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
        // The entry function is either `int main(void)` or `int main(int argc, char **argv)`, or
        // `usize efi_main(void *handle, void *system_table)` depending on the target.
        let llfty = if cx.sess().target.os.contains("uefi") {
            cx.type_func(&[cx.type_ptr(), cx.type_ptr()], cx.type_isize())
        } else if cx.sess().target.main_needs_argc_argv {
            cx.type_func(&[cx.type_int(), cx.type_ptr()], cx.type_int())
        } else {
            cx.type_func(&[], cx.type_int())
        };

        let main_ret_ty = cx.tcx().fn_sig(rust_main_def_id).no_bound_vars().unwrap().output();
        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = cx
            .tcx()
            .normalize_erasing_regions(cx.typing_env(), main_ret_ty.no_bound_vars().unwrap());

        let Some(llfn) = cx.declare_c_main(llfty) else {
            // FIXME: We should be smart and show a better diagnostic here.
            let span = cx.tcx().def_span(rust_main_def_id);
            cx.tcx().dcx().emit_fatal(errors::MultipleMainFunctions { span });
        };

        // `main` should respect same config for frame pointer elimination as rest of code
        cx.set_frame_pointer_type(llfn);
        cx.apply_target_cpu_attr(llfn);

        let llbb = Bx::append_block(cx, llfn, "top");
        let mut bx = Bx::build(cx, llbb);

        bx.insert_reference_to_gdb_debug_scripts_section_global();

        let isize_ty = cx.type_isize();
        let ptr_ty = cx.type_ptr();
        let (arg_argc, arg_argv) = get_argc_argv(&mut bx);

        let EntryFnType::Main { sigpipe } = entry_type;
        let (start_fn, start_ty, args, instance) = {
            let start_def_id = cx.tcx().require_lang_item(LangItem::Start, DUMMY_SP);
            let start_instance = ty::Instance::expect_resolve(
                cx.tcx(),
                cx.typing_env(),
                start_def_id,
                cx.tcx().mk_args(&[main_ret_ty.into()]),
                DUMMY_SP,
            );
            let start_fn = cx.get_fn_addr(start_instance);

            let i8_ty = cx.type_i8();
            let arg_sigpipe = bx.const_u8(sigpipe);

            let start_ty = cx.type_func(&[cx.val_ty(rust_main), isize_ty, ptr_ty, i8_ty], isize_ty);
            (
                start_fn,
                start_ty,
                vec![rust_main, arg_argc, arg_argv, arg_sigpipe],
                Some(start_instance),
            )
        };

        let result = bx.call(start_ty, None, None, start_fn, &args, None, instance);
        if cx.sess().target.os.contains("uefi") {
            bx.ret(result);
        } else {
            let cast = bx.intcast(result, cx.type_int(), true);
            bx.ret(cast);
        }

        llfn
    }
}

/// Obtain the `argc` and `argv` values to pass to the rust start function
/// (i.e., the "start" lang item).
fn get_argc_argv<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(bx: &mut Bx) -> (Bx::Value, Bx::Value) {
    if bx.cx().sess().target.os.contains("uefi") {
        // Params for UEFI
        let param_handle = bx.get_param(0);
        let param_system_table = bx.get_param(1);
        let ptr_size = bx.tcx().data_layout.pointer_size;
        let ptr_align = bx.tcx().data_layout.pointer_align.abi;
        let arg_argc = bx.const_int(bx.cx().type_isize(), 2);
        let arg_argv = bx.alloca(2 * ptr_size, ptr_align);
        bx.store(param_handle, arg_argv, ptr_align);
        let arg_argv_el1 = bx.inbounds_ptradd(arg_argv, bx.const_usize(ptr_size.bytes()));
        bx.store(param_system_table, arg_argv_el1, ptr_align);
        (arg_argc, arg_argv)
    } else if bx.cx().sess().target.main_needs_argc_argv {
        // Params from native `main()` used as args for rust start function
        let param_argc = bx.get_param(0);
        let param_argv = bx.get_param(1);
        let arg_argc = bx.intcast(param_argc, bx.cx().type_isize(), true);
        let arg_argv = param_argv;
        (arg_argc, arg_argv)
    } else {
        // The Rust start function doesn't need `argc` and `argv`, so just pass zeros.
        let arg_argc = bx.const_int(bx.cx().type_int(), 0);
        let arg_argv = bx.const_null(bx.cx().type_ptr());
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

    if tcx.sess.target.need_explicit_cpu && tcx.sess.opts.cg.target_cpu.is_none() {
        // The target has no default cpu, but none is set explicitly
        tcx.dcx().emit_fatal(errors::CpuRequired);
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);

    // Run the monomorphization collector and partition the collected items into
    // codegen units.
    let MonoItemPartitions { codegen_units, autodiff_items, .. } =
        tcx.collect_and_partition_mono_items(());
    let autodiff_fncs = autodiff_items.to_vec();

    // Force all codegen_unit queries so they are already either red or green
    // when compile_codegen_unit accesses them. We are not able to re-execute
    // the codegen_unit query from just the DepNode, so an unknown color would
    // lead to having to re-execute compile_codegen_unit, possibly
    // unnecessarily.
    if tcx.dep_graph.is_fully_enabled() {
        for cgu in codegen_units {
            tcx.ensure_ok().codegen_unit(cgu.name());
        }
    }

    let metadata_module = need_metadata_module.then(|| {
        // Emit compressed metadata object.
        let metadata_cgu_name =
            cgu_name_builder.build_cgu_name(LOCAL_CRATE, &["crate"], Some("metadata")).to_string();
        tcx.sess.time("write_compressed_metadata", || {
            let file_name = tcx.output_filenames(()).temp_path_for_cgu(
                OutputType::Metadata,
                &metadata_cgu_name,
                tcx.sess.invocation_temp.as_deref(),
            );
            let data = create_compressed_metadata_file(
                tcx.sess,
                &metadata,
                &exported_symbols::metadata_symbol_name(tcx),
            );
            if let Err(error) = std::fs::write(&file_name, data) {
                tcx.dcx().emit_fatal(errors::MetadataObjectFileWrite { error });
            }
            CompiledModule {
                name: metadata_cgu_name,
                kind: ModuleKind::Metadata,
                object: Some(file_name),
                dwarf_object: None,
                bytecode: None,
                assembly: None,
                llvm_ir: None,
                links_from_incr_cache: Vec::new(),
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

        ongoing_codegen.wait_for_signal_to_codegen_item();
        ongoing_codegen.check_for_errors(tcx.sess);

        // These modules are generally cheap and won't throw off scheduling.
        let cost = 0;
        submit_codegened_module_to_llvm(
            &backend,
            &ongoing_codegen.coordinator.sender,
            ModuleCodegen::new_allocator(llmod_id, module_llvm),
            cost,
        );
    }

    if !autodiff_fncs.is_empty() {
        ongoing_codegen.submit_autodiff_items(autodiff_fncs);
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
        sorted_cgus.sort_by_key(|cgu| cmp::Reverse(cgu.size_estimate()));

        let (first_half, second_half) = sorted_cgus.split_at(sorted_cgus.len() / 2);
        first_half.iter().interleave(second_half.iter().rev()).copied().collect()
    };

    // Calculate the CGU reuse
    let cgu_reuse = tcx.sess.time("find_cgu_reuse", || {
        codegen_units.iter().map(|cgu| determine_cgu_reuse(tcx, cgu)).collect::<Vec<_>>()
    });

    crate::assert_module_sources::assert_module_sources(tcx, &|cgu_reuse_tracker| {
        for (i, cgu) in codegen_units.iter().enumerate() {
            let cgu_reuse = cgu_reuse[i];
            cgu_reuse_tracker.set_actual_reuse(cgu.name().as_str(), cgu_reuse);
        }
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
                (i, IntoDynSyncSend(module))
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

        match cgu_reuse {
            CguReuse::No => {
                let (module, cost) = if let Some(cgu) = pre_compiled_cgus.remove(&i) {
                    cgu.0
                } else {
                    let start_time = Instant::now();
                    let module = backend.compile_codegen_unit(tcx, cgu.name());
                    total_codegen_time += start_time.elapsed();
                    module
                };
                // This will unwind if there are errors, which triggers our `AbortCodegenOnDrop`
                // guard. Unfortunately, just skipping the `submit_codegened_module_to_llvm` makes
                // compilation hang on post-monomorphization errors.
                tcx.dcx().abort_if_errors();

                submit_codegened_module_to_llvm(
                    &backend,
                    &ongoing_codegen.coordinator.sender,
                    module,
                    cost,
                );
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
            }
        }
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

/// Returns whether a call from the current crate to the [`Instance`] would produce a call
/// from `compiler_builtins` to a symbol the linker must resolve.
///
/// Such calls from `compiler_bultins` are effectively impossible for the linker to handle. Some
/// linkers will optimize such that dead calls to unresolved symbols are not an error, but this is
/// not guaranteed. So we used this function in codegen backends to ensure we do not generate any
/// unlinkable calls.
///
/// Note that calls to LLVM intrinsics are uniquely okay because they won't make it to the linker.
pub fn is_call_from_compiler_builtins_to_upstream_monomorphization<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    fn is_llvm_intrinsic(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        if let Some(name) = tcx.codegen_fn_attrs(def_id).link_name {
            name.as_str().starts_with("llvm.")
        } else {
            false
        }
    }

    let def_id = instance.def_id();
    !def_id.is_local()
        && tcx.is_compiler_builtins(LOCAL_CRATE)
        && !is_llvm_intrinsic(tcx, def_id)
        && !tcx.should_codegen_locally(instance)
}

impl CrateInfo {
    pub fn new(tcx: TyCtxt<'_>, target_cpu: String) -> CrateInfo {
        let crate_types = tcx.crate_types().to_vec();
        let exported_symbols = crate_types
            .iter()
            .map(|&c| (c, crate::back::linker::exported_symbols(tcx, c)))
            .collect();
        let linked_symbols =
            crate_types.iter().map(|&c| (c, crate::back::linker::linked_symbols(tcx, c))).collect();
        let local_crate_name = tcx.crate_name(LOCAL_CRATE);
        let crate_attrs = tcx.hir_attrs(rustc_hir::CRATE_HIR_ID);
        let subsystem =
            ast::attr::first_attr_value_str_by_name(crate_attrs, sym::windows_subsystem);
        let windows_subsystem = subsystem.map(|subsystem| {
            if subsystem != sym::windows && subsystem != sym::console {
                tcx.dcx().emit_fatal(errors::InvalidWindowsSubsystem { subsystem });
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
        // postorder of all crates putting the leaves at the rightmost positions.
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

        let crates = tcx.crates(());
        let n_crates = crates.len();
        let mut info = CrateInfo {
            target_cpu,
            target_features: tcx.global_backend_features(()).clone(),
            crate_types,
            exported_symbols,
            linked_symbols,
            local_crate_name,
            compiler_builtins,
            profiler_runtime: None,
            is_no_builtins: Default::default(),
            native_libraries: Default::default(),
            used_libraries: tcx.native_libraries(LOCAL_CRATE).iter().map(Into::into).collect(),
            crate_name: UnordMap::with_capacity(n_crates),
            used_crates,
            used_crate_source: UnordMap::with_capacity(n_crates),
            dependency_formats: Arc::clone(tcx.dependency_formats(())),
            windows_subsystem,
            natvis_debugger_visualizers: Default::default(),
            lint_levels: CodegenLintLevels::from_tcx(tcx),
        };

        info.native_libraries.reserve(n_crates);

        for &cnum in crates.iter() {
            info.native_libraries
                .insert(cnum, tcx.native_libraries(cnum).iter().map(Into::into).collect());
            info.crate_name.insert(cnum, tcx.crate_name(cnum));

            let used_crate_source = tcx.used_crate_source(cnum);
            info.used_crate_source.insert(cnum, Arc::clone(used_crate_source));
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
            let missing_weak_lang_items: FxIndexSet<Symbol> = info
                .used_crates
                .iter()
                .flat_map(|&cnum| tcx.missing_lang_items(cnum))
                .filter(|l| l.is_weak())
                .filter_map(|&l| {
                    let name = l.link_name()?;
                    lang_items::required(tcx, l).then_some(name)
                })
                .collect();
            let prefix = match (target.is_like_windows, target.arch.as_ref()) {
                (true, "x86") => "_",
                (true, "arm64ec") => "#",
                _ => "",
            };

            // This loop only adds new items to values of the hash map, so the order in which we
            // iterate over the values is not important.
            #[allow(rustc::potential_query_instability)]
            info.linked_symbols
                .iter_mut()
                .filter(|(crate_type, _)| {
                    !matches!(crate_type, CrateType::Rlib | CrateType::Staticlib)
                })
                .for_each(|(_, linked_symbols)| {
                    let mut symbols = missing_weak_lang_items
                        .iter()
                        .map(|item| {
                            (
                                format!("{prefix}{}", mangle_internal_symbol(tcx, item.as_str())),
                                SymbolExportKind::Text,
                            )
                        })
                        .collect::<Vec<_>>();
                    symbols.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    linked_symbols.extend(symbols);
                    if tcx.allocator_kind(()).is_some() {
                        // At least one crate needs a global allocator. This crate may be placed
                        // after the crate that defines it in the linker order, in which case some
                        // linkers return an error. By adding the global allocator shim methods to
                        // the linked_symbols list, linking the generated symbols.o will ensure that
                        // circular dependencies involving the global allocator don't lead to linker
                        // errors.
                        linked_symbols.extend(ALLOCATOR_METHODS.iter().map(|method| {
                            (
                                format!(
                                    "{prefix}{}",
                                    mangle_internal_symbol(
                                        tcx,
                                        global_fn_name(method.name).as_str()
                                    )
                                ),
                                SymbolExportKind::Text,
                            )
                        }));
                    }
                });
        }

        let embed_visualizers = tcx.crate_types().iter().any(|&crate_type| match crate_type {
            CrateType::Executable | CrateType::Dylib | CrateType::Cdylib | CrateType::Sdylib => {
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
                // We don't invoke the linker for these, so we don't need to collect the NatVis for
                // them.
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

pub(crate) fn provide(providers: &mut Providers) {
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
            config::OptLevel::More => return config::OptLevel::More,
            config::OptLevel::Aggressive => return config::OptLevel::Aggressive,
            // If globally optimize-for-size has been requested, use -O2 instead (if optimize(size)
            // are present).
            config::OptLevel::Size => config::OptLevel::More,
            config::OptLevel::SizeMin => config::OptLevel::More,
        };

        let defids = tcx.collect_and_partition_mono_items(cratenum).all_mono_items;

        let any_for_speed = defids.items().any(|id| {
            let CodegenFnAttrs { optimize, .. } = tcx.codegen_fn_attrs(*id);
            matches!(optimize, OptimizeAttr::Speed)
        });

        if any_for_speed {
            return for_speed;
        }

        tcx.sess.opts.optimize
    };
}

pub fn determine_cgu_reuse<'tcx>(tcx: TyCtxt<'tcx>, cgu: &CodegenUnit<'tcx>) -> CguReuse {
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
    tcx.dep_graph.assert_dep_node_not_yet_allocated_in_current_session(&dep_node, || {
        format!(
            "CompileCodegenUnit dep-node for CGU `{}` already exists before marking.",
            cgu.name()
        )
    });

    if tcx.try_mark_green(&dep_node) {
        // We can re-use either the pre- or the post-thinlto state. If no LTO is
        // being performed then we can use post-LTO artifacts, otherwise we must
        // reuse pre-LTO artifacts
        match compute_per_cgu_lto_type(
            &tcx.sess.lto(),
            &tcx.sess.opts,
            tcx.crate_types(),
            ModuleKind::Regular,
        ) {
            ComputedLtoType::No => CguReuse::PostLto,
            _ => CguReuse::PreLto,
        }
    } else {
        CguReuse::No
    }
}

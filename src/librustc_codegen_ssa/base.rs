//! Codegen the completed AST to the LLVM IR.
//!
//! Some functions here, such as codegen_block and codegen_expr, return a value --
//! the result of the codegen to LLVM -- while others, such as codegen_fn
//! and mono_item, are called only for the side effect of adding a
//! particular definition to the LLVM IR output we're producing.
//!
//! Hopefully useful general knowledge about codegen:
//!
//! * There's no way to find out the `Ty` type of a Value. Doing so
//!   would be "trying to get the eggs out of an omelette" (credit:
//!   pcwalton). You can, instead, find out its `llvm::Type` by calling `val_ty`,
//!   but one `llvm::Type` corresponds to many `Ty`s; for instance, `tup(int, int,
//!   int)` and `rec(x=int, y=int, z=int)` will have the same `llvm::Type`.

use crate::{ModuleCodegen, ModuleKind, CachedModuleCodegen};

use rustc::dep_graph::cgu_reuse_tracker::CguReuse;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::middle::cstore::EncodedMetadata;
use rustc::middle::lang_items::StartFnLangItem;
use rustc::middle::weak_lang_items;
use rustc::mir::mono::{CodegenUnitNameBuilder, CodegenUnit, MonoItem};
use rustc::ty::{self, Ty, TyCtxt, Instance};
use rustc::ty::layout::{self, Align, TyLayout, LayoutOf, VariantIdx, HasTyCtxt};
use rustc::ty::query::Providers;
use rustc::middle::cstore::{self, LinkagePreference};
use rustc::util::common::{time, print_time_passes_entry};
use rustc::session::config::{self, EntryFnType, Lto};
use rustc::session::Session;
use rustc::util::nodemap::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use rustc_codegen_utils::{symbol_names_test, check_for_rustc_errors_attr};
use rustc::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};
use crate::mir::place::PlaceRef;
use crate::back::write::{OngoingCodegen, start_async_codegen, submit_pre_lto_module_to_llvm,
    submit_post_lto_module_to_llvm};
use crate::{MemFlags, CrateInfo};
use crate::callee;
use crate::common::{RealPredicate, TypeKind, IntPredicate};
use crate::meth;
use crate::mir;

use crate::traits::*;

use std::any::Any;
use std::cmp;
use std::ops::{Deref, DerefMut};
use std::time::{Instant, Duration};
use std::sync::mpsc;
use syntax_pos::Span;
use syntax::attr;
use rustc::hir;

use crate::mir::operand::OperandValue;

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

pub fn compare_simd_types<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    lhs: Bx::Value,
    rhs: Bx::Value,
    t: Ty<'tcx>,
    ret_ty: Bx::Type,
    op: hir::BinOpKind,
) -> Bx::Value {
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

/// Retrieves the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit funny. It is intended for use
/// in an upcast, where the new vtable for an object will be derived
/// from the old one.
pub fn unsized_info<'tcx, Cx: CodegenMethods<'tcx>>(
    cx: &Cx,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
    old_info: Option<Cx::Value>,
) -> Cx::Value {
    let (source, target) = cx.tcx().struct_lockstep_tails(source, target);
    match (&source.sty, &target.sty) {
        (&ty::Array(_, len), &ty::Slice(_)) => {
            cx.const_usize(len.unwrap_usize(cx.tcx()))
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
            cx.const_ptrcast(meth::get_vtable(cx, source, data.principal()),
                            cx.backend_type(vtable_ptr))
        }
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}",
                  source,
                  target),
    }
}

/// Coerce `src` to `dst_ty`. `src_ty` must be a thin pointer.
pub fn unsize_thin_ptr<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    src: Bx::Value,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>,
) -> (Bx::Value, Bx::Value) {
    debug!("unsize_thin_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (&src_ty.sty, &dst_ty.sty) {
        (&ty::Ref(_, a, _),
         &ty::Ref(_, b, _)) |
        (&ty::Ref(_, a, _),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
            assert!(bx.cx().type_is_sized(a));
            let ptr_ty = bx.cx().type_ptr_to(bx.cx().backend_type(bx.cx().layout_of(b)));
            (bx.pointercast(src, ptr_ty), unsized_info(bx.cx(), a, b, None))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            let (a, b) = (src_ty.boxed_ty(), dst_ty.boxed_ty());
            assert!(bx.cx().type_is_sized(a));
            let ptr_ty = bx.cx().type_ptr_to(bx.cx().backend_type(bx.cx().layout_of(b)));
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
            (bx.bitcast(lldata, bx.cx().scalar_pair_element_backend_type(dst_layout, 0, true)),
             bx.bitcast(llextra, bx.cx().scalar_pair_element_backend_type(dst_layout, 1, true)))
        }
        _ => bug!("unsize_thin_ptr: called on bad types"),
    }
}

/// Coerce `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty` and store the result in `dst`
pub fn coerce_unsized_into<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    src: PlaceRef<'tcx, Bx::Value>,
    dst: PlaceRef<'tcx, Bx::Value>,
) {
    let src_ty = src.layout.ty;
    let dst_ty = dst.layout.ty;
    let mut coerce_ptr = || {
        let (base, info) = match bx.load_operand(src).val {
            OperandValue::Pair(base, info) => {
                // fat-ptr to fat-ptr unsize preserves the vtable
                // i.e., &'a fmt::Debug+Send => &'a fmt::Debug
                // So we need to pointercast the base to ensure
                // the types match up.
                let thin_ptr = dst.layout.field(bx.cx(), FAT_PTR_ADDR);
                (bx.pointercast(base, bx.cx().backend_type(thin_ptr)), info)
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

pub fn cast_shift_expr_rhs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    op: hir::BinOpKind,
    lhs: Bx::Value,
    rhs: Bx::Value,
) -> Bx::Value {
    cast_shift_rhs(bx, op, lhs, rhs)
}

fn cast_shift_rhs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    op: hir::BinOpKind,
    lhs: Bx::Value,
    rhs: Bx::Value,
) -> Bx::Value {
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
            // FIXME (#1877: If in the future shifting by negative
            // values is no longer undefined then this is wrong.
            bx.zext(rhs, lhs_llty)
        } else {
            rhs
        }
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
    sess.target.target.options.is_like_msvc
}

pub fn from_immediate<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    val: Bx::Value,
) -> Bx::Value {
    if bx.cx().val_ty(val) == bx.cx().type_i1() {
        bx.zext(val, bx.cx().type_i8())
    } else {
        val
    }
}

pub fn to_immediate<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    val: Bx::Value,
    layout: layout::TyLayout<'_>,
) -> Bx::Value {
    if let layout::Abi::Scalar(ref scalar) = layout.abi {
        return to_immediate_scalar(bx, val, scalar);
    }
    val
}

pub fn to_immediate_scalar<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    val: Bx::Value,
    scalar: &layout::Scalar,
) -> Bx::Value {
    if scalar.is_bool() {
        return bx.trunc(val, bx.cx().type_i1());
    }
    val
}

pub fn memcpy_ty<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    dst: Bx::Value,
    dst_align: Align,
    src: Bx::Value,
    src_align: Align,
    layout: TyLayout<'tcx>,
    flags: MemFlags,
) {
    let size = layout.size.bytes();
    if size == 0 {
        return;
    }

    bx.memcpy(dst, dst_align, src, src_align, bx.cx().const_usize(size), flags);
}

pub fn codegen_instance<'a, 'tcx: 'a, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    instance: Instance<'tcx>,
) {
    // this is an info! to allow collecting monomorphization statistics
    // and to allow finding the last function before LLVM aborts from
    // release builds.
    info!("codegen_instance({})", instance);

    let sig = instance.fn_sig(cx.tcx());
    let sig = cx.tcx().normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);

    let lldecl = cx.instances().borrow().get(&instance).cloned().unwrap_or_else(||
        bug!("Instance `{:?}` not already declared", instance));

    let mir = cx.tcx().instance_mir(instance.def);
    mir::codegen_mir::<Bx>(cx, lldecl, &mir, instance, sig);
}

/// Creates the `main` function which will initialize the rust runtime and call
/// users main function.
pub fn maybe_create_entry_wrapper<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(cx: &'a Bx::CodegenCx) {
    let (main_def_id, span) = match cx.tcx().entry_fn(LOCAL_CRATE) {
        Some((def_id, _)) => { (def_id, cx.tcx().def_span(def_id)) },
        None => return,
    };

    let instance = Instance::mono(cx.tcx(), main_def_id);

    if !cx.codegen_unit().contains_item(&MonoItem::Fn(instance)) {
        // We want to create the wrapper in the same codegen unit as Rust's main
        // function.
        return;
    }

    let main_llfn = cx.get_fn(instance);

    let et = cx.tcx().entry_fn(LOCAL_CRATE).map(|e| e.1);
    match et {
        Some(EntryFnType::Main) => create_entry_fn::<Bx>(cx, span, main_llfn, main_def_id, true),
        Some(EntryFnType::Start) => create_entry_fn::<Bx>(cx, span, main_llfn, main_def_id, false),
        None => {}    // Do nothing.
    }

    fn create_entry_fn<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        cx: &'a Bx::CodegenCx,
        sp: Span,
        rust_main: Bx::Value,
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
            &main_ret_ty.no_bound_vars().unwrap(),
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
        let param_argc = bx.get_param(0);
        let param_argv = bx.get_param(1);
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

pub fn codegen_crate<B: ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'tcx>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
    rx: mpsc::Receiver<Box<dyn Any + Send>>,
) -> OngoingCodegen<B> {
    check_for_rustc_errors_attr(tcx);

    // Skip crate items and just output metadata in -Z no-codegen mode.
    if tcx.sess.opts.debugging_opts.no_codegen ||
       !tcx.sess.opts.output_types.should_codegen() {
        let ongoing_codegen = start_async_codegen(
            backend,
            tcx,
            metadata,
            rx,
            1);

        ongoing_codegen.codegen_finished(tcx);

        assert_and_save_dep_graph(tcx);

        ongoing_codegen.check_for_errors(tcx.sess);

        return ongoing_codegen;
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);

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

    let ongoing_codegen = start_async_codegen(
        backend.clone(),
        tcx,
        metadata,
        rx,
        codegen_units.len());
    let ongoing_codegen = AbortCodegenOnDrop::<B>(Some(ongoing_codegen));

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
        let mut modules = backend.new_metadata(tcx, &llmod_id);
        time(tcx.sess, "write allocator module", || {
            backend.codegen_allocator(tcx, &mut modules, kind)
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

    if need_metadata_module {
        // Codegen the encoded metadata.
        tcx.sess.profiler(|p| p.start_activity("codegen crate metadata"));

        let metadata_cgu_name = cgu_name_builder.build_cgu_name(LOCAL_CRATE,
                                                                &["crate"],
                                                                Some("metadata")).as_str()
                                                                                 .to_string();
        let mut metadata_llvm_module = backend.new_metadata(tcx, &metadata_cgu_name);
        time(tcx.sess, "write compressed metadata", || {
            backend.write_compressed_metadata(tcx, &ongoing_codegen.metadata,
                                              &mut metadata_llvm_module);
        });
        tcx.sess.profiler(|p| p.end_activity("codegen crate metadata"));

        let metadata_module = ModuleCodegen {
            name: metadata_cgu_name,
            module_llvm: metadata_llvm_module,
            kind: ModuleKind::Metadata,
        };
        ongoing_codegen.submit_pre_codegened_module_to_llvm(tcx, metadata_module);
    }

    // We sort the codegen units by size. This way we can schedule work for LLVM
    // a bit more efficiently.
    let codegen_units = {
        let mut codegen_units = codegen_units;
        codegen_units.sort_by_cached_key(|cgu| cmp::Reverse(cgu.size_estimate()));
        codegen_units
    };

    let mut total_codegen_time = Duration::new(0, 0);

    for cgu in codegen_units.into_iter() {
        ongoing_codegen.wait_for_signal_to_codegen_item();
        ongoing_codegen.check_for_errors(tcx.sess);

        let cgu_reuse = determine_cgu_reuse(tcx, &cgu);
        tcx.sess.cgu_reuse_tracker.set_actual_reuse(&cgu.name().as_str(), cgu_reuse);

        match cgu_reuse {
            CguReuse::No => {
                tcx.sess.profiler(|p| p.start_activity(format!("codegen {}", cgu.name())));
                let start_time = Instant::now();
                backend.compile_codegen_unit(tcx, *cgu.name());
                total_codegen_time += start_time.elapsed();
                tcx.sess.profiler(|p| p.end_activity(format!("codegen {}", cgu.name())));
                false
            }
            CguReuse::PreLto => {
                submit_pre_lto_module_to_llvm(&backend, tcx, CachedModuleCodegen {
                    name: cgu.name().to_string(),
                    source: cgu.work_product(tcx),
                });
                true
            }
            CguReuse::PostLto => {
                submit_post_lto_module_to_llvm(&backend, tcx, CachedModuleCodegen {
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

    ::rustc_incremental::assert_module_sources::assert_module_sources(tcx);

    symbol_names_test::report_symbol_names(tcx);

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
struct AbortCodegenOnDrop<B: ExtraBackendMethods>(Option<OngoingCodegen<B>>);

impl<B: ExtraBackendMethods> AbortCodegenOnDrop<B> {
    fn into_inner(mut self) -> OngoingCodegen<B> {
        self.0.take().unwrap()
    }
}

impl<B: ExtraBackendMethods> Deref for AbortCodegenOnDrop<B> {
    type Target = OngoingCodegen<B>;

    fn deref(&self) -> &OngoingCodegen<B> {
        self.0.as_ref().unwrap()
    }
}

impl<B: ExtraBackendMethods> DerefMut for AbortCodegenOnDrop<B> {
    fn deref_mut(&mut self) -> &mut OngoingCodegen<B> {
        self.0.as_mut().unwrap()
    }
}

impl<B: ExtraBackendMethods> Drop for AbortCodegenOnDrop<B> {
    fn drop(&mut self) {
        if let Some(codegen) = self.0.take() {
            codegen.codegen_aborted();
        }
    }
}

fn assert_and_save_dep_graph(tcx: TyCtxt<'_>) {
    time(tcx.sess,
         "assert dep graph",
         || ::rustc_incremental::assert_dep_graph(tcx));

    time(tcx.sess,
         "serialize dep graph",
         || ::rustc_incremental::save_dep_graph(tcx));
}

impl CrateInfo {
    pub fn new(tcx: TyCtxt<'_>) -> CrateInfo {
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
            lang_item_to_crate: Default::default(),
            missing_lang_items: Default::default(),
        };
        let lang_items = tcx.lang_items();

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

        return info;
    }
}

fn is_codegened_item(tcx: TyCtxt<'_>, id: DefId) -> bool {
    let (all_mono_items, _) =
        tcx.collect_and_partition_mono_items(LOCAL_CRATE);
    all_mono_items.contains(&id)
}

pub fn provide_both(providers: &mut Providers<'_>) {
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
        for id in &*defids {
            let hir::CodegenFnAttrs { optimize, .. } = tcx.codegen_fn_attrs(*id);
            match optimize {
                attr::OptimizeAttr::None => continue,
                attr::OptimizeAttr::Size => continue,
                attr::OptimizeAttr::Speed => {
                    return for_speed;
                }
            }
        }
        return tcx.sess.opts.optimize;
    };

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
        tcx.arena.alloc(dllimports)
    };

    providers.is_dllimport_foreign_item = |tcx, def_id| {
        tcx.dllimport_foreign_items(def_id.krate).contains(&def_id)
    };
}

fn determine_cgu_reuse<'tcx>(tcx: TyCtxt<'tcx>, cgu: &CodegenUnit<'tcx>) -> CguReuse {
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

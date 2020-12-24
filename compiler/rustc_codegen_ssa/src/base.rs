//! Codegen the completed AST to the LLVM IR.
//!
//! Some functions here, such as `codegen_block` and `codegen_expr`, return a value --
//! the result of the codegen to LLVM -- while others, such as `codegen_fn`
//! and `mono_item`, are called only for the side effect of adding a
//! particular definition to the LLVM IR output we're producing.
//!
//! Hopefully useful general knowledge about codegen:
//!
//! * There's no way to find out the `Ty` type of a `Value`. Doing so
//!   would be "trying to get the eggs out of an omelette" (credit:
//!   pcwalton). You can, instead, find out its `llvm::Type` by calling `val_ty`,
//!   but one `llvm::Type` corresponds to many `Ty`s; for instance, `tup(int, int,
//!   int)` and `rec(x=int, y=int, z=int)` will have the same `llvm::Type`.

use crate::back::write::{
    compute_per_cgu_lto_type, start_async_codegen, submit_codegened_module_to_llvm,
    submit_post_lto_module_to_llvm, submit_pre_lto_module_to_llvm, ComputedLtoType, OngoingCodegen,
};
use crate::common::{IntPredicate, RealPredicate, TypeKind};
use crate::meth;
use crate::mir;
use crate::mir::operand::OperandValue;
use crate::mir::place::PlaceRef;
use crate::traits::*;
use crate::{CachedModuleCodegen, CrateInfo, MemFlags, ModuleCodegen, ModuleKind};

use rustc_attr as attr;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::profiling::print_time_passes_entry;
use rustc_data_structures::sync::{par_iter, Lock, ParallelIterator};
use rustc_hir as hir;
use rustc_hir::def_id::{LocalDefId, LOCAL_CRATE};
use rustc_hir::lang_items::LangItem;
use rustc_index::vec::Idx;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::middle::cstore::EncodedMetadata;
use rustc_middle::middle::cstore::{self, LinkagePreference};
use rustc_middle::middle::lang_items;
use rustc_middle::mir::mono::{CodegenUnit, CodegenUnitNameBuilder, MonoItem};
use rustc_middle::ty::layout::{HasTyCtxt, TyAndLayout};
use rustc_middle::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_session::cgu_reuse_tracker::CguReuse;
use rustc_session::config::{self, EntryFnType};
use rustc_session::utils::NativeLibKind;
use rustc_session::Session;
use rustc_symbol_mangling::test as symbol_names_test;
use rustc_target::abi::{Align, LayoutOf, VariantIdx};

use std::cmp;
use std::ops::{Deref, DerefMut};
use std::time::{Duration, Instant};

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
pub fn unsized_info<'tcx, Cx: CodegenMethods<'tcx>>(
    cx: &Cx,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
    old_info: Option<Cx::Value>,
) -> Cx::Value {
    let (source, target) =
        cx.tcx().struct_lockstep_tails_erasing_lifetimes(source, target, cx.param_env());
    match (source.kind(), target.kind()) {
        (&ty::Array(_, len), &ty::Slice(_)) => {
            cx.const_usize(len.eval_usize(cx.tcx(), ty::ParamEnv::reveal_all()))
        }
        (&ty::Dynamic(..), &ty::Dynamic(..)) => {
            // For now, upcasts are limited to changes in marker
            // traits, and hence never actually require an actual
            // change to the vtable.
            old_info.expect("unsized_info: missing old info for trait upcast")
        }
        (_, &ty::Dynamic(ref data, ..)) => {
            let vtable_ptr = cx.layout_of(cx.tcx().mk_mut_ptr(target)).field(cx, FAT_PTR_EXTRA);
            cx.const_ptrcast(
                meth::get_vtable(cx, source, data.principal()),
                cx.backend_type(vtable_ptr),
            )
        }
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}", source, target),
    }
}

/// Coerces `src` to `dst_ty`. `src_ty` must be a thin pointer.
pub fn unsize_thin_ptr<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    src: Bx::Value,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>,
) -> (Bx::Value, Bx::Value) {
    debug!("unsize_thin_ptr: {:?} => {:?}", src_ty, dst_ty);
    match (src_ty.kind(), dst_ty.kind()) {
        (&ty::Ref(_, a, _), &ty::Ref(_, b, _) | &ty::RawPtr(ty::TypeAndMut { ty: b, .. }))
        | (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }), &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
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
            // FIXME(eddyb) move these out of this `match` arm, so they're always
            // applied, uniformly, no matter the source/destination types.
            (
                bx.bitcast(lldata, bx.cx().scalar_pair_element_backend_type(dst_layout, 0, true)),
                bx.bitcast(llextra, bx.cx().scalar_pair_element_backend_type(dst_layout, 1, true)),
            )
        }
        _ => bug!("unsize_thin_ptr: called on bad types"),
    }
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
                OperandValue::Pair(base, info) => {
                    // fat-ptr to fat-ptr unsize preserves the vtable
                    // i.e., &'a fmt::Debug+Send => &'a fmt::Debug
                    // So we need to pointercast the base to ensure
                    // the types match up.
                    // FIXME(eddyb) use `scalar_pair_element_backend_type` here,
                    // like `unsize_thin_ptr` does.
                    let thin_ptr = dst.layout.field(bx.cx(), FAT_PTR_ADDR);
                    (bx.pointercast(base, bx.cx().backend_type(thin_ptr)), info)
                }
                OperandValue::Immediate(base) => unsize_thin_ptr(bx, base, src_ty, dst_ty),
                OperandValue::Ref(..) => bug!(),
            };
            OperandValue::Pair(base, info).store(bx, dst);
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

    mir::codegen_mir::<Bx>(cx, instance);
}

/// Creates the `main` function which will initialize the rust runtime and call
/// users main function.
pub fn maybe_create_entry_wrapper<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
) -> Option<Bx::Function> {
    let main_def_id = cx.tcx().entry_fn(LOCAL_CRATE).map(|(def_id, _)| def_id)?;
    let instance = Instance::mono(cx.tcx(), main_def_id.to_def_id());

    if !cx.codegen_unit().contains_item(&MonoItem::Fn(instance)) {
        // We want to create the wrapper in the same codegen unit as Rust's main
        // function.
        return None;
    }

    let main_llfn = cx.get_fn_addr(instance);

    return cx.tcx().entry_fn(LOCAL_CRATE).map(|(_, et)| {
        let use_start_lang_item = EntryFnType::Start != et;
        create_entry_fn::<Bx>(cx, main_llfn, main_def_id, use_start_lang_item)
    });

    fn create_entry_fn<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
        cx: &'a Bx::CodegenCx,
        rust_main: Bx::Value,
        rust_main_def_id: LocalDefId,
        use_start_lang_item: bool,
    ) -> Bx::Function {
        // The entry function is either `int main(void)` or `int main(int argc, char **argv)`,
        // depending on whether the target needs `argc` and `argv` to be passed in.
        let llfty = if cx.sess().target.main_needs_argc_argv {
            cx.type_func(&[cx.type_int(), cx.type_ptr_to(cx.type_i8p())], cx.type_int())
        } else {
            cx.type_func(&[], cx.type_int())
        };

        let main_ret_ty = cx.tcx().fn_sig(rust_main_def_id).output();
        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = cx.tcx().erase_regions(main_ret_ty.no_bound_vars().unwrap());

        let llfn = match cx.declare_c_main(llfty) {
            Some(llfn) => llfn,
            None => {
                // FIXME: We should be smart and show a better diagnostic here.
                let span = cx.tcx().def_span(rust_main_def_id);
                cx.sess()
                    .struct_span_err(span, "entry symbol `main` declared multiple times")
                    .help("did you use `#[no_mangle]` on `fn main`? Use `#[start]` instead")
                    .emit();
                cx.sess().abort_if_errors();
                bug!();
            }
        };

        // `main` should respect same config for frame pointer elimination as rest of code
        cx.set_frame_pointer_elimination(llfn);
        cx.apply_target_cpu_attr(llfn);

        let mut bx = Bx::new_block(&cx, llfn, "top");

        bx.insert_reference_to_gdb_debug_scripts_section_global();

        let (arg_argc, arg_argv) = get_argc_argv(cx, &mut bx);

        let (start_fn, args) = if use_start_lang_item {
            let start_def_id = cx.tcx().require_lang_item(LangItem::Start, None);
            let start_fn = cx.get_fn_addr(
                ty::Instance::resolve(
                    cx.tcx(),
                    ty::ParamEnv::reveal_all(),
                    start_def_id,
                    cx.tcx().intern_substs(&[main_ret_ty.into()]),
                )
                .unwrap()
                .unwrap(),
            );
            (
                start_fn,
                vec![bx.pointercast(rust_main, cx.type_ptr_to(cx.type_i8p())), arg_argc, arg_argv],
            )
        } else {
            debug!("using user-defined start fn");
            (rust_main, vec![arg_argc, arg_argv])
        };

        let result = bx.call(start_fn, &args, None);
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

pub fn codegen_crate<B: ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'tcx>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> OngoingCodegen<B> {
    // Skip crate items and just output metadata in -Z no-codegen mode.
    if tcx.sess.opts.debugging_opts.no_codegen || !tcx.sess.opts.output_types.should_codegen() {
        let ongoing_codegen = start_async_codegen(backend, tcx, metadata, 1);

        ongoing_codegen.codegen_finished(tcx);

        finalize_tcx(tcx);

        ongoing_codegen.check_for_errors(tcx.sess);

        return ongoing_codegen;
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);

    // Run the monomorphization collector and partition the collected items into
    // codegen units.
    let codegen_units = tcx.collect_and_partition_mono_items(LOCAL_CRATE).1;

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

    let ongoing_codegen = start_async_codegen(backend.clone(), tcx, metadata, codegen_units.len());
    let ongoing_codegen = AbortCodegenOnDrop::<B>(Some(ongoing_codegen));

    // Codegen an allocator shim, if necessary.
    //
    // If the crate doesn't have an `allocator_kind` set then there's definitely
    // no shim to generate. Otherwise we also check our dependency graph for all
    // our output crate types. If anything there looks like its a `Dynamic`
    // linkage, then it's already got an allocator shim and we'll be using that
    // one instead. If nothing exists then it's our job to generate the
    // allocator!
    let any_dynamic_crate = tcx.dependency_formats(LOCAL_CRATE).iter().any(|(_, list)| {
        use rustc_middle::middle::dependency_format::Linkage;
        list.iter().any(|&linkage| linkage == Linkage::Dynamic)
    });
    let allocator_module = if any_dynamic_crate {
        None
    } else if let Some(kind) = tcx.allocator_kind() {
        let llmod_id =
            cgu_name_builder.build_cgu_name(LOCAL_CRATE, &["crate"], Some("allocator")).to_string();
        let mut modules = backend.new_metadata(tcx, &llmod_id);
        tcx.sess.time("write_allocator_module", || {
            backend.codegen_allocator(tcx, &mut modules, kind, tcx.lang_items().oom().is_some())
        });

        Some(ModuleCodegen { name: llmod_id, module_llvm: modules, kind: ModuleKind::Allocator })
    } else {
        None
    };

    if let Some(allocator_module) = allocator_module {
        ongoing_codegen.submit_pre_codegened_module_to_llvm(tcx, allocator_module);
    }

    if need_metadata_module {
        // Codegen the encoded metadata.
        let metadata_cgu_name =
            cgu_name_builder.build_cgu_name(LOCAL_CRATE, &["crate"], Some("metadata")).to_string();
        let mut metadata_llvm_module = backend.new_metadata(tcx, &metadata_cgu_name);
        tcx.sess.time("write_compressed_metadata", || {
            backend.write_compressed_metadata(
                tcx,
                &ongoing_codegen.metadata,
                &mut metadata_llvm_module,
            );
        });

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
        let mut codegen_units = codegen_units.iter().collect::<Vec<_>>();
        codegen_units.sort_by_cached_key(|cgu| cmp::Reverse(cgu.size_estimate()));
        codegen_units
    };

    let total_codegen_time = Lock::new(Duration::new(0, 0));

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
    let pre_compile_cgus = |cgu_reuse: &[CguReuse]| {
        if cfg!(parallel_compiler) {
            tcx.sess.time("compile_first_CGU_batch", || {
                // Try to find one CGU to compile per thread.
                let cgus: Vec<_> = cgu_reuse
                    .iter()
                    .enumerate()
                    .filter(|&(_, reuse)| reuse == &CguReuse::No)
                    .take(tcx.sess.threads())
                    .collect();

                // Compile the found CGUs in parallel.
                par_iter(cgus)
                    .map(|(i, _)| {
                        let start_time = Instant::now();
                        let module = backend.compile_codegen_unit(tcx, codegen_units[i].name());
                        let mut time = total_codegen_time.lock();
                        *time += start_time.elapsed();
                        (i, module)
                    })
                    .collect()
            })
        } else {
            FxHashMap::default()
        }
    };

    let mut cgu_reuse = Vec::new();
    let mut pre_compiled_cgus: Option<FxHashMap<usize, _>> = None;

    for (i, cgu) in codegen_units.iter().enumerate() {
        ongoing_codegen.wait_for_signal_to_codegen_item();
        ongoing_codegen.check_for_errors(tcx.sess);

        // Do some setup work in the first iteration
        if pre_compiled_cgus.is_none() {
            // Calculate the CGU reuse
            cgu_reuse = tcx.sess.time("find_cgu_reuse", || {
                codegen_units.iter().map(|cgu| determine_cgu_reuse(tcx, &cgu)).collect()
            });
            // Pre compile some CGUs
            pre_compiled_cgus = Some(pre_compile_cgus(&cgu_reuse));
        }

        let cgu_reuse = cgu_reuse[i];
        tcx.sess.cgu_reuse_tracker.set_actual_reuse(&cgu.name().as_str(), cgu_reuse);

        match cgu_reuse {
            CguReuse::No => {
                let (module, cost) =
                    if let Some(cgu) = pre_compiled_cgus.as_mut().unwrap().remove(&i) {
                        cgu
                    } else {
                        let start_time = Instant::now();
                        let module = backend.compile_codegen_unit(tcx, cgu.name());
                        let mut time = total_codegen_time.lock();
                        *time += start_time.elapsed();
                        module
                    };
                submit_codegened_module_to_llvm(
                    &backend,
                    &ongoing_codegen.coordinator_send,
                    module,
                    cost,
                );
                false
            }
            CguReuse::PreLto => {
                submit_pre_lto_module_to_llvm(
                    &backend,
                    tcx,
                    &ongoing_codegen.coordinator_send,
                    CachedModuleCodegen {
                        name: cgu.name().to_string(),
                        source: cgu.work_product(tcx),
                    },
                );
                true
            }
            CguReuse::PostLto => {
                submit_post_lto_module_to_llvm(
                    &backend,
                    &ongoing_codegen.coordinator_send,
                    CachedModuleCodegen {
                        name: cgu.name().to_string(),
                        source: cgu.work_product(tcx),
                    },
                );
                true
            }
        };
    }

    ongoing_codegen.codegen_finished(tcx);

    // Since the main thread is sometimes blocked during codegen, we keep track
    // -Ztime-passes output manually.
    print_time_passes_entry(
        tcx.sess.time_passes(),
        "codegen_to_LLVM_IR",
        total_codegen_time.into_inner(),
    );

    rustc_incremental::assert_module_sources::assert_module_sources(tcx);

    symbol_names_test::report_symbol_names(tcx);

    ongoing_codegen.check_for_errors(tcx.sess);

    finalize_tcx(tcx);

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

fn finalize_tcx(tcx: TyCtxt<'_>) {
    tcx.sess.time("assert_dep_graph", || rustc_incremental::assert_dep_graph(tcx));
    tcx.sess.time("serialize_dep_graph", || rustc_incremental::save_dep_graph(tcx));

    // We assume that no queries are run past here. If there are new queries
    // after this point, they'll show up as "<unknown>" in self-profiling data.
    {
        let _prof_timer = tcx.prof.generic_activity("self_profile_alloc_query_strings");
        tcx.alloc_self_profile_query_strings();
    }
}

impl CrateInfo {
    pub fn new(tcx: TyCtxt<'_>) -> CrateInfo {
        let mut info = CrateInfo {
            panic_runtime: None,
            compiler_builtins: None,
            profiler_runtime: None,
            is_no_builtins: Default::default(),
            native_libraries: Default::default(),
            used_libraries: tcx.native_libraries(LOCAL_CRATE).iter().map(Into::into).collect(),
            link_args: tcx.link_args(LOCAL_CRATE),
            crate_name: Default::default(),
            used_crates_dynamic: cstore::used_crates(tcx, LinkagePreference::RequireDynamic),
            used_crates_static: cstore::used_crates(tcx, LinkagePreference::RequireStatic),
            used_crate_source: Default::default(),
            lang_item_to_crate: Default::default(),
            missing_lang_items: Default::default(),
            dependency_formats: tcx.dependency_formats(LOCAL_CRATE),
        };
        let lang_items = tcx.lang_items();

        let crates = tcx.crates();

        let n_crates = crates.len();
        info.native_libraries.reserve(n_crates);
        info.crate_name.reserve(n_crates);
        info.used_crate_source.reserve(n_crates);
        info.missing_lang_items.reserve(n_crates);

        for &cnum in crates.iter() {
            info.native_libraries
                .insert(cnum, tcx.native_libraries(cnum).iter().map(Into::into).collect());
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
            if tcx.is_no_builtins(cnum) {
                info.is_no_builtins.insert(cnum);
            }
            let missing = tcx.missing_lang_items(cnum);
            for &item in missing.iter() {
                if let Ok(id) = lang_items.require(item) {
                    info.lang_item_to_crate.insert(item, id.krate);
                }
            }

            // No need to look for lang items that don't actually need to exist.
            let missing =
                missing.iter().cloned().filter(|&l| lang_items::required(tcx, l)).collect();
            info.missing_lang_items.insert(cnum, missing);
        }

        info
    }
}

pub fn provide_both(providers: &mut Providers) {
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
            let CodegenFnAttrs { optimize, .. } = tcx.codegen_fn_attrs(*id);
            match optimize {
                attr::OptimizeAttr::None => continue,
                attr::OptimizeAttr::Size => continue,
                attr::OptimizeAttr::Speed => {
                    return for_speed;
                }
            }
        }
        tcx.sess.opts.optimize
    };

    providers.dllimport_foreign_items = |tcx, krate| {
        let module_map = tcx.foreign_modules(krate);

        let dllimports = tcx
            .native_libraries(krate)
            .iter()
            .filter(|lib| {
                if !matches!(lib.kind, NativeLibKind::Dylib | NativeLibKind::Unspecified) {
                    return false;
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
        dllimports
    };

    providers.is_dllimport_foreign_item =
        |tcx, def_id| tcx.dllimport_foreign_items(def_id.krate).contains(&def_id);
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

    if tcx.dep_graph.try_mark_green(tcx, &dep_node).is_some() {
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

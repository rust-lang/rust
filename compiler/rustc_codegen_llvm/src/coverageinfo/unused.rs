use crate::abi::Abi;
use crate::builder::Builder;
use crate::common::CodegenCx;
use crate::coverageinfo::map_data::FunctionCoverage;
use crate::llvm;

use rustc_codegen_ssa::traits::{BuilderMethods, ConstMethods, MiscMethods};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::coverage::CounterId;
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::{self, GenericArgs, Instance, Ty};

/// When finalizing the coverage map, `FunctionCoverage` only has the `CodeRegion`s and counters for
/// the functions that went through codegen; such as public functions and "used" functions
/// (functions referenced by other "used" or public items). Any other functions considered unused,
/// or "Unreachable", were still parsed and processed through the MIR stage, but were not
/// codegenned. (Note that `-Clink-dead-code` can force some unused code to be codegenned, but
/// that flag is known to cause other errors, when combined with `-C instrument-coverage`; and
/// `-Clink-dead-code` will not generate code for unused generic functions.)
///
/// We can find the unused functions (including generic functions) by the set difference of all MIR
/// `DefId`s (`tcx` query `mir_keys`) minus the codegenned `DefId`s (`tcx` query
/// `codegened_and_inlined_items`).
///
/// These unused functions are then codegen'd in one of the CGUs which is marked as the
/// "code coverage dead code cgu" during the partitioning process. This prevents us from generating
/// code regions for the same function more than once which can lead to linker errors regarding
/// duplicate symbols.
pub(crate) fn synthesize_unused_functions(cx: &CodegenCx<'_, '_>) {
    assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

    let tcx = cx.tcx;

    let ignore_unused_generics = tcx.sess.instrument_coverage_except_unused_generics();

    let eligible_def_ids: Vec<DefId> = tcx
        .mir_keys(())
        .iter()
        .filter_map(|local_def_id| {
            let def_id = local_def_id.to_def_id();
            let kind = tcx.def_kind(def_id);
            // `mir_keys` will give us `DefId`s for all kinds of things, not
            // just "functions", like consts, statics, etc. Filter those out.
            // If `ignore_unused_generics` was specified, filter out any
            // generic functions from consideration as well.
            if !matches!(
                kind,
                DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::Generator
            ) {
                return None;
            }
            if ignore_unused_generics && tcx.generics_of(def_id).requires_monomorphization(tcx) {
                return None;
            }
            Some(local_def_id.to_def_id())
        })
        .collect();

    let codegenned_def_ids = tcx.codegened_and_inlined_items(());

    for non_codegenned_def_id in
        eligible_def_ids.into_iter().filter(|id| !codegenned_def_ids.contains(id))
    {
        let codegen_fn_attrs = tcx.codegen_fn_attrs(non_codegenned_def_id);

        // If a function is marked `#[no_coverage]`, then skip generating a
        // dead code stub for it.
        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NO_COVERAGE) {
            debug!("skipping unused fn marked #[no_coverage]: {:?}", non_codegenned_def_id);
            continue;
        }

        debug!("generating unused fn: {:?}", non_codegenned_def_id);
        define_unused_fn(cx, non_codegenned_def_id);
    }
}

/// Functions with MIR-based coverage are normally codegenned _only_ if
/// called. LLVM coverage tools typically expect every function to be
/// defined (even if unused), with at least one call to LLVM intrinsic
/// `instrprof.increment`.
///
/// Codegen a small function that will never be called, with one counter
/// that will never be incremented.
///
/// For used/called functions, the coverageinfo was already added to the
/// `function_coverage_map` (keyed by function `Instance`) during codegen.
/// But in this case, since the unused function was _not_ previously
/// codegenned, collect the coverage `CodeRegion`s from the MIR and add
/// them. The first `CodeRegion` is used to add a single counter, with the
/// same counter ID used in the injected `instrprof.increment` intrinsic
/// call. Since the function is never called, all other `CodeRegion`s can be
/// added as `unreachable_region`s.
fn define_unused_fn<'tcx>(cx: &CodegenCx<'_, 'tcx>, def_id: DefId) {
    let instance = declare_unused_fn(cx, def_id);
    codegen_unused_fn_and_counter(cx, instance);
    add_unused_function_coverage(cx, instance, def_id);
}

fn declare_unused_fn<'tcx>(cx: &CodegenCx<'_, 'tcx>, def_id: DefId) -> Instance<'tcx> {
    let tcx = cx.tcx;

    let instance = Instance::new(
        def_id,
        GenericArgs::for_item(tcx, def_id, |param, _| {
            if let ty::GenericParamDefKind::Lifetime = param.kind {
                tcx.lifetimes.re_erased.into()
            } else {
                tcx.mk_param_from_def(param)
            }
        }),
    );

    let llfn = cx.declare_fn(
        tcx.symbol_name(instance).name,
        cx.fn_abi_of_fn_ptr(
            ty::Binder::dummy(tcx.mk_fn_sig(
                [Ty::new_unit(tcx)],
                Ty::new_unit(tcx),
                false,
                hir::Unsafety::Unsafe,
                Abi::Rust,
            )),
            ty::List::empty(),
        ),
        None,
    );

    llvm::set_linkage(llfn, llvm::Linkage::PrivateLinkage);
    llvm::set_visibility(llfn, llvm::Visibility::Default);

    assert!(cx.instances.borrow_mut().insert(instance, llfn).is_none());

    instance
}

const UNUSED_FUNCTION_COUNTER_ID: CounterId = CounterId::START;

fn codegen_unused_fn_and_counter<'tcx>(cx: &CodegenCx<'_, 'tcx>, instance: Instance<'tcx>) {
    let llfn = cx.get_fn(instance);
    let llbb = Builder::append_block(cx, llfn, "unused_function");
    let mut bx = Builder::build(cx, llbb);
    let fn_name = bx.get_pgo_func_name_var(instance);
    let hash = bx.const_u64(0);
    let num_counters = bx.const_u32(1);
    let index = bx.const_u32(u32::from(UNUSED_FUNCTION_COUNTER_ID));
    debug!(
        "codegen intrinsic instrprof.increment(fn_name={:?}, hash={:?}, num_counters={:?},
            index={:?}) for unused function: {:?}",
        fn_name, hash, num_counters, index, instance
    );
    bx.instrprof_increment(fn_name, hash, num_counters, index);
    bx.ret_void();
}

fn add_unused_function_coverage<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    instance: Instance<'tcx>,
    def_id: DefId,
) {
    let tcx = cx.tcx;

    let mut function_coverage = FunctionCoverage::unused(tcx, instance);
    for (index, &code_region) in tcx.covered_code_regions(def_id).iter().enumerate() {
        if index == 0 {
            // Insert at least one real counter so the LLVM CoverageMappingReader will find expected
            // definitions.
            function_coverage.add_counter(UNUSED_FUNCTION_COUNTER_ID, code_region.clone());
        } else {
            function_coverage.add_unreachable_region(code_region.clone());
        }
    }

    if let Some(coverage_context) = cx.coverage_context() {
        coverage_context.function_coverage_map.borrow_mut().insert(instance, function_coverage);
    } else {
        bug!("Could not get the `coverage_context`");
    }
}

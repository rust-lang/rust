use std::cell::{OnceCell, RefCell};
use std::ffi::{CStr, CString};

use rustc_codegen_ssa::traits::{
    ConstCodegenMethods, CoverageInfoBuilderMethods, MiscCodegenMethods,
};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::ty::Instance;
use tracing::{debug, instrument};

use crate::builder::Builder;
use crate::common::CodegenCx;
use crate::llvm;

pub(crate) mod ffi;
mod llvm_cov;
mod mapgen;

/// Extra per-CGU context/state needed for coverage instrumentation.
pub(crate) struct CguCoverageContext<'ll, 'tcx> {
    /// Associates function instances with an LLVM global that holds the
    /// function's symbol name, as needed by LLVM coverage intrinsics.
    ///
    /// Instances in this map are also considered "used" for the purposes of
    /// emitting covfun records. Every covfun record holds a hash of its
    /// symbol name, and `llvm-cov` will exit fatally if it can't resolve that
    /// hash back to an entry in the binary's `__llvm_prf_names` linker section.
    pub(crate) pgo_func_name_var_map: RefCell<FxIndexMap<Instance<'tcx>, &'ll llvm::Value>>,

    covfun_section_name: OnceCell<CString>,
}

impl<'ll, 'tcx> CguCoverageContext<'ll, 'tcx> {
    pub(crate) fn new() -> Self {
        Self { pgo_func_name_var_map: Default::default(), covfun_section_name: Default::default() }
    }

    /// Returns the list of instances considered "used" in this CGU, as
    /// inferred from the keys of `pgo_func_name_var_map`.
    pub(crate) fn instances_used(&self) -> Vec<Instance<'tcx>> {
        // Collecting into a Vec is way easier than trying to juggle RefCell
        // projections, and this should only run once per CGU anyway.
        self.pgo_func_name_var_map.borrow().keys().copied().collect::<Vec<_>>()
    }
}

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    pub(crate) fn coverageinfo_finalize(&mut self) {
        mapgen::finalize(self)
    }

    /// Returns the section name to use when embedding per-function coverage information
    /// in the object file, according to the target's object file format. LLVM's coverage
    /// tools use information from this section when producing coverage reports.
    ///
    /// Typical values are:
    /// - `__llvm_covfun` on Linux
    /// - `__LLVM_COV,__llvm_covfun` on macOS (includes `__LLVM_COV,` segment prefix)
    /// - `.lcovfun$M` on Windows (includes `$M` sorting suffix)
    fn covfun_section_name(&self) -> &CStr {
        self.coverage_cx()
            .covfun_section_name
            .get_or_init(|| llvm_cov::covfun_section_name(self.llmod))
    }

    /// For LLVM codegen, returns a function-specific `Value` for a global
    /// string, to hold the function name passed to LLVM intrinsic
    /// `instrprof.increment()`. The `Value` is only created once per instance.
    /// Multiple invocations with the same instance return the same `Value`.
    ///
    /// This has the side-effect of causing coverage codegen to consider this
    /// function "used", making it eligible to emit an associated covfun record.
    fn ensure_pgo_func_name_var(&self, instance: Instance<'tcx>) -> &'ll llvm::Value {
        debug!("getting pgo_func_name_var for instance={:?}", instance);
        let mut pgo_func_name_var_map = self.coverage_cx().pgo_func_name_var_map.borrow_mut();
        pgo_func_name_var_map.entry(instance).or_insert_with(|| {
            let llfn = self.get_fn(instance);
            let mangled_fn_name: &str = self.tcx.symbol_name(instance).name;
            llvm_cov::create_pgo_func_name_var(llfn, mangled_fn_name)
        })
    }
}

impl<'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'_, '_, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    fn add_coverage(&mut self, instance: Instance<'tcx>, kind: &CoverageKind) {
        // Our caller should have already taken care of inlining subtleties,
        // so we can assume that counter/expression IDs in this coverage
        // statement are meaningful for the given instance.
        //
        // (Either the statement was not inlined and directly belongs to this
        // instance, or it was inlined *from* this instance.)

        let bx = self;

        // Due to LocalCopy instantiation or MIR inlining, coverage statements
        // can end up in a crate that isn't doing coverage instrumentation.
        // When that happens, we currently just discard those statements, so
        // the corresponding code will be undercounted.
        // FIXME(Zalathar): Find a better solution for mixed-coverage builds.
        let Some(_coverage_cx) = &bx.cx.coverage_cx else { return };

        let Some(function_coverage_info) =
            bx.tcx.instance_mir(instance.def).function_coverage_info.as_deref()
        else {
            debug!("function has a coverage statement but no coverage info");
            return;
        };
        let Some(ids_info) = bx.tcx.coverage_ids_info(instance.def) else {
            debug!("function has a coverage statement but no IDs info");
            return;
        };

        match *kind {
            CoverageKind::SpanMarker | CoverageKind::BlockMarker { .. } => unreachable!(
                "marker statement {kind:?} should have been removed by CleanupPostBorrowck"
            ),
            CoverageKind::VirtualCounter { bcb }
                if let Some(&id) = ids_info.phys_counter_for_node.get(&bcb) =>
            {
                let fn_name = bx.ensure_pgo_func_name_var(instance);
                let hash = bx.const_u64(function_coverage_info.function_source_hash);
                let num_counters = bx.const_u32(ids_info.num_counters);
                let index = bx.const_u32(id.as_u32());
                debug!(
                    "codegen intrinsic instrprof.increment(fn_name={:?}, hash={:?}, num_counters={:?}, index={:?})",
                    fn_name, hash, num_counters, index,
                );
                bx.instrprof_increment(fn_name, hash, num_counters, index);
            }
            // If a BCB doesn't have an associated physical counter, there's nothing to codegen.
            CoverageKind::VirtualCounter { .. } => {}
        }
    }
}

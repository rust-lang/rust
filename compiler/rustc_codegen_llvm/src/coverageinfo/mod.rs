use std::cell::{OnceCell, RefCell};
use std::ffi::{CStr, CString};

use rustc_abi::Size;
use rustc_codegen_ssa::traits::{
    BuilderMethods, ConstCodegenMethods, CoverageInfoBuilderMethods, MiscCodegenMethods,
};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::ty::Instance;
use rustc_middle::ty::layout::HasTyCtxt;
use tracing::{debug, instrument};

use crate::builder::Builder;
use crate::common::CodegenCx;
use crate::coverageinfo::map_data::FunctionCoverageCollector;
use crate::llvm;

pub(crate) mod ffi;
mod llvm_cov;
pub(crate) mod map_data;
mod mapgen;

/// A context object for maintaining all state needed by the coverageinfo module.
pub(crate) struct CrateCoverageContext<'ll, 'tcx> {
    /// Coverage data for each instrumented function identified by DefId.
    pub(crate) function_coverage_map:
        RefCell<FxIndexMap<Instance<'tcx>, FunctionCoverageCollector<'tcx>>>,
    pub(crate) pgo_func_name_var_map: RefCell<FxHashMap<Instance<'tcx>, &'ll llvm::Value>>,
    pub(crate) mcdc_condition_bitmap_map: RefCell<FxHashMap<Instance<'tcx>, Vec<&'ll llvm::Value>>>,

    covfun_section_name: OnceCell<CString>,
}

impl<'ll, 'tcx> CrateCoverageContext<'ll, 'tcx> {
    pub(crate) fn new() -> Self {
        Self {
            function_coverage_map: Default::default(),
            pgo_func_name_var_map: Default::default(),
            mcdc_condition_bitmap_map: Default::default(),
            covfun_section_name: Default::default(),
        }
    }

    fn take_function_coverage_map(
        &self,
    ) -> FxIndexMap<Instance<'tcx>, FunctionCoverageCollector<'tcx>> {
        self.function_coverage_map.replace(FxIndexMap::default())
    }

    /// LLVM use a temp value to record evaluated mcdc test vector of each decision, which is
    /// called condition bitmap. In order to handle nested decisions, several condition bitmaps can
    /// be allocated for a function body. These values are named `mcdc.addr.{i}` and are a 32-bit
    /// integers. They respectively hold the condition bitmaps for decisions with a depth of `i`.
    fn try_get_mcdc_condition_bitmap(
        &self,
        instance: &Instance<'tcx>,
        decision_depth: u16,
    ) -> Option<&'ll llvm::Value> {
        self.mcdc_condition_bitmap_map
            .borrow()
            .get(instance)
            .and_then(|bitmap_map| bitmap_map.get(decision_depth as usize))
            .copied() // Dereference Option<&&Value> to Option<&Value>
    }
}

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    pub(crate) fn coverageinfo_finalize(&self) {
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
    fn get_pgo_func_name_var(&self, instance: Instance<'tcx>) -> &'ll llvm::Value {
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
    fn init_coverage(&mut self, instance: Instance<'tcx>) {
        let Some(function_coverage_info) =
            self.tcx.instance_mir(instance.def).function_coverage_info.as_deref()
        else {
            return;
        };

        // If there are no MC/DC bitmaps to set up, return immediately.
        if function_coverage_info.mcdc_bitmap_bits == 0 {
            return;
        }

        let fn_name = self.get_pgo_func_name_var(instance);
        let hash = self.const_u64(function_coverage_info.function_source_hash);
        let bitmap_bits = self.const_u32(function_coverage_info.mcdc_bitmap_bits as u32);
        self.mcdc_parameters(fn_name, hash, bitmap_bits);

        // Create pointers named `mcdc.addr.{i}` to stack-allocated condition bitmaps.
        let mut cond_bitmaps = vec![];
        for i in 0..function_coverage_info.mcdc_num_condition_bitmaps {
            // MC/DC intrinsics will perform loads/stores that use the ABI default
            // alignment for i32, so our variable declaration should match.
            let align = self.tcx.data_layout.i32_align.abi;
            let cond_bitmap = self.alloca(Size::from_bytes(4), align);
            llvm::set_value_name(cond_bitmap, format!("mcdc.addr.{i}").as_bytes());
            self.store(self.const_i32(0), cond_bitmap, align);
            cond_bitmaps.push(cond_bitmap);
        }

        self.coverage_cx().mcdc_condition_bitmap_map.borrow_mut().insert(instance, cond_bitmaps);
    }

    #[instrument(level = "debug", skip(self))]
    fn add_coverage(&mut self, instance: Instance<'tcx>, kind: &CoverageKind) {
        // Our caller should have already taken care of inlining subtleties,
        // so we can assume that counter/expression IDs in this coverage
        // statement are meaningful for the given instance.
        //
        // (Either the statement was not inlined and directly belongs to this
        // instance, or it was inlined *from* this instance.)

        let bx = self;

        let Some(function_coverage_info) =
            bx.tcx.instance_mir(instance.def).function_coverage_info.as_deref()
        else {
            debug!("function has a coverage statement but no coverage info");
            return;
        };

        // FIXME(#132395): Unwrapping `coverage_cx` here has led to ICEs in the
        // wild, so keep this early-return until we understand why.
        let mut coverage_map = match bx.coverage_cx {
            Some(ref cx) => cx.function_coverage_map.borrow_mut(),
            None => return,
        };
        let func_coverage = coverage_map
            .entry(instance)
            .or_insert_with(|| FunctionCoverageCollector::new(instance, function_coverage_info));

        match *kind {
            CoverageKind::SpanMarker | CoverageKind::BlockMarker { .. } => unreachable!(
                "marker statement {kind:?} should have been removed by CleanupPostBorrowck"
            ),
            CoverageKind::CounterIncrement { id } => {
                func_coverage.mark_counter_id_seen(id);
                // We need to explicitly drop the `RefMut` before calling into
                // `instrprof_increment`, as that needs an exclusive borrow.
                drop(coverage_map);

                // The number of counters passed to `llvm.instrprof.increment` might
                // be smaller than the number originally inserted by the instrumentor,
                // if some high-numbered counters were removed by MIR optimizations.
                // If so, LLVM's profiler runtime will use fewer physical counters.
                let num_counters =
                    bx.tcx().coverage_ids_info(instance.def).max_counter_id.as_u32() + 1;
                assert!(
                    num_counters as usize <= function_coverage_info.num_counters,
                    "num_counters disagreement: query says {num_counters} but function info only has {}",
                    function_coverage_info.num_counters
                );

                let fn_name = bx.get_pgo_func_name_var(instance);
                let hash = bx.const_u64(function_coverage_info.function_source_hash);
                let num_counters = bx.const_u32(num_counters);
                let index = bx.const_u32(id.as_u32());
                debug!(
                    "codegen intrinsic instrprof.increment(fn_name={:?}, hash={:?}, num_counters={:?}, index={:?})",
                    fn_name, hash, num_counters, index,
                );
                bx.instrprof_increment(fn_name, hash, num_counters, index);
            }
            CoverageKind::ExpressionUsed { id } => {
                func_coverage.mark_expression_id_seen(id);
            }
            CoverageKind::CondBitmapUpdate { index, decision_depth } => {
                drop(coverage_map);
                let cond_bitmap = bx
                    .coverage_cx()
                    .try_get_mcdc_condition_bitmap(&instance, decision_depth)
                    .expect("mcdc cond bitmap should have been allocated for updating");
                let cond_index = bx.const_i32(index as i32);
                bx.mcdc_condbitmap_update(cond_index, cond_bitmap);
            }
            CoverageKind::TestVectorBitmapUpdate { bitmap_idx, decision_depth } => {
                drop(coverage_map);
                let cond_bitmap = bx.coverage_cx()
                                    .try_get_mcdc_condition_bitmap(&instance, decision_depth)
                                    .expect("mcdc cond bitmap should have been allocated for merging into the global bitmap");
                assert!(
                    bitmap_idx as usize <= function_coverage_info.mcdc_bitmap_bits,
                    "bitmap index of the decision out of range"
                );

                let fn_name = bx.get_pgo_func_name_var(instance);
                let hash = bx.const_u64(function_coverage_info.function_source_hash);
                let bitmap_index = bx.const_u32(bitmap_idx);
                bx.mcdc_tvbitmap_update(fn_name, hash, bitmap_index, cond_bitmap);
                bx.mcdc_condbitmap_reset(cond_bitmap);
            }
        }
    }
}

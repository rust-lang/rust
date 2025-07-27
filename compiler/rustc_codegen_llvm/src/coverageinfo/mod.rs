use std::cell::{OnceCell, RefCell};
use std::ffi::{CStr, CString};

use rustc_abi::Size;
use rustc_codegen_ssa::traits::{
    BuilderMethods, ConstCodegenMethods, CoverageInfoBuilderMethods, MiscCodegenMethods,
};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
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
    pub(crate) mcdc_condition_bitmap_map: RefCell<FxHashMap<Instance<'tcx>, Vec<&'ll llvm::Value>>>,

    covfun_section_name: OnceCell<CString>,
}

impl<'ll, 'tcx> CguCoverageContext<'ll, 'tcx> {
    pub(crate) fn new() -> Self {
        Self {
            pgo_func_name_var_map: Default::default(),
            mcdc_condition_bitmap_map: Default::default(),
            covfun_section_name: Default::default(),
        }
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

        let fn_name = self.ensure_pgo_func_name_var(instance);
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

        // Due to LocalCopy instantiation or MIR inlining, coverage statements
        // can end up in a crate that isn't doing coverage instrumentation.
        // When that happens, we currently just discard those statements, so
        // the corresponding code will be undercounted.
        // FIXME(Zalathar): Find a better solution for mixed-coverage builds.
        let Some(coverage_cx) = &bx.cx.coverage_cx else { return };

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
            CoverageKind::CondBitmapUpdate { index, decision_depth } => {
                let cond_bitmap = coverage_cx
                    .try_get_mcdc_condition_bitmap(&instance, decision_depth)
                    .expect("mcdc cond bitmap should have been allocated for updating");
                let cond_index = bx.const_i32(index as i32);
                bx.mcdc_condbitmap_update(cond_index, cond_bitmap);
            }
            CoverageKind::TestVectorBitmapUpdate { bitmap_idx, decision_depth } => {
                let cond_bitmap =
                    coverage_cx.try_get_mcdc_condition_bitmap(&instance, decision_depth).expect(
                        "mcdc cond bitmap should have been allocated for merging \
                        into the global bitmap",
                    );
                assert!(
                    bitmap_idx as usize <= function_coverage_info.mcdc_bitmap_bits,
                    "bitmap index of the decision out of range"
                );

                let fn_name = bx.ensure_pgo_func_name_var(instance);
                let hash = bx.const_u64(function_coverage_info.function_source_hash);
                let bitmap_index = bx.const_u32(bitmap_idx);
                bx.mcdc_tvbitmap_update(fn_name, hash, bitmap_index, cond_bitmap);
                bx.mcdc_condbitmap_reset(cond_bitmap);
            }
        }
    }
}

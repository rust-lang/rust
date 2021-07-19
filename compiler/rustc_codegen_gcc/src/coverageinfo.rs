use gccjit::RValue;
use rustc_codegen_ssa::traits::{CoverageInfoBuilderMethods, CoverageInfoMethods};
use rustc_hir::def_id::DefId;
use rustc_middle::mir::coverage::{
    CodeRegion,
    CounterValueReference,
    ExpressionOperandId,
    InjectedExpressionId,
    Op,
};
use rustc_middle::ty::Instance;

use crate::builder::Builder;
use crate::context::CodegenCx;

impl<'a, 'gcc, 'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn set_function_source_hash(
        &mut self,
        _instance: Instance<'tcx>,
        _function_source_hash: u64,
    ) -> bool {
        unimplemented!();
        /*if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "ensuring function source hash is set for instance={:?}; function_source_hash={}",
                instance, function_source_hash,
            );
            let mut coverage_map = coverage_context.function_coverage_map.borrow_mut();
            coverage_map
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .set_function_source_hash(function_source_hash);
            true
        } else {
            false
        }*/
    }

    fn add_coverage_counter(&mut self, _instance: Instance<'tcx>, _id: CounterValueReference, _region: CodeRegion) -> bool {
        /*if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "adding counter to coverage_regions: instance={:?}, function_source_hash={}, id={:?}, \
                at {:?}",
                instance, function_source_hash, id, region,
            );
            let mut coverage_regions = coverage_context.function_coverage_map.borrow_mut();
            coverage_regions
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .add_counter(function_source_hash, id, region);
            true
        } else {
            false
        }*/
        // TODO
        false
    }

    fn add_coverage_counter_expression(&mut self, _instance: Instance<'tcx>, _id: InjectedExpressionId, _lhs: ExpressionOperandId, _op: Op, _rhs: ExpressionOperandId, _region: Option<CodeRegion>) -> bool {
        /*if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "adding counter expression to coverage_regions: instance={:?}, id={:?}, {:?} {:?} {:?}, \
                at {:?}",
                instance, id, lhs, op, rhs, region,
            );
            let mut coverage_regions = coverage_context.function_coverage_map.borrow_mut();
            coverage_regions
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .add_counter_expression(id, lhs, op, rhs, region);
            true
        } else {
            false
        }*/
        // TODO
        false
    }

    fn add_coverage_unreachable(&mut self, _instance: Instance<'tcx>, _region: CodeRegion) -> bool {
        /*if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "adding unreachable code to coverage_regions: instance={:?}, at {:?}",
                instance, region,
            );
            let mut coverage_regions = coverage_context.function_coverage_map.borrow_mut();
            coverage_regions
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .add_unreachable_region(region);
            true
        } else {
            false
        }*/
        // TODO
        false
    }
}

impl<'gcc, 'tcx> CoverageInfoMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn coverageinfo_finalize(&self) {
        // TODO
        //mapgen::finalize(self)
    }

    fn get_pgo_func_name_var(&self, _instance: Instance<'tcx>) -> RValue<'gcc> {
        unimplemented!();
        /*if let Some(coverage_context) = self.coverage_context() {
            debug!("getting pgo_func_name_var for instance={:?}", instance);
            let mut pgo_func_name_var_map = coverage_context.pgo_func_name_var_map.borrow_mut();
            pgo_func_name_var_map
                .entry(instance)
                .or_insert_with(|| create_pgo_func_name_var(self, instance))
        } else {
            bug!("Could not get the `coverage_context`");
        }*/
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
    fn define_unused_fn(&self, _def_id: DefId) {
        unimplemented!();
        /*let instance = declare_unused_fn(self, &def_id);
        codegen_unused_fn_and_counter(self, instance);
        add_unused_function_coverage(self, instance, def_id);*/
    }
}

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
    }

    fn add_coverage_counter(&mut self, _instance: Instance<'tcx>, _id: CounterValueReference, _region: CodeRegion) -> bool {
        // TODO(antoyo)
        false
    }

    fn add_coverage_counter_expression(&mut self, _instance: Instance<'tcx>, _id: InjectedExpressionId, _lhs: ExpressionOperandId, _op: Op, _rhs: ExpressionOperandId, _region: Option<CodeRegion>) -> bool {
        // TODO(antoyo)
        false
    }

    fn add_coverage_unreachable(&mut self, _instance: Instance<'tcx>, _region: CodeRegion) -> bool {
        // TODO(antoyo)
        false
    }
}

impl<'gcc, 'tcx> CoverageInfoMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn coverageinfo_finalize(&self) {
        // TODO(antoyo)
    }

    fn get_pgo_func_name_var(&self, _instance: Instance<'tcx>) -> RValue<'gcc> {
        unimplemented!();
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
    }
}

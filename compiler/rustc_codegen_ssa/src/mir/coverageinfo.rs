use crate::traits::*;

use rustc_middle::mir::coverage::*;
use rustc_middle::mir::Coverage;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_coverage(&self, bx: &mut Bx, coverage: Coverage) {
        let Coverage { kind, code_region } = coverage;
        match kind {
            CoverageKind::Counter { function_source_hash, id } => {
                bx.add_coverage_counter(
                    self.instance,
                    function_source_hash,
                    id,
                    code_region.expect("counters always have code regions"),
                );

                let coverageinfo = bx.tcx().coverageinfo(self.instance.def_id());

                let fn_name = bx.create_pgo_func_name_var(self.instance);
                let hash = bx.const_u64(function_source_hash);
                let num_counters = bx.const_u32(coverageinfo.num_counters);
                let id = bx.const_u32(u32::from(id));
                debug!(
                    "codegen intrinsic instrprof.increment(fn_name={:?}, hash={:?}, num_counters={:?}, index={:?})",
                    fn_name, hash, num_counters, id,
                );
                bx.instrprof_increment(fn_name, hash, num_counters, id);
            }
            CoverageKind::Expression { id, lhs, op, rhs } => {
                bx.add_coverage_counter_expression(self.instance, id, lhs, op, rhs, code_region);
            }
            CoverageKind::Unreachable => {
                bx.add_coverage_unreachable(
                    self.instance,
                    code_region.expect("unreachable regions always have code regions"),
                );
            }
        }
    }
}

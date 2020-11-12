use crate::traits::*;

use rustc_middle::mir::coverage::*;
use rustc_middle::mir::Coverage;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_coverage(&self, bx: &mut Bx, coverage: Coverage) {
        let Coverage { kind, code_region } = coverage;
        match kind {
            CoverageKind::Counter { function_source_hash, id } => {
                if bx.set_function_source_hash(self.instance, function_source_hash) {
                    // If `set_function_source_hash()` returned true, the coverage map is enabled,
                    // so continue adding the counter.
                    if let Some(code_region) = code_region {
                        // Note: Some counters do not have code regions, but may still be referenced
                        // from expressions. In that case, don't add the counter to the coverage map,
                        // but do inject the counter intrinsic.
                        bx.add_coverage_counter(self.instance, id, code_region);
                    }

                    let coverageinfo = bx.tcx().coverageinfo(self.instance.def_id());

                    let fn_name = bx.create_pgo_func_name_var(self.instance);
                    let hash = bx.const_u64(function_source_hash);
                    let num_counters = bx.const_u32(coverageinfo.num_counters);
                    let index = bx.const_u32(u32::from(id));
                    debug!(
                        "codegen intrinsic instrprof.increment(fn_name={:?}, hash={:?}, num_counters={:?}, index={:?})",
                        fn_name, hash, num_counters, index,
                    );
                    bx.instrprof_increment(fn_name, hash, num_counters, index);
                }
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

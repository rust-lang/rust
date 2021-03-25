use crate::traits::*;

use rustc_middle::mir::coverage::*;
use rustc_middle::mir::Coverage;
use rustc_middle::mir::SourceScope;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_coverage(&self, bx: &mut Bx, coverage: Coverage, scope: SourceScope) {
        // Determine the instance that coverage data was originally generated for.
        let scope_data = &self.mir.source_scopes[scope];
        let instance = if let Some((inlined_instance, _)) = scope_data.inlined {
            self.monomorphize(inlined_instance)
        } else if let Some(inlined_scope) = scope_data.inlined_parent_scope {
            self.monomorphize(self.mir.source_scopes[inlined_scope].inlined.unwrap().0)
        } else {
            self.instance
        };

        let Coverage { kind, code_region } = coverage;
        match kind {
            CoverageKind::Counter { function_source_hash, id } => {
                if bx.set_function_source_hash(instance, function_source_hash) {
                    // If `set_function_source_hash()` returned true, the coverage map is enabled,
                    // so continue adding the counter.
                    if let Some(code_region) = code_region {
                        // Note: Some counters do not have code regions, but may still be referenced
                        // from expressions. In that case, don't add the counter to the coverage map,
                        // but do inject the counter intrinsic.
                        bx.add_coverage_counter(instance, id, code_region);
                    }

                    let coverageinfo = bx.tcx().coverageinfo(instance.def_id());

                    let fn_name = bx.get_pgo_func_name_var(instance);
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
                bx.add_coverage_counter_expression(instance, id, lhs, op, rhs, code_region);
            }
            CoverageKind::Unreachable => {
                bx.add_coverage_unreachable(
                    instance,
                    code_region.expect("unreachable regions always have code regions"),
                );
            }
        }
    }
}

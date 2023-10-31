use crate::traits::*;

use rustc_middle::mir::Coverage;
use rustc_middle::mir::SourceScope;

use super::FunctionCx;

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_coverage(&self, bx: &mut Bx, coverage: &Coverage, scope: SourceScope) {
        // Determine the instance that coverage data was originally generated for.
        let instance = if let Some(inlined) = scope.inlined_instance(&self.mir.source_scopes) {
            self.monomorphize(inlined)
        } else {
            self.instance
        };

        // Handle the coverage info in a backend-specific way.
        bx.add_coverage(instance, coverage);
    }
}

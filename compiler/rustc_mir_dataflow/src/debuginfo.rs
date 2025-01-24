use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;

/// Return the set of locals that appear in debuginfo.
pub fn debuginfo_locals(body: &Body<'_>) -> DenseBitSet<Local> {
    let mut visitor = DebuginfoLocals(DenseBitSet::new_empty(body.local_decls.len()));
    for debuginfo in body.var_debug_info.iter() {
        visitor.visit_var_debug_info(debuginfo);
    }
    visitor.0
}

struct DebuginfoLocals(DenseBitSet<Local>);

impl Visitor<'_> for DebuginfoLocals {
    fn visit_local(&mut self, local: Local, _: PlaceContext, _: Location) {
        self.0.insert(local);
    }
}

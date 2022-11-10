use crate::dep_graph::DepContext;
use crate::query::QueryInfo;

pub trait Value<Tcx: DepContext>: Sized {
    fn from_cycle_error(tcx: Tcx, cycle: &[QueryInfo]) -> Self;
}

impl<Tcx: DepContext, T> Value<Tcx> for T {
    default fn from_cycle_error(tcx: Tcx, _: &[QueryInfo]) -> T {
        tcx.sess().abort_if_errors();
        // Ideally we would use `bug!` here. But bug! is only defined in rustc_middle, and it's
        // non-trivial to define it earlier.
        panic!("Value::from_cycle_error called without errors");
    }
}

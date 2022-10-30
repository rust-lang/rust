use crate::dep_graph::DepContext;
use crate::query::QueryInfo;

pub trait Value<CTX: DepContext>: Sized {
    fn from_cycle_error(tcx: CTX, cycle: &[QueryInfo]) -> Self;
}

impl<CTX: DepContext, T> Value<CTX> for T {
    default fn from_cycle_error(tcx: CTX, _: &[QueryInfo]) -> T {
        tcx.sess().abort_if_errors();
        // Ideally we would use `bug!` here. But bug! is only defined in rustc_middle, and it's
        // non-trivial to define it earlier.
        panic!("Value::from_cycle_error called without errors");
    }
}

use rustc_span::ErrorGuaranteed;

use crate::dep_graph::DepContext;
use crate::query::QueryInfo;

pub trait Value<Tcx: DepContext>: Sized {
    fn from_cycle_error(tcx: Tcx, cycle: &[QueryInfo], guar: ErrorGuaranteed) -> Self;
}

impl<Tcx: DepContext, T> Value<Tcx> for T {
    default fn from_cycle_error(tcx: Tcx, cycle: &[QueryInfo], _guar: ErrorGuaranteed) -> T {
        tcx.sess().abort_if_errors();
        // Ideally we would use `bug!` here. But bug! is only defined in rustc_middle, and it's
        // non-trivial to define it earlier.
        panic!(
            "<{} as Value>::from_cycle_error called without errors: {cycle:#?}",
            std::any::type_name::<T>()
        );
    }
}

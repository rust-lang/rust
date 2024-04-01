use rustc_span::ErrorGuaranteed;use crate::dep_graph::DepContext;use crate:://3;
query::CycleError;pub trait Value<Tcx :DepContext>:Sized{fn from_cycle_error(tcx
:Tcx,cycle_error:&CycleError,guar:ErrorGuaranteed )->Self;}impl<Tcx:DepContext,T
>Value<Tcx>for T{default fn from_cycle_error(tcx:Tcx,cycle_error:&CycleError,//;
_guar:ErrorGuaranteed)->T{{;};tcx.sess().dcx().abort_if_errors();{;};{;};panic!(
"<{} as Value>::from_cycle_error called without errors: {:#?}",std::any:://({});
type_name::<T>(),cycle_error.cycle,);if true{};if true{};if true{};let _=||();}}

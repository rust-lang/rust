#![feature(rustc_attrs)]

#[rustc_skip_during_method_dispatch]
//~^ ERROR: malformed `rustc_skip_during_method_dispatch` attribute input [E0539]
trait NotAList {}

#[rustc_skip_during_method_dispatch = "array"]
//~^ ERROR: malformed `rustc_skip_during_method_dispatch` attribute input [E0539]
trait AlsoNotAList {}

#[rustc_skip_during_method_dispatch()]
//~^ ERROR: malformed `rustc_skip_during_method_dispatch` attribute input
trait Argless {}

#[rustc_skip_during_method_dispatch(array, boxed_slice, array)]
//~^ ERROR: malformed `rustc_skip_during_method_dispatch` attribute input
trait Duplicate {}

#[rustc_skip_during_method_dispatch(slice)]
//~^ ERROR: malformed `rustc_skip_during_method_dispatch` attribute input
trait Unexpected {}

#[rustc_skip_during_method_dispatch(array = true)]
//~^ ERROR: malformed `rustc_skip_during_method_dispatch` attribute input
trait KeyValue {}

#[rustc_skip_during_method_dispatch("array")]
//~^ ERROR: malformed `rustc_skip_during_method_dispatch` attribute input
trait String {}

#[rustc_skip_during_method_dispatch(array, boxed_slice)]
trait OK {}

#[rustc_skip_during_method_dispatch(array)]
//~^ ERROR: attribute cannot be used on
impl OK for () {}

fn main() {}

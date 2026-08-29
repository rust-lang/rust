//@ revisions: gate_off gate_on
//@ dont-require-annotations: HELP

#![cfg_attr(gate_on, feature(core_intrinsics))]
fn main() {
    let _ = transmute::<usize>();
    //~^ ERROR cannot find function `transmute` in this scope

    let _ = fabs(1.0);
    //~^ ERROR cannot find function `fabs` in this scope
    //[gate_on]~| HELP consider importing this function
}

// compile-pass
// aux-build:const_fn_lib.rs

extern crate const_fn_lib;

fn main() {
    const_fn_lib::bar()();
    const_fn_lib::bar_inlined()();
    const_fn_lib::bar_inlined_always()();
}

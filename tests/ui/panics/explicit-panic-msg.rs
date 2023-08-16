#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(non_fmt_panics)]

// run-fail
//@error-in-other-file:wooooo
//@ignore-target-emscripten no processes

fn main() {
    let mut a = 1;
    if 1 == 1 {
        a = 2;
    }
    panic!(format!("woooo{}", "o"));
}

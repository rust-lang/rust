//@ check-pass
//@ revisions: warn allow
//@ compile-flags: -Z tiny-const-eval-limit

#![cfg_attr(warn, warn(long_running_const_eval))]
#![cfg_attr(allow, allow(long_running_const_eval))]

const fn simple_loop(n: u32) -> u32 {
    let mut index = 0;
    while index < n {
        //~^ WARN is taking a long time
        //[warn]~| WARN is taking a long time
        //[warn]~| WARN is taking a long time
        index = index + 1;
    }
    0
}

const X: u32 = simple_loop(19);
const Y: u32 = simple_loop(35);

fn main() {
    println!("{X}");
}

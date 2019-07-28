// run-pass
#![allow(dead_code)]
// Just testing that panic!() type checks in statement or expr


#![allow(unreachable_code)]

fn f() {
    panic!();

    let _x: isize = panic!();
}

pub fn main() {

}

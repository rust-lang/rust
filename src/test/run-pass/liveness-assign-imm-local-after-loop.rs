// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

fn test(_cond: bool) {
    let v: isize;
    v = 1;
    loop { } // loop never terminates, so no error is reported
    v = 2;
}

pub fn main() {
    // note: don't call test()... :)
}

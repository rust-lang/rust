//@ check-pass
// testcase for https://github.com/rust-lang/rust/issues/143256

#![deny(unused_parens)]
#![allow(unreachable_code, unused_variables, dead_code)]

fn foo() {
    let _x = || 'outer: loop {
        let inner = 'inner: loop {
            let i = Default::default();
            // the parentheses here are necessary
            if (break 'outer i) {
                loop {
                    break 'inner 5i8;
                }
            } else if true {
                break 'inner 6;
            }
            break 7;
        };
        break inner < 8;
    };
}

fn main() {}

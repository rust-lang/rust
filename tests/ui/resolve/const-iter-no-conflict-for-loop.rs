//! regression test for <https://github.com/rust-lang/rust/issues/27639>
//! Ensure that a constant named `iter` does not
//! interfere with the name resolution of the `iter` methods used internally
//! by `for` loops
//@ run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

fn main() {
    const iter: i32 = 0;

    for i in 1..10 {
        println!("{}", i);
    }
}

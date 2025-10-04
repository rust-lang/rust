// Regression test for <https://github.com/rust-lang/rust/issues/147303>.

#![deny(unused_assignments, unused_variables)]

mod m1 {
    const _MAX_FMTVER_X1X_EVENTNUM: i32 = 0;
}

mod m2 {
    fn fun(rough: i32) {} //~ERROR unused variable
}

fn main() {}

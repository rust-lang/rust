//! Regression test for https://github.com/rust-lang/rust/issues/11958

//@ run-pass

// We shouldn't need to rebind a moved upvar as mut if it's already
// marked as mut

pub fn main() {
    let mut x = 1;
    let _thunk = Box::new(move|| { x = 2; });
    //~^ WARN value assigned to `x` is never read
    //~| WARN unused variable: `x`
}

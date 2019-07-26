// run-pass
#![forbid(warnings)]

// We shouldn't need to rebind a moved upvar as mut if it's already
// marked as mut

pub fn main() {
    let mut x = 1;
    let _thunk = Box::new(move|| { x = 2; });
}

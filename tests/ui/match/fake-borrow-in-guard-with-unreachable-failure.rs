//! Test to document strange behavior related to <https://github.com/rust-lang/rust/issues/161578>:
//! in these gaurds, the guard's failure path is unreachable after mutating `x`. This means the fake
//! borrow on `x` is unreachable (and therefore dead) when assigning to `x`, so `x` can be
//! overwritten. This should be sound since we can't continue matching after diverging.
//@ check-pass

fn main() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    match x {
        Some(_) if { x = None; return } => {}
        Some(_) if true || { x = None; return } => {}
        Some(_) if { x = None; false } || return => {}
        Some(_) if false && ({ x = None; false } || return) => {}
        Some(b) => println!("{b}"),
        None => println!("none"),
    }
}

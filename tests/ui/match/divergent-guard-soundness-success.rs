//! Test to document strange behavior related to <https://github.com/rust-lang/rust/issues/161578>:
//! in these gaurds, the guard's success and failures are both unreachable after mutating `x`. This
//! means the fake reads on the fake borrow on `x` are unreachable when assigning to `x`, so `x` can
//! be overwritten. This is fine since we don't create bindings and don't continue matching.
//@ check-pass

fn main() {
    let mut x: Option<&u64> = Some(&7);
    match x {
        Some(&y) if { x = None; return } => {}
        Some(&y) if true || { x = None; return } => {}
        Some(b) => println!("{b}"),
        None => println!("none"),
    }
}

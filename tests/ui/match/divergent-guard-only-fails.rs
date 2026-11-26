//! Regression test for <https://github.com/rust-lang/rust/issues/161578>: the fake borrow on `x`
//! below is ignored because the fake read keeping it live is only present at the start of the arm
//! body, which is unreachable after `x` is mutated.
//@ check-pass
//@ known-bug: #161578

fn main() {
    let mut x: Option<u64> = Some(7);
    match x {
        // These guards can fail but not succeed after assigning `x = None`.
        Some(_) if { x = None; false } && return => {}
        Some(_) if false || ({ x = None; false } && return) => {}

        // This arm will be taken, with `x` now being `None`.
        Some(b) => println!("{b}"),
        None => {}
    }

    let mut a: &[u8] = &[7];
    match a[0] {
        // These guards can fail but not succeed after assigning `a = &[]`.
        _ if { a = &[]; false } && return => {}
        _ if false || ({ a = &[]; false } && return) => {}

        // This arm will be taken, with `a` now being `&[]`.
        b => println!("{b}"),
    }
}

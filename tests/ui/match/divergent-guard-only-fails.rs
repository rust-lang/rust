//! Regression test for <https://github.com/rust-lang/rust/issues/161578>: the fake borrow on `x`
//! below was ignored because the fake read keeping it live was only present at the start of the arm
//! body, which is unreachable after `x` is mutated.

fn main() {
    let mut x: Option<u64> = Some(7);
    match x {
        // These guards can fail but not succeed after assigning `x = None`.
        Some(_) if { x = None; false } && return => {}
        //~^ ERROR: cannot assign `x` in match guard
        Some(_) if false || ({ x = None; false } && return) => {}
        //~^ ERROR: cannot assign `x` in match guard

        // This arm would have been taken, with `x` now being `None`.
        Some(b) => println!("{b}"),
        None => {}
    }

    let mut a: &[u8] = &[7];
    match a[0] {
        // These guards can fail but not succeed after assigning `a = &[]`.
        _ if { a = &[]; false } && return => {}
        //~^ ERROR: cannot assign `a` in match guard
        _ if false || ({ a = &[]; false } && return) => {}
        //~^ ERROR: cannot assign `a` in match guard

        // This arm would have been taken, with `a` now being `&[]`.
        b => println!("{b}"),
    }
}

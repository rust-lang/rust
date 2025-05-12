//@ run-rustfix
//@ rustfix-only-machine-applicable
//@ check-pass

// See <https://github.com/rust-lang/rust/issues/121330>.

fn cmp<T: ?Sized>(a: *mut T, b: *mut T) -> bool {
    let _ = a == b;
    //~^ WARN ambiguous wide pointer comparison
    panic!();
}

fn main() {}

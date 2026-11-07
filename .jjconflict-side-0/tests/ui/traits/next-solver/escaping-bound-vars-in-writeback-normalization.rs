//@ compile-flags: -Znext-solver
//@ check-pass

trait Trivial {
    type Assoc;
}

impl<T: ?Sized> Trivial for T {
    type Assoc = ();
}

fn main() {
    // During writeback, we call `normalize_erasing_regions`, which will walk past
    // the `for<'a>` binder and try to normalize `<&'a () as Trivial>::Assoc` directly.
    // We need to handle this case in the new deep normalizer similarly to how it
    // is handled in the old solver.
    let x: Option<for<'a> fn(<&'a () as Trivial>::Assoc)> = None;
}

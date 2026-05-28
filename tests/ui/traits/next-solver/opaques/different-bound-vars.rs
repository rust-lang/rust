// Check whether we support defining uses with different bound vars.
// This needs to handle both mismatches for the same opaque type storage
// entry, but also between different entries.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

fn foo<T, U>(b: bool) -> impl Sized {
    if b {
        let _: for<'a> fn(&'a ()) = foo::<T, U>(false);
        let _: for<'b> fn(&'b ()) = foo::<U, T>(false);
        //[current]~^ ERROR concrete type differs from previous defining opaque type use
    }

    (|&()| ()) as for<'c> fn(&'c ())
}

fn main() {}

// Ensure that you cannot use generic types to return a region outside
// of its bound.  Here, in the `return_it()` fn, we call with() but
// with R bound to &int from the return_it.  Meanwhile, with()
// provides a value that is only good within its own stack frame. This
// used to successfully compile because we failed to account for the
// fact that fn(x: &int) rebound the region &.

fn with<R>(f: fn(x: &int) -> R) -> R {
    f(&3)
}

fn return_it() -> &int {
    with(|o| o) //~ ERROR mismatched types
        //~^ ERROR reference is not valid outside of its lifetime
}

fn main() {
    let x = return_it();
    debug!{"foo=%d", *x};
}

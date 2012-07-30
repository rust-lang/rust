// Similar to regions-ret-borrowed.rs, but using a named lifetime.  At
// some point regions-ret-borrowed reported an error but this file did
// not, due to special hardcoding around the anonymous region.

fn with<R>(f: fn(x: &a/int) -> R) -> R {
    f(&3)
}

fn return_it() -> &a/int {
    with(|o| o) //~ ERROR mismatched types
        //~^ ERROR reference is not valid outside of its lifetime
}

fn main() {
    let x = return_it();
    debug!{"foo=%d", *x};
}

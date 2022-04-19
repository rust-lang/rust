// Ensure that you cannot use generic types to return a region outside
// of its bound.  Here, in the `return_it()` fn, we call with() but
// with R bound to &isize from the return_it.  Meanwhile, with()
// provides a value that is only good within its own stack frame. This
// used to successfully compile because we failed to account for the
// fact that fn(x: &isize) rebound the region &.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn with<R, F>(f: F) -> R where F: FnOnce(&isize) -> R {
    f(&3)
}

fn return_it<'a>() -> &'a isize {
    with(|o| o)
    //[base]~^ ERROR cannot infer an appropriate lifetime due to conflicting requirements [E0495]
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
    let x = return_it();
    println!("foo={}", *x);
}

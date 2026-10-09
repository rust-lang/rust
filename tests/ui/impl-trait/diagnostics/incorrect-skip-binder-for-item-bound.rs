//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #139381
//@ needs-rustc-debug-assertions

// An ICE caused by an incorrect `skip_binder()` in diagnostics code, see #139381.

trait A<'a> {
    type Assoc: ?Sized;
}

impl<'a> A<'a> for () {
    type Assoc = &'a ();
}

fn hello() -> impl for<'a> A<'a, Assoc: Into<u8> + 'static + Copy> {
    //[next]~^ ERROR: the trait bound `for<'a> u8: From<&'a ()>` is not satisfied
    ()
}

fn main() {}

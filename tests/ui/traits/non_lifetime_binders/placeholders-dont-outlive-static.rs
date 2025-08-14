//@ revisions: good bad

//@[good] known-bug: unknown
// `for<T> T: 'static` doesn't imply itself when processing outlives obligations

#![feature(non_lifetime_binders)]
//[bad]~^ WARN the feature `non_lifetime_binders` is incomplete

fn foo() where for<T> T: 'static {}

#[cfg(bad)]
fn bad() {
    foo();
    //[bad]~^ ERROR the placeholder type `T` may not live long enough
}

#[cfg(good)]
fn good() where for<T> T: 'static {
    foo();
}

fn main() {}

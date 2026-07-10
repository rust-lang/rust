//! Regression test for <https://github.com/rust-lang/rust/issues/35976>.
//! This used to emit spurious `Sized` bound unsatisfied error when trait
//! was not imported, instead of suggestion to import it.
//@ edition:2015
//@ revisions: imported unimported
//@[imported] check-pass

mod private {
    pub trait Future {
        //[unimported]~^^ HELP perhaps add a `use` for it
        fn wait(&self) where Self: Sized;
    }

    impl Future for Box<dyn Future> {
        fn wait(&self) { }
    }
}

#[cfg(imported)]
use private::Future;

fn bar(arg: Box<dyn private::Future>) {
    // Importing the trait means that we don't autoderef `Box<dyn Future>`
    arg.wait();
    //[unimported]~^ ERROR the `wait` method cannot be invoked on a trait object
}

fn main() {}

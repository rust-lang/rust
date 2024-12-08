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

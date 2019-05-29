mod private {
    pub trait Future {
        fn wait(&self) where Self: Sized;
    }

    impl Future for Box<dyn Future> {
        fn wait(&self) { }
    }
}

//use private::Future;

fn bar(arg: Box<dyn private::Future>) {
    arg.wait();
    //~^ ERROR the `wait` method cannot be invoked on a trait object
}

fn main() {

}

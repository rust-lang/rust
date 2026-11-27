//@ revisions: current next
//@[next] compile-flags: -Znext-solver

trait MyDebug {
    fn my_debug(&self);
}

impl<T> MyDebug for &T
where
    T: std::fmt::Debug,
{
    fn my_debug(&self) {}
}

fn my_foo() -> impl std::fmt::Debug {
    if false {
        let x = my_foo();
        x.my_debug();
        //~^ ERROR no method named `my_debug` found
    }
    ()
}

fn main() {}

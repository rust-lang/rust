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
        //[next]~^ type annotations needed
        x.my_debug();
        //[current]~^ no method named `my_debug` found
    }
    ()
}

fn my_bar() -> impl std::fmt::Debug {
    if false {
        let x = &my_bar();
        //[next]~^ type annotations needed
        x.my_debug();
    }
    ()
}

fn main() {}

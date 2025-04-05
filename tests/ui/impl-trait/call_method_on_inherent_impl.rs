//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[current] check-pass

trait MyDebug {
    fn my_debug(&self);
}

impl<T> MyDebug for T
where
    T: std::fmt::Debug,
{
    fn my_debug(&self) {}
}

fn my_foo() -> impl std::fmt::Debug {
    if false {
        let x = my_foo();
        //[next]~^ ERROR type annotations needed
        x.my_debug();
    }
    ()
}

fn main() {}

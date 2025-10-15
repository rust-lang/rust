//@ revisions: current next
//@[next] compile-flags: -Znext-solver

trait MyDebug {
    fn my_debug(&self);
}

impl MyDebug for &() {
    fn my_debug(&self) {}
}

fn my_foo() -> impl std::fmt::Debug {
    if false {
        let x = &my_foo();
        x.my_debug();
        //~^ ERROR: no method named `my_debug`
    }
    ()
}

fn main() {}

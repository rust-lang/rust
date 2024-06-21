//@ edition:2021


struct Wrapper<T>(T);

trait Foo {
    fn bar() -> Wrapper<Missing<impl Sized>>;
    //~^ ERROR: cannot find type `Missing` [E0412]
}

impl Foo for () {
    fn bar() -> Wrapper<i32> {
        Wrapper(0)
    }
}

fn main() {}

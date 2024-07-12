//@ edition:2021


struct Wrapper<T>(T);

trait Foo {
    fn bar() -> Wrapper<Missing<impl Sized>>;
    //~^ ERROR: cannot find type `Missing`
}

impl Foo for () {
    fn bar() -> Wrapper<i32> {
        Wrapper(0)
    }
}

fn main() {}

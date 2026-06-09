struct Wrapper<T>(T);

trait Foo {
    fn bar() -> Wrapper<impl Sized>;
}

impl Foo for () {
    fn bar() -> i32 {
        //~^ ERROR method `bar` has an incompatible return type for trait
        0
    }
}

fn main() {}

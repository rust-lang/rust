//@ run-rustfix

struct Foo;

fn From<i32> for Foo {
    //~^ ERROR you might have meant to write `impl` instead of `fn`
    fn from(_a: i32) -> Self {
        Foo
    }
}

fn main() {}

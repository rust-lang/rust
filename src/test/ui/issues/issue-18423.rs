// Test that `Box` cannot be used with a lifetime argument.

struct Foo<'a> {
    x: Box<'a, isize>
    //~^ ERROR this struct takes 0 lifetime arguments but 1 lifetime argument was supplied
}

fn main() { }

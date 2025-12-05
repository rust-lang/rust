//@ run-pass
// Test that the cache results from the default method do not pollute
// the cache for the later call in `load()`.
//
// See issue #18209.


pub trait Foo {
    fn load_from() -> Box<Self>;
    fn load() -> Box<Self> {
        Foo::load_from()
    }
}

pub fn load<M: Foo>() -> Box<M> {
    Foo::load()
}

fn main() { }

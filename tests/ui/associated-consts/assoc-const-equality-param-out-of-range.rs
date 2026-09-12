//! Regression test for https://github.com/rust-lang/rust/issues/145824.
//!
//! An associated const equality bound whose value refers to the impl's own const parameter
//! used to ICE with "type parameter out of range when instantiating".

trait Widget {
    const WIDTH: usize;
}

struct Boxed<const WIDTH: usize, T> {
    inner: T,
}

impl<const WIDTH: usize, T> Boxed<WIDTH, T> {
    fn new(_: T) -> Self
    where
        T: Widget<WIDTH = { WIDTH }>,
        //~^ ERROR associated const equality is incomplete
    {
        loop {}
    }

    fn empty<const X: usize>() -> Boxed<X, Empty<{ X }>> {
        Boxed::new(Empty)
    }
}

struct Empty<const WIDTH: usize>;

impl<const WIDTH: usize> Widget for Empty<WIDTH> {
    const WIDTH: usize = WIDTH;
}

fn main() {}

trait Foo {
    type Bar;
}

unsafe fn noop<F: Foo>(foo: F::Bar) -> F::Bar {
    ::std::mem::transmute(foo) //~ ERROR cannot transmute between types of different sizes
}

fn main() {}

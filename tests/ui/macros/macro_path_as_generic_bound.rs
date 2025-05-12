trait Foo {}

macro_rules! foo(($t:path) => {
    impl<T: $t> Foo for T {}
});

foo!(m::m2::A); //~ ERROR failed to resolve

fn main() {}

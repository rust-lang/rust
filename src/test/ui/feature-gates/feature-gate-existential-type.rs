// Check that existential types must be ungated to use the `existential` keyword

existential type Foo: std::fmt::Debug; //~ ERROR existential types are unstable

trait Bar {
    type Baa: std::fmt::Debug;
    fn define() -> Self::Baa;
}

impl Bar for () {
    existential type Baa: std::fmt::Debug; //~ ERROR existential types are unstable
    fn define() -> Self::Baa { 0 }
}

fn define() -> Foo { 0 }

fn main() {}

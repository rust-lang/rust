// Ensure that `impl` blocks with different mistakes regarding const parameters are properly
// handled. Issue #91119.
mod a {
    struct Foo<const AHA: usize>{}

    impl Foo {} //~ ERROR: missing generics for struct `a::Foo`
}

mod b {
    struct Foo<const AHA: usize>{}

    impl Foo<AHA> {} //~ ERROR: cannot find const `AHA` in this scope
}

mod c {
    struct Foo<const AHA: usize>{}

    impl<AHA> Foo<AHA> {} //~ ERROR: type provided when a constant was expected
}

mod d {
    struct Foo<const AHA: usize>{}

    impl Foo<{ AHA }> {} //~ ERROR: cannot find value `AHA` in this scope
}

mod e {
    struct Foo<const AHA: usize>{}

    impl<const AHA: u32> Foo<AHA> {} //~ ERROR: the constant `AHA` is not of type `usize`
}

mod f {
    struct Foo<const AHA: usize>{}
    trait T {}

    impl T for Foo {} //~ ERROR: missing generics for struct `f::Foo`
}

mod g {
    struct Foo {}
    trait T<const AHA: usize> {}

    impl T for Foo {} //~ ERROR: missing generics for trait `g::T`
}

mod h {
    struct Foo {}
    trait T<const AHA: usize> {}

    impl T<AHA> for Foo {} //~ ERROR: cannot find type `AHA` in this scope
}

mod i {
    struct Foo {}
    trait T<const AHA: usize> {}

    impl<AHA> T<AHA> for Foo {} //~ ERROR: type provided when a constant was expected
}

mod j {
    struct Foo {}
    trait T<const AHA: usize> {}

    impl T<{ AHA }> for Foo {} //~ ERROR: cannot find value `AHA` in this scope
}

mod k {
    struct Foo {}
    trait T<const AHA: usize> {}

    impl<const AHA: u32> T<AHA> for Foo {} //~ ERROR: the constant `AHA` is not of type `usize`
}

fn main() {}

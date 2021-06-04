struct Foo<T = impl Copy>(T);
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

type Result<T, E = impl std::error::Error> = std::result::Result<T, E>;
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

// should not cause ICE
fn x() -> Foo {
    Foo(0)
}

fn main() -> Result<()> {}

struct Foo<T = impl Copy>(T);
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults

type Result<T, E = impl std::error::Error> = std::result::Result<T, E>;
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults

// should not cause ICE
fn x() -> Foo {
    Foo(0)
}

fn main() -> Result<()> {}

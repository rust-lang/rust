struct Foo<T = impl Copy>(T);
//~^ ERROR `impl Trait` isn't allowed within type [E0562]

type Result<T, E = impl std::error::Error> = std::result::Result<T, E>;
//~^ ERROR `impl Trait` isn't allowed within type [E0562]

// should not cause ICE
fn x() -> Foo {
    Foo(0)
}

fn main() -> Result<()> {}

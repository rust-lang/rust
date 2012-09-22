use a::Foo;

mod a {
    #[legacy_exports];
    struct Foo {
        priv x: int
    }

    fn make() -> Foo {
        Foo { x: 3 }
    }
}

fn main() {
    match a::make() {
        Foo { x: _ } => {}  //~ ERROR field `x` is private
    }
}


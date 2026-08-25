//! regression test for <https://github.com/rust-lang/rust/issues/17405>
enum Foo {
    Bar(isize)
}

fn main() {
    match Foo::Bar(1) {
        Foo { i } => () //~ ERROR expected struct, variant or union type, found enum `Foo`
    }
}

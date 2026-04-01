//! regression test for https://github.com/rust-lang/rust/issues/23253
enum Foo {
    Bar,
}

fn main() {
    Foo::Bar.a;
    //~^ ERROR no field `a` on type `Foo`
}

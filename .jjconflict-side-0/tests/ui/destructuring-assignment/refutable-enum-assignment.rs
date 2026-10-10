// Regression test for <https://github.com/rust-lang/rust/issues/157553>.

enum Foo {
    One,
    Two,
}

fn main() {
    Foo::One = Foo::One;
    //~^ ERROR refutable pattern in assignment
}

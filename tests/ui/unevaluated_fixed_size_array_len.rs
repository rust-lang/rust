// https://github.com/rust-lang/rust/issues/49208

trait Foo {
    fn foo();
}

impl Foo for [(); 1] {
    fn foo() {}
}

fn main() {
    <[(); 0] as Foo>::foo() //~ ERROR E0277
}

trait Foo {
    fn foo<T>(x: T);
}

impl Foo for bool {
    fn foo<T>(x: T) where T: Copy {} //~ ERROR E0276
}

fn main() {
}

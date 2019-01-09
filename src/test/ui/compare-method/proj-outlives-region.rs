// Test that we elaborate `Type: 'region` constraints and infer various important things.

trait Master<'a, T: ?Sized, U> {
    fn foo() where T: 'a;
}

// `U::Item: 'a` does not imply that `U: 'a`
impl<'a, U: Iterator> Master<'a, U::Item, U> for () {
    fn foo() where U: 'a { } //~ ERROR E0276
}

fn main() {
    println!("Hello, world!");
}

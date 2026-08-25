struct Foo<'a>(&'a ());

// We need a lifetime in scope or else we do not write a user type annotation as a fast-path.
impl<'a> Foo<'a> {
    fn bar<const V: u8>() {
        let V;
        //~^ ERROR constant parameters cannot be referenced in patterns
    }
}
fn main() {}

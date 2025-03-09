struct Foo;

impl<'b> Foo {
    fn bar<const V: u8>() {

        let V; //~ ERROR constant parameters cannot be referenced in patterns
    }
}
fn main() {

}

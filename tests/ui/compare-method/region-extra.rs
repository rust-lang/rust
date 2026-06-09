// Test that you cannot add an extra where clause in the impl relating
// two regions.

trait Master<'a, 'b> {
    fn foo();
}

impl<'a, 'b> Master<'a, 'b> for () {
    fn foo() where 'a: 'b { } //~ ERROR impl has stricter
}

fn main() {
    println!("Hello, world!");
}

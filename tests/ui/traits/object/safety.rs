// Check that static methods render the trait dyn-incompatible.

trait Tr {
    fn foo();
    fn bar(&self) { }
}

struct St;

impl Tr for St {
    fn foo() {}
}

fn main() {
    let _: &dyn Tr = &St; //~ ERROR E0038
}

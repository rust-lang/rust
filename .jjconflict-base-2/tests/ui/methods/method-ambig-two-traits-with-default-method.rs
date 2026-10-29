// Test that we correctly report an ambiguity where two applicable traits
// are in scope and the method being invoked is a default method not
// defined directly in the impl.

trait Foo { fn method(&self) {} }
trait Bar { fn method(&self) {} }

impl Foo for usize {}
impl Bar for usize {}

fn main() {
    1_usize.method(); //~ ERROR E0034
}

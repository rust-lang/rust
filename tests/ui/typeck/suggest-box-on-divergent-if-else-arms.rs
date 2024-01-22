// run-rustfix
trait Trait {}
struct Struct;
impl Trait for Struct {}
fn foo() -> Box<dyn Trait> {
    Box::new(Struct)
}
fn main() {
    let _ = if true {
        foo()
    } else {
        Struct //~ ERROR E0308
    };
}

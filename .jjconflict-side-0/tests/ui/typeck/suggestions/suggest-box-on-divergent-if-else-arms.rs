//@ run-rustfix
trait Trait {}
struct Struct;
impl Trait for Struct {}
fn foo() -> Box<dyn Trait> {
    Box::new(Struct)
}
fn bar() -> impl Trait {
    Struct
}
fn main() {
    let _ = if true {
        Struct
    } else {
        foo() //~ ERROR E0308
    };
    let _ = if true {
        foo()
    } else {
        Struct //~ ERROR E0308
    };
    let _ = if true {
        Struct
    } else {
        bar() //~ ERROR E0308
    };
    let _ = if true {
        bar()
    } else {
        Struct //~ ERROR E0308
    };
}

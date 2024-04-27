trait Trait {}
struct Struct;
impl Trait for Struct {}
fn foo() -> impl Trait {
    Struct
}
fn main() {
    let a: Box<dyn Trait> = if true {
        Box::new(Struct)
    } else {
        foo() //~ ERROR E0308
    };
    let a: dyn Trait = if true {
        Struct //~ ERROR E0308
    } else {
        foo() //~ ERROR E0308
    };
}

//@ aux-build:namespaced_enums.rs
extern crate namespaced_enums;

fn main() {
    let _ = namespaced_enums::A;
    //~^ ERROR cannot find value `A`
    let _ = namespaced_enums::B(10);
    //~^ ERROR cannot find function, tuple struct or tuple variant `B`
    let _ = namespaced_enums::C { a: 10 };
    //~^ ERROR cannot find struct, variant or union type `C`
}

pub enum EFoo { A, B, C, D }

pub trait Foo {
    const X: EFoo;
}

struct Abc;

impl Foo for Abc {
    const X: EFoo = EFoo::B;
}

struct Def;
impl Foo for Def {
    const X: EFoo = EFoo::D;
}

pub fn test<A: Foo, B: Foo>(arg: EFoo) {
    match arg {
        A::X => println!("A::X"),
        //~^ ERROR constant pattern cannot depend on generic parameters
        B::X => println!("B::X"),
        //~^ ERROR constant pattern cannot depend on generic parameters
        _ => (),
    }
}

pub fn test_let_pat<A: Foo, B: Foo>(arg: EFoo, A::X: EFoo) {
    //~^ ERROR constant pattern cannot depend on generic parameters
    let A::X = arg;
    //~^ ERROR constant pattern cannot depend on generic parameters
}

fn main() {
}

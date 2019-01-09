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
        //~^ error: associated consts cannot be referenced in patterns [E0158]
        B::X => println!("B::X"),
        //~^ error: associated consts cannot be referenced in patterns [E0158]
        _ => (),
    }
}

fn main() {
}

mod b {
    #[derive(Default)]
    pub struct A(u32);
}

impl b::A {
    fn inherent_bypass(&self) {
        let Self(x) = self;
        //~^ ERROR: tuple struct constructor `A` is private
        println!("{x}");
    }
}

pub trait B {
    fn f(&self);
}

impl B for b::A {
    fn f(&self) {
        let Self(a) = self;
        //~^ ERROR: tuple struct constructor `A` is private
        println!("{}", a);
    }
}

pub trait Projector {
    type P;
}

impl Projector for () {
    type P = b::A;
}

pub trait Bypass2 {
    fn f2(&self);
}

impl Bypass2 for <() as Projector>::P {
    fn f2(&self) {
        let Self(a) = self;
        //~^ ERROR: tuple struct constructor `A` is private
        println!("{}", a);
    }
}

fn main() {}

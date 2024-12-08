pub trait Foo {
    const Y: usize;
}

struct Abc;
impl Foo for Abc {
    const Y: usize = 8;
}

struct Def;
impl Foo for Def {
    const Y: usize = 33;
}

pub fn test<A: Foo, B: Foo>() {
    let _array: [u32; <A as Foo>::Y];
    //~^ ERROR generic parameters may not be used
}

fn main() {}

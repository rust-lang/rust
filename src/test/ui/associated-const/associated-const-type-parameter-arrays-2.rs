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
    let _array = [4; <A as Foo>::Y];
    //~^ ERROR type parameters can't appear within an array length expression [E0447]
}

fn main() {
}

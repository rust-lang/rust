struct A<T>(T);

trait Foo {
    type B;
}

impl Foo for A<u32> {
    type B = i32;
}

impl Foo for A<i32> {
    type B = i32;
}

fn main() {
    A::B::<>::C
    //~^ ERROR ambiguous associated type
}

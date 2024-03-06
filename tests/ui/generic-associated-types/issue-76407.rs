//@ check-pass

trait Marker {}

impl Marker for u32 {}

trait MyTrait {
    type Item<'a>;
}

struct MyStruct;

impl MyTrait for MyStruct {
    type Item<'a> = u32;
}

fn ty_check<T>()
where
    T: MyTrait,
    for<'a> T::Item<'a>: Marker
{}

fn main() {
    ty_check::<MyStruct>();
}

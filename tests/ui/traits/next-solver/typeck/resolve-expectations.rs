//@ check-pass
//@ compile-flags: -Znext-solver

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

fn id<T>(t: T) -> T { t }

trait Foo {}
impl Foo for i32 {}
impl Foo for u32 {}

fn main() {
    // Make sure we resolve expected pointee of addr-of.
    id::<<&&dyn Foo as Mirror>::Assoc>(&id(&1));

    // Make sure we resolve expected element of array.
    id::<<[Box<dyn Foo>; 2] as Mirror>::Assoc>([Box::new(1i32), Box::new(1u32)]);

    // Make sure we resolve expected element of tuple.
    id::<<(Box<dyn Foo>,) as Mirror>::Assoc>((Box::new(1i32),));
}

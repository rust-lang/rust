// compile-flags: -Ztrait-solver=next
// check-pass

#![feature(unsized_tuple_coercion)]

trait Foo {}

impl Foo for i32 {}

fn main() {
    // Unsizing via struct
    let _: Box<dyn Foo> = Box::new(1i32);

    // Slice unsizing
    let y = [1, 2, 3];
    let _: &[i32] = &y;

    // Tuple unsizing
    let hi = (1i32,);
    let _: &(dyn Foo,) = &hi;

    // Dropping auto traits
    let a: &(dyn Foo + Send) = &1;
    let _: &dyn Foo = a;
}

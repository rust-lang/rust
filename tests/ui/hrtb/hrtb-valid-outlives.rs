// Test that valid HRTB usage with explicit outlives constraints works correctly.
// This should continue to compile after the fix.
//
//@ check-pass

trait Subtrait<'a, 'b>: Supertrait<'a, 'b>
where
    'a: 'b,
{
}

trait Supertrait<'a, 'b>
where
    'a: 'b,
{
    fn convert<T: ?Sized>(x: &'a T) -> &'b T;
}

struct MyStruct;

impl<'a: 'b, 'b> Supertrait<'a, 'b> for MyStruct {
    fn convert<T: ?Sized>(x: &'a T) -> &'b T {
        x
    }
}

impl<'a: 'b, 'b> Subtrait<'a, 'b> for MyStruct {}

// This is valid because we explicitly require 'a: 'b
fn valid_conversion<'a: 'b, 'b, T: ?Sized>(x: &'a T) -> &'b T
where
    MyStruct: Subtrait<'a, 'b>,
{
    MyStruct::convert(x)
}

fn main() {
    let x = String::from("Hello World");
    let y = valid_conversion::<'_, '_, _>(&x);
    println!("{}", y);
}

// This test demonstrates the unsoundness in issue #84591
// where HRTB on subtraits can imply HRTB on supertraits without
// preserving necessary outlives constraints, allowing unsafe lifetime extension.
//
// This test should FAIL to compile once the fix is implemented.

trait Subtrait<'a, 'b, R>: Supertrait<'a, 'b> {}

trait Supertrait<'a, 'b> {
    fn convert<T: ?Sized>(x: &'a T) -> &'b T;
}

fn need_hrtb_subtrait<S, T: ?Sized>(x: &T) -> &T
where
    S: for<'a, 'b> Subtrait<'a, 'b, &'b &'a ()>,
{
    need_hrtb_supertrait::<S, T>(x)
}

fn need_hrtb_supertrait<S, T: ?Sized>(x: &T) -> &T
where
    S: for<'a, 'b> Supertrait<'a, 'b>,
{
    S::convert(x)
}

struct MyStruct;

impl<'a: 'b, 'b> Supertrait<'a, 'b> for MyStruct {
    fn convert<T: ?Sized>(x: &'a T) -> &'b T {
        x
    }
}

impl<'a, 'b> Subtrait<'a, 'b, &'b &'a ()> for MyStruct {}

fn extend_lifetime<'a, 'b, T: ?Sized>(x: &'a T) -> &'b T {
    need_hrtb_subtrait::<MyStruct, T>(x)
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    let d;
    {
        let x = String::from("Hello World");
        d = extend_lifetime(&x);
    }
    println!("{}", d);
}

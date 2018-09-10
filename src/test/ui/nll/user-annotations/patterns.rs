// Test that various patterns also enforce types.

#![feature(nll)]

fn variable_no_initializer() {
    let x = 22;
    let y: &'static u32;
    y = &x; //~ ERROR
}

fn tuple_no_initializer() {
    // FIXME(#47187): We are not propagating ascribed type through tuples.

    let x = 22;
    let (y, z): (&'static u32, &'static u32);
    y = &x;
}

fn ref_with_ascribed_static_type() -> u32 {
    // Check the behavior in some wacky cases.
    let x = 22;
    let y = &x; //~ ERROR
    let ref z: &'static u32 = y; //~ ERROR
    **z
}

fn ref_with_ascribed_any_type() -> u32 {
    let x = 22;
    let y = &x;
    let ref z: &u32 = y;
    **z
}

struct Single<T> { value: T }

fn struct_no_initializer() {
    // FIXME(#47187): We are not propagating ascribed type through patterns.

    let x = 22;
    let Single { value: y }: Single<&'static u32>;
    y = &x;
}

fn variable_with_initializer() {
    let x = 22;
    let y: &'static u32 = &x; //~ ERROR
}

fn underscore_with_initializer() {
    let x = 22;
    let _: &'static u32 = &x; //~ ERROR

    let _: Vec<&'static String> = vec![&String::new()];
    //~^ ERROR borrowed value does not live long enough [E0597]

    let (_, a): (Vec<&'static String>, _) = (vec![&String::new()], 44);
    //~^ ERROR borrowed value does not live long enough [E0597]

    let (_a, b): (Vec<&'static String>, _) = (vec![&String::new()], 44);
    //~^ ERROR borrowed value does not live long enough [E0597]
}

fn pair_underscores_with_initializer() {
    let x = 22;
    let (_, _): (&'static u32, u32) = (&x, 44); //~ ERROR
}

fn pair_variable_with_initializer() {
    let x = 22;
    let (y, _): (&'static u32, u32) = (&x, 44); //~ ERROR
}

fn struct_single_field_variable_with_initializer() {
    let x = 22;
    let Single { value: y }: Single<&'static u32> = Single { value: &x }; //~ ERROR
}

fn struct_single_field_underscore_with_initializer() {
    let x = 22;
    let Single { value: _ }: Single<&'static u32> = Single { value: &x }; //~ ERROR
}

struct Double<T> { value1: T, value2: T }

fn struct_double_field_underscore_with_initializer() {
    let x = 22;
    let Double { value1: _, value2: _ }: Double<&'static u32> = Double {
        value1: &x, //~ ERROR
        value2: &44,
    };
}

fn static_to_a_to_static_through_variable<'a>(x: &'a u32) -> &'static u32 {
    // The error in this test is inconsistency with
    // `static_to_a_to_static_through_tuple`, but "feels right" to
    // me. It occurs because we special case the single binding case
    // and force the type of `y` to be `&'a u32`, even though the
    // right-hand side has type `&'static u32`.

    let y: &'a u32 = &22;
    y //~ ERROR
}

fn static_to_a_to_static_through_tuple<'a>(x: &'a u32) -> &'static u32 {
    // FIXME(#47187): The fact that this type-checks is perhaps surprising.
    // What happens is that the right-hand side is constrained to have
    // type `&'a u32`, which is possible, because it has type
    // `&'static u32`. The variable `y` is then forced to have type
    // `&'static u32`, but it is constrained only by the right-hand
    // side, not the ascribed type, and hence it passes.

    let (y, _z): (&'a u32, u32) = (&22, 44);
    y
}

fn a_to_static_then_static<'a>(x: &'a u32) -> &'static u32 {
    let (y, _z): (&'static u32, u32) = (x, 44); //~ ERROR
    y
}

fn main() { }

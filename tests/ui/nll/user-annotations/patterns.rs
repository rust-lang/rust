// Test that various patterns also enforce types.

fn variable_no_initializer() {
    let x = 22;
    let y: &'static u32;
    y = &x; //~ ERROR
}

fn tuple_no_initializer() {


    let x = 22;
    let (y, z): (&'static u32, &'static u32);
    y = &x; //~ ERROR
}

fn ref_with_ascribed_static_type() -> u32 {
    // Check the behavior in some wacky cases.
    let x = 22;
    let y = &x; //~ ERROR
    let ref z: &'static u32 = y;
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


    let x = 22;
    let Single { value: y }: Single<&'static u32>;
    y = &x; //~ ERROR
}


fn struct_no_initializer_must_normalize() {
    trait Indirect { type Assoc; }
    struct StaticU32;
    impl Indirect for StaticU32 { type Assoc = &'static u32; }
    struct Single2<T: Indirect> { value: <T as Indirect>::Assoc }

    let x = 22;
    let Single2 { value: mut _y }: Single2<StaticU32>;
    _y = &x; //~ ERROR
}

fn variable_with_initializer() {
    let x = 22;
    let y: &'static u32 = &x; //~ ERROR
}

fn underscore_with_initializer() {
    let x = 22;
    let _: &'static u32 = &x; //~ ERROR

    let _: Vec<&'static String> = vec![&String::new()];
    //~^ ERROR temporary value dropped while borrowed [E0716]

    let (_, a): (Vec<&'static String>, _) = (vec![&String::new()], 44);
    //~^ ERROR temporary value dropped while borrowed [E0716]

    let (_a, b): (Vec<&'static String>, _) = (vec![&String::new()], 44);
    //~^ ERROR temporary value dropped while borrowed [E0716]
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






    let y: &'a u32 = &22;
    y //~ ERROR
}

fn static_to_a_to_static_through_tuple<'a>(x: &'a u32) -> &'static u32 {







    let (y, _z): (&'a u32, u32) = (&22, 44);
    y //~ ERROR
}

fn static_to_a_to_static_through_struct<'a>(_x: &'a u32) -> &'static u32 {
    let Single { value: y }: Single<&'a u32> = Single { value: &22 };
    y //~ ERROR
}

fn a_to_static_then_static<'a>(x: &'a u32) -> &'static u32 {
    let (y, _z): (&'static u32, u32) = (x, 44); //~ ERROR
    y
}

fn main() { }

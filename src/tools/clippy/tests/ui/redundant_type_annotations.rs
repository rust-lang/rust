#![allow(unused)]
#![warn(clippy::redundant_type_annotations)]

#[derive(Debug, Default)]
struct Cake<T> {
    _data: T,
}

fn make_something<T>() -> T {
    unimplemented!()
}

fn make_cake<T: Default>() -> Cake<T> {
    Cake::<T>::default()
}

fn plus_one<T: std::ops::Add<u8, Output = T>>(val: T) -> T {
    val + 1
}

#[derive(Default)]
struct Slice {
    inner: u32,
}

#[derive(Default)]
struct Pie {
    inner: u32,
    inner_struct: Slice,
}

enum Pizza {
    One,
    Two,
}

fn return_a_string() -> String {
    String::new()
}

fn return_a_struct() -> Pie {
    Pie::default()
}

fn return_an_enum() -> Pizza {
    Pizza::One
}

fn return_an_int() -> u32 {
    5
}

impl Pie {
    fn return_an_int(&self) -> u32 {
        self.inner
    }

    fn return_a_ref(&self) -> &u32 {
        &self.inner
    }

    fn return_a_ref_to_struct(&self) -> &Slice {
        &self.inner_struct
    }

    fn associated_return_an_int() -> u32 {
        5
    }

    fn new() -> Self {
        Self::default()
    }

    fn associated_return_a_string() -> String {
        String::from("")
    }

    fn test_method_call(&self) {
        // Everything here should be lint

        let v: u32 = self.return_an_int();
        //~^ ERROR: redundant type annotation
        //~| NOTE: `-D clippy::redundant-type-annotations` implied by `-D warnings`
        let v: &u32 = self.return_a_ref();
        //~^ ERROR: redundant type annotation
        let v: &Slice = self.return_a_ref_to_struct();
        //~^ ERROR: redundant type annotation
    }
}

fn test_generics() {
    // The type annotation is needed to determine T
    let _c: Cake<i32> = make_something();

    // The type annotation is needed to determine the topic
    let _c: Cake<u8> = make_cake();

    // This could be lint, but currently doesn't
    let _c: Cake<u8> = make_cake::<u8>();

    // This could be lint, but currently doesn't
    let _c: u8 = make_something::<u8>();

    // This could be lint, but currently doesn't
    let _c: u8 = plus_one(5_u8);

    // Annotation needed otherwise T is i32
    let _c: u8 = plus_one(5);

    // This could be lint, but currently doesn't
    let _return: String = String::from("test");
}

fn test_non_locals() {
    // This shouldn't be lint
    fn _arg(x: u32) -> u32 {
        x
    }

    // This could lint, but probably shouldn't
    let _closure_arg = |x: u32| x;
}

trait Trait {
    type AssocTy;
}

impl Trait for () {
    type AssocTy = String;
}

fn test_complex_types<T>() {
    // Shouldn't be lint, since the literal will be i32 otherwise
    let _u8: u8 = 128;

    // This could be lint, but currently doesn't
    let _tuple_i32: (i32, i32) = (12, 13);

    // Shouldn't be lint, since the tuple will be i32 otherwise
    let _tuple_u32: (u32, u32) = (1, 2);

    // Should be lint, since the type is determined by the init value, but currently doesn't
    let _tuple_u32: (u32, u32) = (3_u32, 4_u32);

    // This could be lint, but currently doesn't
    let _array: [i32; 3] = [5, 6, 7];

    // Shouldn't be lint
    let _array: [u32; 2] = [8, 9];

    let ty_param: T = make_something();

    let assoc_ty: <() as Trait>::AssocTy = String::new();
}

fn test_functions() {
    // Everything here should be lint

    let _return: String = return_a_string();
    //~^ ERROR: redundant type annotation

    let _return: Pie = return_a_struct();
    //~^ ERROR: redundant type annotation

    let _return: Pizza = return_an_enum();
    //~^ ERROR: redundant type annotation

    let _return: u32 = return_an_int();
    //~^ ERROR: redundant type annotation

    let _return: String = String::new();
    //~^ ERROR: redundant type annotation

    let new_pie: Pie = Pie::new();
    //~^ ERROR: redundant type annotation

    let _return: u32 = new_pie.return_an_int();
    //~^ ERROR: redundant type annotation

    let _return: u32 = Pie::associated_return_an_int();
    //~^ ERROR: redundant type annotation

    let _return: String = Pie::associated_return_a_string();
    //~^ ERROR: redundant type annotation
}

fn test_simple_types() {
    // Everything here should be lint

    let _var: u32 = u32::MAX;
    //~^ ERROR: redundant type annotation

    let _var: u32 = 5_u32;
    //~^ ERROR: redundant type annotation

    let _var: &str = "test";
    //~^ ERROR: redundant type annotation

    let _var: &[u8; 4] = b"test";
    //~^ ERROR: redundant type annotation

    let _var: bool = false;
    //~^ ERROR: redundant type annotation
}

fn issue12212() {
    // This should not be linted
    let _var: &[u8] = b"test";
}

fn issue11190() {}

fn main() {}

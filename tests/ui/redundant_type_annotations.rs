#![allow(unused)]
#![warn(clippy::redundant_type_annotations)]

#[derive(Debug, Default)]
struct Cake<T> {
    _data: T,
}

fn make_something<T: Default>() -> T {
    T::default()
}

fn make_cake<T: Default>() -> Cake<T> {
    Cake::<T>::default()
}

fn plus_one<T: std::ops::Add<u8, Output = T>>(val: T) -> T {
    val + 1
}

struct Pie {
    inner: u32,
}

enum Pizza {
    One,
    Two,
}

fn return_a_string() -> String {
    String::new()
}

fn return_a_struct() -> Pie {
    Pie { inner: 5 }
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

    fn associated_return_an_int() -> u32 {
        5
    }

    fn new() -> Self {
        Self { inner: 5 }
    }

    fn associated_return_a_string() -> String {
        String::from("")
    }

    fn test_method_call(&self) {
        let v: u32 = self.return_an_int(); // Should lint
    }
}

fn test_generics() {
    // The type annotation is needed to determine T
    let _c: Cake<i32> = make_something();

    // The type annotation is needed to determine the topic
    let _c: Cake<u8> = make_cake();

    // This should lint (doesn't)
    let _c: Cake<u8> = make_cake::<u8>();

    // This should lint (doesn't)
    let _c: u8 = make_something::<u8>();

    // This should lint (doesn't)
    let _c: u8 = plus_one(5_u8);

    // Annotation needed otherwise T is i32
    let _c: u8 = plus_one(5);
}

fn test_non_locals() {
    // This shouldn't lint
    fn _arg(x: u32) -> u32 {
        x
    }

    // This could lint, but probably shouldn't
    let _closure_arg = |x: u32| x;
}

fn test_complex_types() {
    // Shouldn't lint, since the literal will be i32 otherwise
    let _u8: u8 = 128;

    // Should lint (doesn't)
    let _tuple_i32: (i32, i32) = (12, 13);

    // Shouldn't lint, since the tuple will be i32 otherwise
    let _tuple_u32: (u32, u32) = (1, 2);

    // Should lint, since the type is determined by the init value (doesn't)
    let _tuple_u32: (u32, u32) = (3_u32, 4_u32);

    // Should lint (doesn't)
    let _array: [i32; 3] = [5, 6, 7];

    // Shouldn't lint
    let _array: [u32; 2] = [8, 9];
}

fn test_functions() {
    // Everything here should lint

    let _return: String = return_a_string();

    let _return: Pie = return_a_struct();

    let _return: Pizza = return_an_enum();

    let _return: u32 = return_an_int();

    let _return: String = String::new();

    let new_pie: Pie = Pie::new();

    let _return: u32 = new_pie.return_an_int();

    let _return: String = String::from("test");

    let _return: u32 = Pie::associated_return_an_int();

    let _return: String = Pie::associated_return_a_string();
}

fn test_simple_types() {
    let _var: u32 = u32::MAX;

    // Should lint
    let _var: u32 = 5_u32;

    // Should lint
    let _var: &str = "test";

    // Should lint
    let _var: &[u8] = b"test";

    // Should lint
    let _var: bool = false;
}

fn main() {}

// TODO: test refs

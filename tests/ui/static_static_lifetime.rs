#[derive(Debug)]
struct Foo {}

static VAR_ONE: &'static str = "Test static #1"; // ERROR Consider removing 'static.

static VAR_TWO: &str = "Test static #2"; // This line should not raise a warning.

static VAR_THREE: &[&'static str] = &["one", "two"]; // ERROR Consider removing 'static

static VAR_FOUR: (&str, (&str, &'static str), &'static str) = ("on", ("th", "th"), "on"); // ERROR Consider removing 'static

static VAR_FIVE: &'static [&[&'static str]] = &[&["test"], &["other one"]]; // ERROR Consider removing 'static

static VAR_SIX: &'static u8 = &5;

static VAR_SEVEN: &[&(&str, &'static [&'static str])] = &[&("one", &["other one"])];

static VAR_HEIGHT: &'static Foo = &Foo {};

static VAR_SLICE: &'static [u8] = b"Test static #3"; // ERROR Consider removing 'static.

static VAR_TUPLE: &'static (u8, u8) = &(1, 2); // ERROR Consider removing 'static.

static VAR_ARRAY: &'static [u8; 1] = b"T"; // ERROR Consider removing 'static.

fn main() {
    let false_positive: &'static str = "test";
    println!("{}", VAR_ONE);
    println!("{}", VAR_TWO);
    println!("{:?}", VAR_THREE);
    println!("{:?}", VAR_FOUR);
    println!("{:?}", VAR_FIVE);
    println!("{:?}", VAR_SIX);
    println!("{:?}", VAR_SEVEN);
    println!("{:?}", VAR_HEIGHT);
    println!("{}", false_positive);
}

// trait Bar {
//     static TRAIT_VAR: &'static str;
// }

// impl Foo {
//     static IMPL_VAR: &'static str = "var";
// }

// impl Bar for Foo {
//     static TRAIT_VAR: &'static str = "foo";
// }

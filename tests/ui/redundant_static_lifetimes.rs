#[derive(Debug)]
struct Foo {}

const VAR_ONE: &'static str = "Test constant #1"; // ERROR Consider removing 'static.

const VAR_TWO: &str = "Test constant #2"; // This line should not raise a warning.

const VAR_THREE: &[&'static str] = &["one", "two"]; // ERROR Consider removing 'static

const VAR_FOUR: (&str, (&str, &'static str), &'static str) = ("on", ("th", "th"), "on"); // ERROR Consider removing 'static

const VAR_FIVE: &'static [&[&'static str]] = &[&["test"], &["other one"]]; // ERROR Consider removing 'static

const VAR_SIX: &'static u8 = &5;

const VAR_SEVEN: &[&(&str, &'static [&'static str])] = &[&("one", &["other one"])];

const VAR_HEIGHT: &'static Foo = &Foo {};

const VAR_SLICE: &'static [u8] = b"Test constant #1"; // ERROR Consider removing 'static.

const VAR_TUPLE: &'static (u8, u8) = &(1, 2); // ERROR Consider removing 'static.

const VAR_ARRAY: &'static [u8; 1] = b"T"; // ERROR Consider removing 'static.

static STATIC_VAR_ONE: &'static str = "Test static #1"; // ERROR Consider removing 'static.

static STATIC_VAR_TWO: &str = "Test static #2"; // This line should not raise a warning.

static STATIC_VAR_THREE: &[&'static str] = &["one", "two"]; // ERROR Consider removing 'static

static STATIC_VAR_FOUR: (&str, (&str, &'static str), &'static str) = ("on", ("th", "th"), "on"); // ERROR Consider removing 'static

static STATIC_VAR_FIVE: &'static [&[&'static str]] = &[&["test"], &["other one"]]; // ERROR Consider removing 'static

static STATIC_VAR_SIX: &'static u8 = &5;

static STATIC_VAR_SEVEN: &[&(&str, &'static [&'static str])] = &[&("one", &["other one"])];

static STATIC_VAR_HEIGHT: &'static Foo = &Foo {};

static STATIC_VAR_SLICE: &'static [u8] = b"Test static #3"; // ERROR Consider removing 'static.

static STATIC_VAR_TUPLE: &'static (u8, u8) = &(1, 2); // ERROR Consider removing 'static.

static STATIC_VAR_ARRAY: &'static [u8; 1] = b"T"; // ERROR Consider removing 'static.

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

trait Bar {
    const TRAIT_VAR: &'static str;
}

impl Foo {
    const IMPL_VAR: &'static str = "var";
}

impl Bar for Foo {
    const TRAIT_VAR: &'static str = "foo";
}

#![allow(const_err)] // make sure we cannot allow away the errors tested here

#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum {
    A = 0,
}
union TransmuteEnum {
    in1: &'static u8,
    out1: Enum,
}

// A pointer is guaranteed non-null
const BAD_ENUM: Enum = unsafe { TransmuteEnum { in1: &1 }.out1 };
//~^ ERROR is undefined behavior

// (Potentially) invalid enum discriminant
#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum2 {
    A = 2,
}
#[repr(transparent)]
#[derive(Copy, Clone)]
struct Wrap<T>(T);
union TransmuteEnum2 {
    in1: usize,
    in2: &'static u8,
    in3: (),
    out1: Enum2,
    out2: Wrap<Enum2>, // something wrapping the enum so that we test layout first, not enum
    out3: Option<Enum2>,
}
const BAD_ENUM2: Enum2 = unsafe { TransmuteEnum2 { in1: 0 }.out1 };
//~^ ERROR is undefined behavior
const BAD_ENUM3: Enum2 = unsafe { TransmuteEnum2 { in2: &0 }.out1 };
//~^ ERROR is undefined behavior
const BAD_ENUM4: Wrap<Enum2> = unsafe { TransmuteEnum2 { in2: &0 }.out2 };
//~^ ERROR is undefined behavior

// Undef enum discriminant.
const BAD_ENUM_UNDEF : Enum2 = unsafe { TransmuteEnum2 { in3: () }.out1 };
//~^ ERROR is undefined behavior

// Pointer value in an enum with a niche that is not just 0.
const BAD_ENUM_PTR: Option<Enum2> = unsafe { TransmuteEnum2 { in2: &0 }.out3 };
//~^ ERROR is undefined behavior

// Invalid enum field content (mostly to test printing of paths for enum tuple
// variants and tuples).
union TransmuteChar {
    a: u32,
    b: char,
}
// Need to create something which does not clash with enum layout optimizations.
const BAD_ENUM_CHAR: Option<(char, char)> = Some(('x', unsafe { TransmuteChar { a: !0 }.b }));
//~^ ERROR is undefined behavior

fn main() {
}

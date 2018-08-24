#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum {
    A = 0,
}
union TransmuteEnum {
    a: &'static u8,
    b: Enum,
}

// A pointer is guaranteed non-null
const BAD_ENUM: Enum = unsafe { TransmuteEnum { a: &1 }.b };
//~^ ERROR this constant likely exhibits undefined behavior

// Invalid enum discriminant
#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum2 {
    A = 2,
}
union TransmuteEnum2 {
    a: usize,
    b: Enum2,
}
const BAD_ENUM2 : Enum2 = unsafe { TransmuteEnum2 { a: 0 }.b };
//~^ ERROR this constant likely exhibits undefined behavior

// Invalid enum field content (mostly to test printing of apths for enum tuple
// variants and tuples).
union TransmuteChar {
    a: u32,
    b: char,
}
// Need to create something which does not clash with enum layout optimizations.
const BAD_ENUM_CHAR : Option<(char, char)> = Some(('x', unsafe { TransmuteChar { a: !0 }.b }));
//~^ ERROR this constant likely exhibits undefined behavior

fn main() {
}

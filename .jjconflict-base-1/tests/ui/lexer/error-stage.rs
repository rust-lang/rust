// This test is about the treatment of invalid literals. In particular, some
// literals are only considered invalid if they survive to HIR lowering.
//
// Literals with bad suffixes
// --------------------------
// Literals consist of a primary part and an optional suffix.
// https://doc.rust-lang.org/reference/tokens.html#suffixes says:
//
//   Any kind of literal (string, integer, etc) with any suffix is valid as a
//   token, and can be passed to a macro without producing an error. The macro
//   itself will decide how to interpret such a token and whether to produce an
//   error or not.
//
//   ```
//   macro_rules! blackhole { ($tt:tt) => () }
//   blackhole!("string"suffix); // OK
//   ```
//
//   However, suffixes on literal tokens parsed as Rust code are restricted.
//   Any suffixes are rejected on non-numeric literal tokens, and numeric
//   literal tokens are accepted only with suffixes from the list below.
//
//   Integer: u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize
//   Floating-point: f32, f64
//
// This means that something like `"string"any_suffix` is a token accepted by
// the lexer, but rejected later for being an invalid combination of primary
// part and suffix.
//
// `0b10f32` is a similar case. `0b10` is a valid primary part that is a valid
// *integer* literal when no suffix is present. It only causes an error later
// when combined with the `f32` float suffix.
//
// However, `0b10.0f32` is different. It is rejected by the lexer because
// `0b10.0` is not a valid token even on its own.
//
// This difference is unfortunate, but it's baked into the language now.
//
// Too-large integer literals
// --------------------------
// https://doc.rust-lang.org/reference/tokens.html#integer-literals says that
// literals like `128_i8` and `256_u8` "are too big for their type, but are
// still valid tokens".

macro_rules! sink {
    ($($x:tt;)*) => {()}
}

// The invalid literals are ignored because the macro consumes them. Except for
// `0b10.0f32` because it's a lexer error.
const _: () = sink! {
    "string"any_suffix; // OK
    10u123; // OK
    10.0f123; // OK
    0b10f32; // OK
    0b10.0f32; //~ ERROR binary float literal is not supported
    999340282366920938463463374607431768211455999; // OK
};

// The invalid literals used to cause errors, but this was changed by #102944.
// Except for `0b010.0f32`, because it's a lexer error.
#[cfg(false)]
fn configured_out() {
    "string"any_suffix; // OK
    10u123; // OK
    10.0f123; // OK
    0b10f32; // OK
    0b10.0f32; //~ ERROR binary float literal is not supported
    999340282366920938463463374607431768211455999; // OK
}

// All the invalid literals cause errors.
fn main() {
    "string"any_suffix; //~ ERROR suffixes on string literals are invalid
    10u123; //~ ERROR invalid width `123` for integer literal
    10.0f123; //~ ERROR invalid width `123` for float literal
    0b10f32; //~ ERROR binary float literal is not supported
    0b10.0f32; //~ ERROR binary float literal is not supported
    999340282366920938463463374607431768211455999; //~ ERROR integer literal is too large
}

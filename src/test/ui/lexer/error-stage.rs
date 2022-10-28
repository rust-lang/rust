macro_rules! sink {
    ($($x:tt;)*) => {()}
}

// The invalid literals are ignored because the macro consumes them.
const _: () = sink! {
    "string"any_suffix; // OK
    10u123; // OK
    10.0f123; // OK
    0b10f32; // OK
    999340282366920938463463374607431768211455999; // OK
};

// The invalid literals cause errors.
#[cfg(FALSE)]
fn configured_out() {
    "string"any_suffix; //~ ERROR suffixes on string literals are invalid
    10u123; //~ ERROR invalid width `123` for integer literal
    10.0f123; //~ ERROR invalid width `123` for float literal
    0b10f32; //~ ERROR binary float literal is not supported
    999340282366920938463463374607431768211455999; //~ ERROR integer literal is too large
}

// The invalid literals cause errors.
fn main() {
    "string"any_suffix; //~ ERROR suffixes on string literals are invalid
    10u123; //~ ERROR invalid width `123` for integer literal
    10.0f123; //~ ERROR invalid width `123` for float literal
    0b10f32; //~ ERROR binary float literal is not supported
    999340282366920938463463374607431768211455999; //~ ERROR integer literal is too large
}

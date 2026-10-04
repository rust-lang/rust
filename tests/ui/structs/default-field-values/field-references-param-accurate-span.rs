//@ build-fail
#![feature(default_field_values)]

struct Z<const X: usize> {
    // Ensure that proper context is shown in lint.
    multiline_field:
        ()
            = { //~ WARN default value
                f::<X>();
                panic!(); //~ ERROR: explicit panic
            },
}

pub const fn f<const N: usize>() {
    let _ = [0u8; N]; // <-- comment out this line to break downstream!
}

fn use_generically<const X: usize>() {
    let x: Z<X> = Z { .. };
}

fn main() {
    let x: Z<0> = Z { .. };
    use_generically::<0>();
}

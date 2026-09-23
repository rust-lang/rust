//@ build-fail
// https://github.com/rust-lang/rust/issues/146496
#![feature(default_field_values)]

struct Z<const X: usize> {
    // Ensure that proper context is shown in lint.
    multiline_field:
        ()
            = { //~ WARN default value
                f::<X>();
                panic!();
                //~^ ERROR: explicit panic
                //~| ERROR: explicit panic
            },
}

pub const fn f<const N: usize>() {
    // *If* we const evaluated `Z.multiline_field` at definition, but then bailed because `f` needs
    // to be const evaluated, commenting out this line would suddenly allow `f` to be evaluated and
    // cause the panic in `multiline_field` to be reached.
    let _ = [0u8; N];
}

const fn const_use_generically<const X: usize>() {
    let x: Z<X> = Z { .. };
}

fn use_generically<const X: usize>() {
    let x: Z<X> = Z { .. };
}

fn main() {
    let x: Z<1> = Z { .. };
    use_generically::<2>();
    const_use_generically::<3>();
    const { const_use_generically::<4>() };
}

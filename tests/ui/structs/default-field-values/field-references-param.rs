//@ build-pass
//@ revisions: warn allow_struct allow_field

#![feature(default_field_values)]

struct W<const X: usize>;

impl<const X: usize> W<X> {
    const fn new() -> Self { W }
}

#[cfg_attr(allow_struct, allow(unevaluated_default_field_value))]
struct Z<const X: usize> {
    // No inference.
    #[cfg_attr(allow_field, allow(unevaluated_default_field_value))]
    one: W<X> = W::<X>::new(), //[warn]~ WARN

    // Inference works too.
    #[cfg_attr(allow_field, allow(unevaluated_default_field_value))]
    two: W<X> = W::new(), //[warn]~ WARN

    // An anon const that is too generic before substitution.
    too_generic: usize = X + 1,
    //[warn]~^ WARN
    //[allow_field]~^^ WARN

    // Directly using a const parameter.
    #[cfg_attr(allow_field, allow(unevaluated_default_field_value))]
    direct: usize = X,
    //[warn]~^ WARN
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

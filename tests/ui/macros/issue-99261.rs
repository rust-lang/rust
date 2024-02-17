//@ check-pass

#![deny(named_arguments_used_positionally)]

fn main() {
    let value: f64 = 314.15926;
    let digits_before_decimal = 1;
    let digits_after_decimal = 2;
    let width = digits_before_decimal + 1 + digits_after_decimal;

    println!(
        "{value:0>width$.digits_after_decimal$}",
        value = value,
        width = width,
        digits_after_decimal = digits_after_decimal,
    );
}

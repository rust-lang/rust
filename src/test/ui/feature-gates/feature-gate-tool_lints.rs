#[warn(clippy::decimal_literal_representation)]
//~^ ERROR scoped lint `clippy::decimal_literal_representation` is experimental
fn main() {
    let a = 65_535;
}

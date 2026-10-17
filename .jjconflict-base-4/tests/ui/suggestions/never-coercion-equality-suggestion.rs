//! Never-to-any coercions should not justify suggesting equality in place of assignment.

#![allow(unreachable_code, unused_mut)]

fn diverging() -> ! {
    panic!()
}

fn direct(value: String) {
    if value = diverging() {}
    //~^ ERROR mismatched types
}

fn inferred_integer() {
    let mut value = 0;
    if value = diverging() {}
    //~^ ERROR mismatched types
}

fn logical_lhs(flag: bool) {
    if flag && flag = diverging() {}
    //~^ ERROR mismatched types

    if flag || flag = diverging() {}
    //~^ ERROR mismatched types
}

fn logical_rhs(value: String, flag: bool) {
    if value = diverging() && flag {}
    //~^ ERROR mismatched types

    if value = diverging() || flag {}
    //~^ ERROR mismatched types
}

fn ordinary_comparison(value: String) {
    // Keep suggesting equality when the types match without never-to-any coercion.
    if value = String::new() {}
    //~^ ERROR mismatched types
}

fn never_comparison(left: !, right: !) {
    // Comparing two never values does not require a never-to-any coercion either.
    if left = right {}
    //~^ ERROR mismatched types
}

fn main() {}

#![feature(plugin)]
#![plugin(clippy)]

#[allow(needless_bool)]
#[deny(bool_comparison)]
fn main() {
    let x = true;
    if x == true { true } else { false }; //~ERROR you can simplify this boolean comparison to `x`
    if x == false { true } else { false }; //~ERROR you can simplify this boolean comparison to `!x`
    if true == x { true } else { false }; //~ERROR you can simplify this boolean comparison to `x`
    if false == x { true } else { false }; //~ERROR you can simplify this boolean comparison to `!x`
}

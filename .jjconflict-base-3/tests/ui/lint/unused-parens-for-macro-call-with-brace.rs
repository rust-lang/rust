//@ run-rustfix

#![deny(unused_parens)]

fn main() {
    macro_rules! x {
        () => { None::<i32> };
    }

    let Some(_) = (x!{}) else { return }; // no error
    let Some(_) = ((x!{})) else { return };
    //~^ ERROR: unnecessary parentheses around assigned value

    let Some((_)) = (x!{}) else { return };
    //~^ ERROR: unnecessary parentheses around pattern

    let _ = x!{};
    let _ = (x!{});
    //~^ ERROR: unnecessary parentheses around assigned value

    if let Some(_) = x!{} {};
    if let Some(_) = (x!{}) {};
    //~^ ERROR: unnecessary parentheses around `let` scrutinee expression

    while let Some(_) = x!{} {};
    while let Some(_) = (x!{}) {};
    //~^ ERROR: unnecessary parentheses around `let` scrutinee expression
}

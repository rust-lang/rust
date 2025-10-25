//@ edition:2024

fn test_where_left_is_not_let() {
    let mut y = 2;
    if let x = 1 && true && y += 2 {};
    //~^ ERROR expected expression, found `let` statement
    //~| NOTE only supported directly in conditions of `if` and `while` expressions
    //~| ERROR mismatched types
    //~| NOTE expected `bool`, found integer
    //~| NOTE expected because this is `bool`
    //~| ERROR binary assignment operation `+=` cannot be used in a let chain
    //~| NOTE cannot use `+=` in a let chain
    //~| HELP you might have meant to compare with `==` instead of assigning with `+=`
}

fn test_where_left_is_let() {
    let mut y = 2;
    if let x = 1 && y += 2 {};
    //~^ ERROR expected expression, found `let` statement
    //~| NOTE only supported directly in conditions of `if` and `while` expressions
    //~| ERROR mismatched types
    //~| NOTE expected `bool`, found integer
    //~| ERROR binary assignment operation `+=` cannot be used in a let chain
    //~| NOTE cannot use `+=` in a let chain
    //~| HELP you might have meant to compare with `==` instead of assigning with `+=`
}

fn main() {}

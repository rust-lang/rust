fn main() {
    let x = Some(42);
    let y = Some(42);
    let z = Some(42);
    if let Some(_) = x
        && Some(x) = x //~^ ERROR expected expression, found `let` statement
        //~| NOTE: only supported directly in conditions of `if` and `while` expressions
    {}

    if Some(_) = y &&
    //~^ NOTE expected `let` expression, found assignment
    //~| ERROR let-chain with missing `let`
        let Some(_) = z
        //~^ ERROR: expected expression, found `let` statement
        //~| NOTE: let expression later in the condition
        //~| NOTE: only supported directly in conditions of `if` and `while` expressions
    {}
}

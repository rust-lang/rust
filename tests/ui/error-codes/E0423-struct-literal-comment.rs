#[derive(PartialEq)]
struct T { pub x: i32 }

#[derive(PartialEq)]
struct U { }

fn main() {
    // Parser will report an error here
    if T { x: 10 } == T {} {}
    //~^ ERROR struct literals are not allowed here
    //~| ERROR expected value, found struct `T`

    // Regression test for the `followed_by_brace` helper:
    // comments inside the braces should not suppress the parenthesized struct literal suggestion.
    if U { /* keep comment here */ } == U {}
    //~^ ERROR E0423
    //~| ERROR expected expression, found `==`
}

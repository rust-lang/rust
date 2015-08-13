#![feature(plugin)]
#![plugin(clippy)]

fn id<X>(x: X) -> X {
    x
}

#[deny(eq_op)]
#[allow(identity_op)]
fn main() {
    // simple values and comparisons
    1 == 1; //~ERROR equal expressions
    "no" == "no"; //~ERROR equal expressions
    // even though I agree that no means no ;-)
    false != false; //~ERROR equal expressions
    1.5 < 1.5; //~ERROR equal expressions
    1u64 >= 1u64; //~ERROR equal expressions

    // casts, methods, parenthesis
    (1 as u64) & (1 as u64); //~ERROR equal expressions
    1 ^ ((((((1)))))); //~ERROR equal expressions
    id((1)) | id(1); //~ERROR equal expressions

    // unary and binary operators
    (-(2) < -(2));  //~ERROR equal expressions
    ((1 + 1) & (1 + 1) == (1 + 1) & (1 + 1));
          //~^ ERROR equal expressions
                    //~^^ ERROR equal expressions
                               //~^^^ ERROR equal expressions
    (1 * 2) + (3 * 4) == 1 * 2 + 3 * 4; //~ERROR equal expressions

    // various other things
    ([1] != [1]); //~ERROR equal expressions
    ((1, 2) != (1, 2)); //~ERROR equal expressions
    [1].len() == [1].len(); //~ERROR equal expressions
    vec![1, 2, 3] == vec![1, 2, 3]; //no error yet, as we don't match macros
}

#![feature(plugin)]
#![plugin(clippy)]

fn id<X>(x: X) -> X {
    x
}

#[deny(eq_op)]
#[allow(identity_op)]
fn main() {
    // simple values and comparisons
    1 == 1; //~ERROR
    "no" == "no"; //~ERROR
    // even though I agree that no means no ;-)
    false != false; //~ERROR
    1.5 < 1.5; //~ERROR
    1u64 >= 1u64; //~ERROR

    // casts, methods, parenthesis
    (1 as u64) & (1 as u64); //~ERROR
    1 ^ ((((((1)))))); //~ERROR
    id((1)) | id(1); //~ERROR

    // unary and binary operators
    (-(2) < -(2));  //~ERROR
    ((1 + 1) & (1 + 1) == (1 + 1) & (1 + 1));
          //~^ ERROR
                    //~^^ ERROR
                               //~^^^ ERROR
    (1 * 2) + (3 * 4) == 1 * 2 + 3 * 4; //~ERROR

    // various other things
    ([1] != [1]); //~ERROR
    ((1, 2) != (1, 2)); //~ERROR
    [1].len() == [1].len(); //~ERROR
    vec![1, 2, 3] == vec![1, 2, 3]; //no error yet, as we don't match macros
}

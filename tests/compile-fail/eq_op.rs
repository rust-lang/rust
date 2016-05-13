#![feature(plugin)]
#![plugin(clippy)]

#[deny(eq_op)]
#[allow(identity_op)]
#[allow(no_effect, unused_variables, unnecessary_operation)]
#[deny(nonminimal_bool)]
fn main() {
    // simple values and comparisons
    1 == 1; //~ERROR equal expressions
    "no" == "no"; //~ERROR equal expressions
    // even though I agree that no means no ;-)
    false != false; //~ERROR equal expressions
    1.5 < 1.5; //~ERROR equal expressions
    1u64 >= 1u64; //~ERROR equal expressions

    // casts, methods, parentheses
    (1 as u64) & (1 as u64); //~ERROR equal expressions
    1 ^ ((((((1)))))); //~ERROR equal expressions

    // unary and binary operators
    (-(2) < -(2));  //~ERROR equal expressions
    ((1 + 1) & (1 + 1) == (1 + 1) & (1 + 1));
          //~^ ERROR equal expressions as operands to `==`
                    //~^^ ERROR equal expressions as operands to `&`
                               //~^^^ ERROR equal expressions as operands to `&`
    (1 * 2) + (3 * 4) == 1 * 2 + 3 * 4; //~ERROR equal expressions

    // various other things
    ([1] != [1]); //~ERROR equal expressions
    ((1, 2) != (1, 2)); //~ERROR equal expressions
    vec![1, 2, 3] == vec![1, 2, 3]; //no error yet, as we don't match macros

    // const folding
    1 + 1 == 2; //~ERROR equal expressions
    1 - 1 == 0; //~ERROR equal expressions as operands to `==`
                //~^ ERROR equal expressions as operands to `-`

    1 - 1; //~ERROR equal expressions
    1 / 1; //~ERROR equal expressions
    true && true; //~ERROR equal expressions
    //~|ERROR this boolean expression can be simplified
    true || true; //~ERROR equal expressions
    //~|ERROR this boolean expression can be simplified

    let a: u32 = unimplemented!();
    let b: u32 = unimplemented!();

    a == b && b == a; //~ERROR equal expressions
    //~|ERROR this boolean expression can be simplified
    a != b && b != a; //~ERROR equal expressions
    //~|ERROR this boolean expression can be simplified
    a < b && b > a; //~ERROR equal expressions
    //~|ERROR this boolean expression can be simplified
    a <= b && b >= a; //~ERROR equal expressions
    //~|ERROR this boolean expression can be simplified

    let mut a = vec![1];
    a == a; //~ERROR equal expressions
    2*a.len() == 2*a.len(); // ok, functions
    a.pop() == a.pop(); // ok, functions
}

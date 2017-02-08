#![feature(plugin)]
#![plugin(clippy)]

#[deny(eq_op)]
#[allow(identity_op, double_parens)]
#[allow(no_effect, unused_variables, unnecessary_operation, short_circuit_statement)]
#[deny(nonminimal_bool)]
fn main() {
    // simple values and comparisons
    1 == 1;
    "no" == "no";
    // even though I agree that no means no ;-)
    false != false;
    1.5 < 1.5;
    1u64 >= 1u64;

    // casts, methods, parentheses
    (1 as u64) & (1 as u64);
    1 ^ ((((((1))))));

    // unary and binary operators
    (-(2) < -(2));
    ((1 + 1) & (1 + 1) == (1 + 1) & (1 + 1));



    (1 * 2) + (3 * 4) == 1 * 2 + 3 * 4;

    // various other things
    ([1] != [1]);
    ((1, 2) != (1, 2));
    vec![1, 2, 3] == vec![1, 2, 3]; //no error yet, as we don't match macros

    // const folding
    1 + 1 == 2;
    1 - 1 == 0;


    1 - 1;
    1 / 1;
    true && true;

    true || true;


    let a: u32 = 0;
    let b: u32 = 0;

    a == b && b == a;

    a != b && b != a;

    a < b && b > a;

    a <= b && b >= a;


    let mut a = vec![1];
    a == a;
    2*a.len() == 2*a.len(); // ok, functions
    a.pop() == a.pop(); // ok, functions
}

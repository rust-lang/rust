// does not test any rustfixable lints

#[rustfmt::skip]
#[warn(clippy::eq_op)]
#[allow(clippy::identity_op, clippy::double_parens, clippy::many_single_char_names)]
#[allow(clippy::no_effect, unused_variables, clippy::unnecessary_operation, clippy::short_circuit_statement)]
#[allow(clippy::nonminimal_bool)]
#[allow(unused)]
#[allow(clippy::unnecessary_cast)]
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

    check_ignore_macro();

    // named constants
    const A: u32 = 10;
    const B: u32 = 10;
    const C: u32 = A / B; // ok, different named constants
    const D: u32 = A / A;
}

#[rustfmt::skip]
macro_rules! check_if_named_foo {
    ($expression:expr) => (
        if stringify!($expression) == "foo" {
            println!("foo!");
        } else {
            println!("not foo.");
        }
    )
}

macro_rules! bool_macro {
    ($expression:expr) => {
        true
    };
}

#[allow(clippy::short_circuit_statement)]
fn check_ignore_macro() {
    check_if_named_foo!(foo);
    // checks if the lint ignores macros with `!` operator
    !bool_macro!(1) && !bool_macro!("");
}

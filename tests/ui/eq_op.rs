// does not test any rustfixable lints

#[rustfmt::skip]
#[warn(clippy::eq_op)]
#[allow(clippy::identity_op, clippy::double_parens, clippy::many_single_char_names)]
#[allow(clippy::no_effect, unused_variables, clippy::unnecessary_operation, clippy::short_circuit_statement)]
#[allow(clippy::nonminimal_bool)]
#[allow(unused)]
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

    check_assert_identical_args();
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

macro_rules! assert_in_macro_def {
    () => {
        let a = 42;
        assert_eq!(a, a);
        assert_ne!(a, a);
        debug_assert_eq!(a, a);
        debug_assert_ne!(a, a);
    };
}

// lint identical args in assert-like macro invocations (see #3574)
fn check_assert_identical_args() {
    // lint also in macro definition
    assert_in_macro_def!();

    let a = 1;
    let b = 2;

    // lint identical args in `assert_eq!`
    assert_eq!(a, a);
    assert_eq!(a + 1, a + 1);
    // ok
    assert_eq!(a, b);
    assert_eq!(a, a + 1);
    assert_eq!(a + 1, b + 1);

    // lint identical args in `assert_ne!`
    assert_ne!(a, a);
    assert_ne!(a + 1, a + 1);
    // ok
    assert_ne!(a, b);
    assert_ne!(a, a + 1);
    assert_ne!(a + 1, b + 1);

    // lint identical args in `debug_assert_eq!`
    debug_assert_eq!(a, a);
    debug_assert_eq!(a + 1, a + 1);
    // ok
    debug_assert_eq!(a, b);
    debug_assert_eq!(a, a + 1);
    debug_assert_eq!(a + 1, b + 1);

    // lint identical args in `debug_assert_ne!`
    debug_assert_ne!(a, a);
    debug_assert_ne!(a + 1, a + 1);
    // ok
    debug_assert_ne!(a, b);
    debug_assert_ne!(a, a + 1);
    debug_assert_ne!(a + 1, b + 1);

    let my_vec = vec![1; 5];
    let mut my_iter = my_vec.iter();
    assert_ne!(my_iter.next(), my_iter.next());
}

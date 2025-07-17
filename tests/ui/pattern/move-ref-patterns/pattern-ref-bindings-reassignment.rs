//! Tests how we behave when the user attempts to mutate an immutable
//! binding that was introduced by either `ref` or `ref mut`
//! patterns.
//!
//! Such bindings cannot be made mutable via the mere addition of the
//! `mut` keyword, and thus we want to check that the compiler does not
//! suggest doing so.

fn main() {
    let (mut one_two, mut three_four) = ((1, 2), (3, 4));

    // Bind via pattern:
    // - `a` as immutable reference (`ref`)
    // - `b` as mutable reference (`ref mut`)
    let &mut (ref a, ref mut b) = &mut one_two;

    // Attempt to reassign immutable `ref`-bound variable
    a = &three_four.0;
    //~^ ERROR cannot assign twice to immutable variable `a`

    // Attempt to reassign mutable `ref mut`-bound variable
    b = &mut three_four.1;
    //~^ ERROR cannot assign twice to immutable variable `b`
}

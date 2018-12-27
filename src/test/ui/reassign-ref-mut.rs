// Tests how we behave when the user attempts to mutate an immutable
// binding that was introduced by either `ref` or `ref mut`
// patterns.
//
// Such bindings cannot be made mutable via the mere addition of the
// `mut` keyword, and thus we want to check that the compiler does not
// suggest doing so.

fn main() {
    let (mut one_two, mut three_four) = ((1, 2), (3, 4));
    let &mut (ref a, ref mut b) = &mut one_two;
    a = &three_four.0;
    //~^ ERROR cannot assign twice to immutable variable `a` [E0384]
    b = &mut three_four.1;
    //~^ ERROR cannot assign twice to immutable variable `b` [E0384]
}

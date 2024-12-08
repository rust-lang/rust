// This test ensures that the "already bound identifier in a product pattern"
// correctly accounts for or-patterns.

enum E<T> { A(T, T), B(T) }

use E::*;

fn main() {
    let (a, a) = (0, 1); // Standard duplication without an or-pattern.
    //~^ ERROR identifier `a` is bound more than once in the same pattern

    let (a, A(a, _) | B(a)) = (0, A(1, 2));
    //~^ ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR identifier `a` is bound more than once in the same pattern

    let (A(a, _) | B(a), a) = (A(0, 1), 2);
    //~^ ERROR identifier `a` is bound more than once in the same pattern

    let (A(a, a) | B(a)) = A(0, 1);
    //~^ ERROR identifier `a` is bound more than once in the same pattern

    let (B(a) | A(a, a)) = A(0, 1);
    //~^ ERROR identifier `a` is bound more than once in the same pattern

    match A(0, 1) {
        B(a) | A(a, a) => {} // Let's ensure `match` has no funny business.
        //~^ ERROR identifier `a` is bound more than once in the same pattern
    }

    let (B(A(a, _) | B(a)) | A(a, A(a, _) | B(a))) = B(B(1));
    //~^ ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR mismatched types

    let (B(_) | A(A(a, _) | B(a), A(a, _) | B(a))) = B(B(1));
    //~^ ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR variable `a` is not bound in all patterns

    let (B(A(a, _) | B(a)) | A(A(a, _) | B(a), A(a, _) | B(a))) = B(B(1));
    //~^ ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR identifier `a` is bound more than once in the same pattern
}

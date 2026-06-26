// This test ensures that or patterns do not allow missing bindings in any of the arms.

//@ edition:2018

#![allow(non_camel_case_types)]

fn main() {}

fn check_handling_of_paths() {
    mod bar {
        pub enum foo {
            alpha,
            beta,
            charlie
        }
    }

    use bar::foo::{alpha, charlie};
    let (alpha | beta | charlie) = alpha; //~  ERROR variable `beta` is not bound in all patterns
    //~^ ERROR: `beta` is named the same as one of the variants
    match Some(alpha) { //~ ERROR `None` not covered
        Some(alpha | beta) => {} //~ ERROR variable `beta` is not bound in all patterns
        //~^ ERROR: `beta` is named the same as one of the variants
    }
}

fn check_misc_nesting() {
    enum E<T> { A(T, T), B(T) }
    use E::*;
    enum Vars3<S, T, U> { V1(S), V2(T), V3(U) }
    use Vars3::*;

    // One level:
    const X: E<u8> = B(0);
    let (A(a, _) | _) = X; //~ ERROR variable `a` is not bound in all patterns
    let (_ | B(a)) = X; //~ ERROR variable `a` is not bound in all patterns
    let (A(..) | B(a)) = X; //~ ERROR variable `a` is not bound in all patterns
    let (A(a, _) | B(_)) = X; //~ ERROR variable `a` is not bound in all patterns
    let (A(_, a) | B(_)) = X; //~ ERROR variable `a` is not bound in all patterns
    let (A(a, b) | B(a)) = X; //~ ERROR variable `b` is not bound in all patterns

    // Two levels:
    const Y: E<E<u8>> = B(B(0));
    let (A(A(..) | B(_), _) | B(a)) = Y; //~ ERROR variable `a` is not bound in all patterns
    let (A(A(..) | B(a), _) | B(A(a, _) | B(a))) = Y;
    //~^ ERROR variable `a` is not bound in all patterns
    let (A(A(a, b) | B(c), d) | B(e)) = Y;
    //~^ ERROR variable `a` is not bound in all patterns
    //~| ERROR variable `a` is not bound in all patterns
    //~| ERROR variable `b` is not bound in all patterns
    //~| ERROR variable `b` is not bound in all patterns
    //~| ERROR variable `c` is not bound in all patterns
    //~| ERROR variable `c` is not bound in all patterns
    //~| ERROR variable `d` is not bound in all patterns
    //~| ERROR variable `e` is not bound in all patterns

    // Three levels:
    let (
            V1(
            //~^ ERROR variable `b` is not bound in all patterns
            //~| ERROR variable `c` is not bound in all patterns
                A(
                    Ok(a) | Err(_), //~ ERROR variable `a` is not bound in all patterns
                    _
                ) |
                B(Ok(a) | Err(a))
            ) |
            V2(
                A(
                    A(_, a) | //~ ERROR variable `b` is not bound in all patterns
                    B(b), //~ ERROR variable `a` is not bound in all patterns
                    _
                ) |
                B(_)
                //~^ ERROR variable `a` is not bound in all patterns
                //~| ERROR variable `b` is not bound in all patterns
            ) |
            V3(c),
            //~^ ERROR variable `a` is not bound in all patterns
        )
        : (Vars3<E<Result<u8, u8>>, E<E<u8>>, u8>,)
        = (V3(0),);
}

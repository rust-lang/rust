/* Checks whether issue #84592 has been resolved. The issue was
 * that in this example, there are two expected/missing lifetime
 * parameters with *different spans*, leading to incorrect
 * suggestions from rustc.
 */

struct TwoLifetimes<'x, 'y> {
    x: &'x (),
    y: &'y (),
}

fn two_lifetimes_needed(a: &(), b: &()) -> TwoLifetimes<'_, '_> {
//~^ ERROR missing lifetime specifiers [E0106]
    TwoLifetimes { x: &(), y: &() }
}

fn main() {}

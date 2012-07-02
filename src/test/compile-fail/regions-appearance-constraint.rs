/*

Tests that borrowing always produces a pointer confined to the
innermost scope.  In this case, the variable `a` gets inferred
to the lifetime of the `if` statement because it is assigned
a borrow of `y` which takes place within the `if`.

Note: If this constraint were lifted (as I contemplated at one point),
it complicates the preservation mechanics in trans, though not
irreperably.  I'm partially including this test so that if these
semantics do change we'll remember to test this scenario.

*/

fn testfn(cond: bool) {
    let mut x = @3;
    let mut y = @4;

    let mut a = &*x;
    //~^ ERROR reference is not valid outside of its lifetime

    let mut exp = 3;
    if cond {
        a = &*y;

        exp = 4;
    }

    x = @5;
    y = @6;
    assert *a == exp;
}

fn main() {
}
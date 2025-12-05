//@ known-bug: #108499

// at lower recursion limits the recursion limit is reached before the bug happens
#![recursion_limit = "2000"]

// this will try to calculate 3↑↑3=3^(3^3)
type Test = <() as Op<((), ()), [[[(); 0]; 0]; 0], [[[(); 0]; 0]; 0],
    [[[[(); 0]; 0]; 0]; 0]>>::Result;

use std::default::Default;

fn main() {
    // force the compiler to actually evaluate `Test`
    println!("{}", Test::default());
}

trait Op<X, A, B, C> {
    type Result;
}

// this recursive function defines the hyperoperation sequence,
// a canonical example of the type of recursion which produces the issue
// the problem seems to be caused by having two recursive calls, the second
// of which depending on the first
impl<
    X: Op<(X, Y), A, [B; 0], [C; 0]>,
    Y: Op<(X, Y), A, X::Result, C>,
    A, B, C,
> Op<(X, Y), A, [[B; 0]; 0], [C; 0]> for () {
    type Result = Y::Result;
}

// base cases
impl<X, A, B> Op<X, A, B, ()> for () {
    type Result = [B; 0];
}

impl<X, A> Op<X, A, [(); 0], [(); 0]> for () {
    type Result = [A; 0];
}

impl<X, A, C> Op<X, A, [(); 0], [[C; 0]; 0]> for () {
    type Result = A;
}

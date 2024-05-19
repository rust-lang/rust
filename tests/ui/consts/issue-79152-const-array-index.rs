//@ check-pass
// Regression test for issue #79152
//
// Tests that we can index an array in a const function

const fn foo() {
    let mut array = [[0; 1]; 1];
    array[0][0] = 1;
}

fn main() {}

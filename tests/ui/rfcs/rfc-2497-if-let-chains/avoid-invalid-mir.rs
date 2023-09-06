// Regression test for #104172

const N: usize = {
    struct U;
    !let y = 42;
    //~^ ERROR expected expression, found `let` statement
    //~| ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions in this position are unstable [E0658]
    3
};

struct S {
    x: [(); N]
}

fn main() {}
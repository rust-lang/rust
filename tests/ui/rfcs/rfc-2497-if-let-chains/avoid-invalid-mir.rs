// Regression test for #104172

const N: usize = {
    struct U;
    !let y = 42;
    //~^ ERROR expected expression, found `let` statement
    3
};

struct S {
    x: [(); N]
}

fn main() {}

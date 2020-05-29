enum Bug<S> {
    Var = {
        let x: S = 0; //~ ERROR: mismatched types
        0
    },
}

fn main() {}

enum Bug<S> { //~ ERROR parameter `S` is never used
    Var = {
        let x: S = 0; //~ ERROR generic parameters may not be used
        0
    },
}

fn main() {}

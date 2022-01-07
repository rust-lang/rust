enum Bug<S> { //~ ERROR parameter `S` is never used
    Var = {
        //~^ ERROR failed to evaluate the given
        let x: S = 0;
        //~^ ERROR generic parameters may not be used
        0
    },
}

fn main() {}

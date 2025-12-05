enum Bug {
    V1 = return [0][0], //~ERROR return statement outside of function body
}

fn main() {}

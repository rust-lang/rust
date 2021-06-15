struct Bug {
    A: [(); {
        let x;
        x
        //~^ error: use of possibly-uninitialized variable
    }],
}

fn main() {}

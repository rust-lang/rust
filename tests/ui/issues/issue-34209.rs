enum S {
    A,
}

fn bug(l: S) {
    match l {
        S::B {} => {}, //~ ERROR no variant named `B` found for enum `S`
    }
}

fn main () {}

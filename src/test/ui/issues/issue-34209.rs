enum S {
    A,
}

fn bug(l: S) {
    match l {
        S::B {} => {}, //~ ERROR no variant `B` in enum `S`
    }
}

fn main () {}

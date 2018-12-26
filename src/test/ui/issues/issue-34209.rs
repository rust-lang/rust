enum S {
    A,
}

fn bug(l: S) {
    match l {
        S::B { } => { },
        //~^ ERROR no variant `B` on enum `S`
    }
}

fn main () {}

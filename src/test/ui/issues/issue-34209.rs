enum S {
    A,
}

fn bug(l: S) {
    match l {
        S::B{ } => { },
        //~^ ERROR ambiguous associated type
    }
}

fn main () {}

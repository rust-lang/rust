//@ compile-flags: -Znext-solver
//@ check-pass

pub fn repro() -> impl FnMut() {
    if true { || () } else { || () }
}

fn main() {}

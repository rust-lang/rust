// run-pass
#![allow(path_statements)]
#![allow(dead_code)]

macro_rules! union {
    () => (struct S;)
}

union!();

fn union() {}

fn main() {
    union();

    let union = 10;

    union;

    union as u8;

    union U {
        a: u8,
    }
}

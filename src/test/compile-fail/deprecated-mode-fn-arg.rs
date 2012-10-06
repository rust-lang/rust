#[forbid(deprecated_mode)];

fn foo(_f: fn(&i: int)) { //~ ERROR by-mutable-reference mode
}

type Bar = fn(&i: int); //~ ERROR by-mutable-reference mode

fn main() {
}
#[forbid(deprecated_mode)];

fn foo(_f: fn(&i: int)) { //~ ERROR explicit mode
}

type Bar = fn(&i: int); //~ ERROR explicit mode

fn main() {
}
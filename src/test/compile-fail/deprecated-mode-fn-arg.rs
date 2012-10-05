#[forbid(deprecated_mode)];

fn foo(_f: fn(&i: int)) { //~ ERROR explicit mode
    //~^ WARNING Obsolete syntax has no effect
}

type Bar = fn(&i: int); //~ ERROR explicit mode
    //~^ WARNING Obsolete syntax has no effect

fn main() {
}
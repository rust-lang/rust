// Issue #33262

pub fn main() {
    for i in 0..a as { }
    //~^ ERROR expected type, found `{`
}

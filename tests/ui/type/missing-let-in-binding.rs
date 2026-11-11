//@ run-rustfix
fn main() {
    let mut _foo: i32 = 1;
    _foo: i32 = 4; //~ ERROR expected identifier, found `:`
}

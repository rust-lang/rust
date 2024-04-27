//@ run-rustfix
fn main() {
    let _x = 42; //~ HELP
    let _y = x; //~ ERROR
}

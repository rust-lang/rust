fn main() {
    let x: fn~() = || (); //~ ERROR expected `(`, found `~`
}

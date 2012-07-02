fn main() {
    let mut x = 3;
    let y;
    x <-> y; //~ ERROR use of possibly uninitialized variable: `y`
    copy x;
}

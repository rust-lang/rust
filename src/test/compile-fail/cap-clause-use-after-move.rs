fn main() {
    let x = 5;
    let _y = fn~(move x) { };
    let _z = x; //~ ERROR use of moved variable: `x`
}

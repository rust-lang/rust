fn main() {
    let x = 5;
    let _y = fn~(move x) { }; //~ WARNING captured variable `x` not used in closure
    let _z = x; //~ ERROR use of moved variable: `x`
}

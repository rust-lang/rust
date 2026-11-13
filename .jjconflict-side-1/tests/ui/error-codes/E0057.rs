fn main() {
    let f = |x| x * 3;
    let a = f(); //~ ERROR E0057
    let b = f(4);
    let c = f(2, 3); //~ ERROR E0057
}

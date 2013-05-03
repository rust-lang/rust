fn main() {
    let mut b = ~3;
    let _x = &mut *b;   //~ NOTE loan of mutable local variable granted here
    let mut y = /*move*/ b; //~ ERROR moving out of mutable local variable prohibited
    *y += 1;
}

fn main() {
    let mut b = ~3;
    let _x = &mut *b;
    let mut y = /*move*/ b; //~ ERROR cannot move out
    *y += 1;
}

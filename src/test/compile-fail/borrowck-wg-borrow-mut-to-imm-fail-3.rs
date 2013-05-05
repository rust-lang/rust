fn main() {
    let mut a = ~3;
    let mut b = &mut a;
    let _c = &mut *b;
    let mut d = /*move*/ a; //~ ERROR cannot move out
    *d += 1;
}

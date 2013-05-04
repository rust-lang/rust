fn main() {
    let mut a = ~3;
    let mut b = &mut a; //~ NOTE loan of mutable local variable granted here
    let _c = &mut *b;
    let mut d = /*move*/ a; //~ ERROR moving out of mutable local variable prohibited due to outstanding loan
    *d += 1;
}

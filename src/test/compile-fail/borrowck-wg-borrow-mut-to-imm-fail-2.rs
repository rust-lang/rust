fn main() {
    let mut b = ~3;
    let _x = &mut *b;   //~ NOTE prior loan as mutable granted here
    let _y = &mut *b;   //~ ERROR loan of dereference of mutable ~ pointer as mutable conflicts with prior loan
}

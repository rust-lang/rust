fn main() {
    let mut b = ~3;
    let _x = &mut *b;
    let _y = &mut *b; //~ ERROR cannot borrow
}

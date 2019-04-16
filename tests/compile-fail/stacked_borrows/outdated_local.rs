fn main() {
    let mut x = 0;
    let y: *const i32 = &x;
    x = 1; // this invalidates y by reactivating the lowermost uniq borrow for this local

    assert_eq!(unsafe { *y }, 1); //~ ERROR borrow stack

    assert_eq!(x, 1);
}

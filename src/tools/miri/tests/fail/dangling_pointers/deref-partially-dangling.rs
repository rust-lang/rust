// Deref a raw ptr to access a field of a large struct, where the field
// is allocated but not the entire struct is.
fn main() {
    let x = (1, 13);
    let xptr = &x as *const _ as *const (i32, i32, i32);
    let val = unsafe { (*xptr).1 }; //~ ERROR: pointer to 12 bytes starting at offset 0 is out-of-bounds
    assert_eq!(val, 13);
}

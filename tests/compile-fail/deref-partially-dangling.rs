// Deref a raw ptr to access a field of a large struct, where the field
// is allocated but not the entire struct is.
fn main() {
    let x = (1, 13);
    let xptr = &x as *const _ as *const (i32, i32, i32);
    let val = unsafe { (*xptr).1 }; //~ ERROR pointer must be in-bounds at offset 12, but is outside bounds of allocation
    assert_eq!(val, 13);
}

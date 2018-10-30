// Deref a raw ptr to access a field of a large struct, where the field
// is allocated but not the entire struct is.
// For now, we want to allow this.

fn main() {
    let x = (1, 1);
    let xptr = &x as *const _ as *const (i32, i32, i32);
    let _val = unsafe { (*xptr).1 };
}

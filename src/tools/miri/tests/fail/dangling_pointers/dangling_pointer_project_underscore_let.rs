// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

fn main() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32 as *const (u8, u8, u8, u8)
    };
    unsafe {
        let _ = (*p).1; //~ ERROR: in-bounds pointer arithmetic failed
    }
}

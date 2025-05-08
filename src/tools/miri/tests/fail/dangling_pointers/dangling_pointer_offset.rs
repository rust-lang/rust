// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

fn main() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32
    };
    let x = unsafe { p.offset(42) }; //~ ERROR: /in-bounds pointer arithmetic failed: .* has been freed/
    panic!("this should never print: {:?}", x);
}

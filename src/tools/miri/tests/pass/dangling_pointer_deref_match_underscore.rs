// A `_` binding in a match is a nop, so we do not detect that the pointer is dangling.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

fn main() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32
    };
    unsafe {
        match *p {
            _ => {}
        }
    }
}

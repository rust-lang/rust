// Test for the behavior described in <https://github.com/rust-lang/rust/issues/87184>.

const PARTIAL_OVERWRITE: () = {
    let mut p = &42;
    unsafe {
        let ptr: *mut _ = &mut p;
        *(ptr as *mut u8) = 123; //~ ERROR constant
        //~| NOTE unable to overwrite parts of a pointer
    }
    let x = *p;
};

fn main() {}

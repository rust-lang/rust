// edition: 2018

// tests that the pointee type of a raw pointer must be known to call methods on it
// see also: `tests/ui/editions/edition-raw-pointer-method-2018.rs`

fn main() {
    let val = 1_u32;
    let ptr = &val as *const u32;
    unsafe {
        let _a: i32 = (ptr as *const _).read();
        //~^ ERROR cannot call a method on a raw pointer with an unknown pointee type [E0699]
        let b = ptr as *const _;
        let _b: u8 = b.read();
        //~^ ERROR cannot call a method on a raw pointer with an unknown pointee type [E0699]
        let _c = (ptr as *const u8).read(); // we know the type here
    }

    let mut val = 2_u32;
    let ptr = &mut val as *mut u32;
    unsafe {
        let _a: i32 = (ptr as *mut _).read();
        //~^ ERROR cannot call a method on a raw pointer with an unknown pointee type [E0699]
        let b = ptr as *mut _;
        b.write(10);
        //~^ ERROR cannot call a method on a raw pointer with an unknown pointee type [E0699]
        (ptr as *mut i32).write(1000); // we know the type here
    }
}

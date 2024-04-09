//@ edition: 2018

// tests that the pointee type of a raw pointer must be known to call methods on it
// see also: `tests/ui/editions/edition-raw-pointer-method-2018.rs`

fn main() {
    let val = 1_u32;
    let ptr = &val as *const u32;
    unsafe {
        let _a: i32 = (ptr as *const _).read();
        //~^ ERROR type annotations needed
        let b = ptr as *const _;
        //~^ ERROR type annotations needed
        let _b: u8 = b.read();
        let _c = (ptr as *const u8).read(); // we know the type here
    }

    let mut val = 2_u32;
    let ptr = &mut val as *mut u32;
    unsafe {
        let _a: i32 = (ptr as *mut _).read();
        //~^ ERROR type annotations needed
        let b = ptr as *mut _;
        //~^ ERROR type annotations needed
        b.write(10);
        (ptr as *mut i32).write(1000); // we know the type here
    }
}

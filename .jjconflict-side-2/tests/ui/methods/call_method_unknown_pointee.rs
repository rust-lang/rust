//@ edition: 2018

// tests that the pointee type of a raw pointer must be known to call methods on it
// see also: `tests/ui/editions/edition-raw-pointer-method-2018.rs`

fn a() {
    let ptr = &1u32 as *const u32;
    unsafe {
        let _a: i32 = (ptr as *const _).read();
        //~^ ERROR type annotations needed
    }
}

fn b() {
    let ptr = &1u32 as *const u32;
    unsafe {
        let b = ptr as *const _;
        //~^ ERROR type annotations needed
        let _b: u8 = b.read();
    }
}


fn c() {
    let ptr = &mut 2u32 as *mut u32;
    unsafe {
        let _c: i32 = (ptr as *mut _).read();
        //~^ ERROR type annotations needed
    }
}

fn d() {
    let ptr = &mut 2u32 as *mut u32;
    unsafe {
        let d = ptr as *mut _;
        //~^ ERROR type annotations needed
        let _d: u8 = d.read();
    }
}

fn main() {}

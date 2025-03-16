//@ check-pass
#![allow(improper_ctypes_definitions)]
#![feature(unsized_fn_params)]
#![crate_type = "lib"]

struct Fat<T: ?Sized>(T);

// Check that computing the fn abi for `bad`, with a external ABI fn ptr that is not FFI-safe, does
// not ICE.

pub fn bad(f: extern "C" fn([u8])) {}

// While these get accepted, they should also not ICE.
// (If we ever reject them, remove them from this test to ensure the `bad` above
// is still tested. Do *not* make this a check/build-fail test.)

pub extern "C" fn declare_bad(_x: str) {}

#[no_mangle]
pub extern "system" fn declare_more_bad(f: dyn FnOnce()) {
}

fn make_bad() -> extern "C" fn(Fat<[u8]>) {
    todo!()
}

pub fn call_bad() {
    let f = make_bad();
    let slice: Box<Fat<[u8]>> = Box::new(Fat([1; 8]));
    f(*slice);
}

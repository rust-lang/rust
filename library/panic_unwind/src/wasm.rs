//! Unwinding panics for wasm32.
use alloc::boxed::Box;
use core::any::Any;

// The type of the exception payload that the wasm engine propagates
// through unwinding for us. LLVM requires that it be a thin pointer.
type Payload = Box<Box<dyn Any + Send>>;

extern "C" {
    /// LLVM lowers this intrinsic to the `throw` instruction.
    #[link_name = "llvm.wasm.throw"]
    fn wasm_throw(tag: i32, ptr: *mut u8) -> !;
}

pub unsafe fn panic(payload: Box<dyn Any + Send>) -> u32 {
    // The payload we pass to `wasm_throw` will be exactly the argument we get
    // in `cleanup` below. So we just box it up once, to get something pointer-sized.
    let payload_box: Payload = Box::new(payload);
    // The wasm `throw` instruction takes a "tag", which differentiates certain
    // types of exceptions from others. LLVM currently just identifies these
    // via integers, with 0 corresponding to C++ exceptions and 1 to C setjmp()/longjmp().
    // Ideally, we'd be able to choose something unique for Rust, such that we
    // don't try to treat a C++ exception payload as a `Box<Box<dyn Any>>`, but
    // otherwise, pretending to be C++ works for now.
    wasm_throw(0, Box::into_raw(payload_box) as *mut u8)
}

pub unsafe fn cleanup(payload_box: *mut u8) -> Box<dyn Any + Send> {
    // Recover the underlying `Box`.
    let payload_box: Payload = Box::from_raw(payload_box as *mut _);
    *payload_box
}

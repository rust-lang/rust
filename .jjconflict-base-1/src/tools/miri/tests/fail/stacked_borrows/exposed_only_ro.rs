//@compile-flags: -Zmiri-permissive-provenance

// If we have only exposed read-only pointers, doing a write through a wildcard ptr should fail.

fn main() {
    let mut x = 0;
    let _fool = &mut x as *mut i32; // this would have fooled the old untagged pointer logic
    let addr = (&x as *const i32).expose_provenance();
    let ptr = std::ptr::with_exposed_provenance_mut::<i32>(addr);
    unsafe { *ptr = 0 }; //~ ERROR: /write access using <wildcard> .* no exposed tags have suitable permission in the borrow stack/
}

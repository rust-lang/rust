//@only-target: x86_64-unknown-linux-gnu i686-unknown-linux-gnu
//@compile-flags: -Zmiri-permissive-provenance -Zmiri-native-lib-enable-tracing

extern "C" {
    fn do_one_deref(ptr: *const *const *const i32) -> usize;
}

fn main() {
    unexposed_reachable_alloc();
}

// Expose 2 pointers by virtue of doing a native read and assert that the 3rd in
// the chain remains properly unexposed.
fn unexposed_reachable_alloc() {
    let inner = 42;
    let intermediate_a = &raw const inner;
    let intermediate_b = &raw const intermediate_a;
    let exposed = &raw const intermediate_b;
    // Discard the return value; it's just there so the access in C doesn't get optimised away.
    unsafe { do_one_deref(exposed) };
    // Native read should have exposed the address of intermediate_b...
    let valid: *const i32 = std::ptr::with_exposed_provenance(intermediate_b.addr());
    // but not of intermediate_a.
    let invalid: *const i32 = std::ptr::with_exposed_provenance(intermediate_a.addr());
    unsafe {
        let _ok = *valid;
        let _not_ok = *invalid; //~ ERROR: Undefined Behavior: memory access failed: attempting to access
    }
}

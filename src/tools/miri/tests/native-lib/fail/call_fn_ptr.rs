//@revisions: trace notrace
//@[trace] only-target: x86_64-unknown-linux-gnu i686-unknown-linux-gnu
//@[trace] compile-flags: -Zmiri-native-lib-enable-tracing
//@compile-flags: -Zmiri-permissive-provenance

fn main() {
    pass_fn_ptr()
}

fn pass_fn_ptr() {
    extern "C" {
        fn call_fn_ptr(s: Option<extern "C" fn()>);
    }

    extern "C" fn nop() {}

    unsafe {
        call_fn_ptr(None); // this one is fine
        call_fn_ptr(Some(nop)); //~ ERROR: unsupported operation: calling a function pointer through the FFI boundary
    }
}

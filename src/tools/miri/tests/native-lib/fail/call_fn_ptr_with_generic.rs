//@revisions: trace notrace
//@[trace] only-target: x86_64-unknown-linux-gnu i686-unknown-linux-gnu
//@[trace] compile-flags: -Zmiri-native-lib-enable-tracing
//@compile-flags: -Zmiri-permissive-provenance

fn main() {
    pass_fn_ptr()
}

fn pass_fn_ptr() {
    extern "C" {
        fn call_fn_ptr(s: extern "C" fn(i32) -> i32);
    }

    extern "C" fn id<T>(x: T) -> T {
        x
    }

    fn pass_via_id_fn_ptr<T>(x: T) -> T {
        let f = id::<T> as extern "C" fn(T) -> T;
        f(x)
    }

    unsafe {
        // As long as we keep the function pointer on the Rust side, this is all fine.
        let closure = || ();
        pass_via_id_fn_ptr(closure)();
        // When we pass it to FFI, we get the expected error.
        call_fn_ptr(id::<i32>); //~ ERROR: unsupported operation: calling a function pointer through the FFI boundary
    }
}

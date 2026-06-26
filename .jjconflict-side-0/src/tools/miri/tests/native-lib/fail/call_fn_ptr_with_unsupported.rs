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

    fn make_fn_ptr_for_type<T>(_x: &T) -> extern "C" fn(T) -> T {
        id::<T> as extern "C" fn(T) -> T
    }

    unsafe {
        let closure = || ();
        let f = make_fn_ptr_for_type(&closure);
        // Transmute the type around so C can invoke this -- with the entirely wrong signature.
        // Wht we're hoping for is that Miri just ignores all the arguments and aborts
        // with a reasonable error instead.
        call_fn_ptr(std::mem::transmute(f));
        //~^ ERROR: unsupported operation: calling a function pointer with unsupported argument/return type
    }
}

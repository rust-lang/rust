//@compile-flags: -Zmiri-disable-abi-check
#![feature(core_intrinsics)]

fn main() {
    fn foo() {}

    extern "C" fn try_fn(ptr: *mut u8) {
        assert!(ptr.is_null());
    }

    extern "Rust" {
        fn malloc(size: usize) -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(0);
        std::mem::transmute::<fn(), extern "C" fn()>(foo)();
        std::intrinsics::r#try(
            std::mem::transmute::<extern "C" fn(*mut u8), _>(try_fn),
            std::ptr::null_mut(),
            |_, _| unreachable!(),
        );
    }
}

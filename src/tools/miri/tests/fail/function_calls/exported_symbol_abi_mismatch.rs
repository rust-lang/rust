//@revisions: no_cache cache fn_ptr

#[no_mangle]
fn foo() {}

fn main() {
    #[cfg(any(cache, fn_ptr))]
    extern "Rust" {
        fn foo();
    }

    #[cfg(fn_ptr)]
    unsafe {
        std::mem::transmute::<unsafe fn(), unsafe extern "C" fn()>(foo)();
        //~[fn_ptr]^ ERROR: calling a function with calling convention "Rust" using calling convention "C"
    }

    // `Instance` caching should not suppress ABI check.
    #[cfg(cache)]
    unsafe {
        foo();
    }

    {
        #[cfg_attr(any(cache, fn_ptr), allow(clashing_extern_declarations))]
        extern "C" {
            fn foo();
        }
        unsafe {
            foo();
            //~[no_cache]^ ERROR: calling a function with calling convention "Rust" using calling convention "C"
            //~[cache]| ERROR: calling a function with calling convention "Rust" using calling convention "C"
        }
    }
}

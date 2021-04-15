// revisions: no_cache cache

#[no_mangle]
fn foo() {}

fn main() {
    #[cfg(cache)]
    {
        // `Instance` caching should not suppress ABI check.
        extern "Rust" {
            fn foo();
        }
        unsafe { foo() }
    }
    #[cfg_attr(cache, allow(clashing_extern_declarations))]
    extern "C" {
        fn foo();
    }
    unsafe { foo() }
    //[no_cache]~^ ERROR Undefined Behavior: calling a function with ABI Rust using caller ABI C
    //[cache]~^^ ERROR Undefined Behavior: calling a function with ABI Rust using caller ABI C
}

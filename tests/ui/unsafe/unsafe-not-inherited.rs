// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck
#![allow(unused, dead_code)]

static mut FOO: u64 = 0;

fn static_mod() {
    unsafe {static BAR: u64 = FOO;}
    //~^ ERROR: use of mutable static is unsafe
    //~| NOTE: use of mutable static
    //~| NOTE: mutable statics can be mutated by multiple threads
    //~| NOTE: items do not inherit unsafety
}

unsafe fn unsafe_call() {}
fn foo() {
    unsafe {
    //~^ NOTE: items do not inherit unsafety
        fn bar() {
            unsafe_call();
            //~^ ERROR: call to unsafe function
            //~| NOTE: call to unsafe function
            //~| NOTE: consult the function's documentation
        }
    }
}

fn main() {}

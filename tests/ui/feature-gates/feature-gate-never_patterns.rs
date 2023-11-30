// Check that never patterns require the feature gate.
use std::ptr::NonNull;

enum Void {}

fn main() {
    let res: Result<u32, Void> = Ok(0);
    let (Ok(_x) | Err(&!)) = res.as_ref();
    //~^ ERROR `!` patterns are experimental
    //~| ERROR: is not bound in all patterns

    unsafe {
        let ptr: *const Void = NonNull::dangling().as_ptr();
        match *ptr {
            ! => {} //~ ERROR `!` patterns are experimental
        }
    }

    // Check that the gate operates even behind `cfg`.
    #[cfg(FALSE)]
    unsafe {
        let ptr: *const Void = NonNull::dangling().as_ptr();
        match *ptr {
            ! => {} //~ ERROR `!` patterns are experimental
        }
    }
}

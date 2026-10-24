// Check that never patterns require the feature gate.
use std::ptr::NonNull;

enum Void {}

fn main() {
    let res: Result<u32, Void> = Ok(0);
    let (Ok(_x) | Err(&!)) = res.as_ref();
    //~^ ERROR `!` patterns are experimental

    unsafe {
        let ptr: *const Void = NonNull::dangling().as_ptr();
        match *ptr {
            !
            //~^ ERROR `!` patterns are experimental
        }
        // Check that the gate operates even behind `cfg`.
        #[cfg(false)]
        match *ptr {
            !
            //~^ ERROR `!` patterns are experimental
        }
        #[cfg(false)]
        match *ptr {
            ! => {}
            //~^ ERROR `!` patterns are experimental
        }
    }

    // Correctly gate match arms with no body.
    match Some(0) {
        None => {}
        Some(_),
        //~^ ERROR unexpected `,` in pattern
    }
    match Some(0) {
        None => {}
        Some(_)
        //~^ ERROR `match` arm with no body
    }
    match Some(0) {
        _ => {}
        Some(_) if false,
        //~^ ERROR `match` arm with no body
        Some(_) if false
        //~^ ERROR `match` arm with no body
    }
    match res {
        Ok(_) => {}
        Err(!),
        //~^ ERROR `!` patterns are experimental
    }
    match res {
        Err(!) if false,
        //~^ ERROR `!` patterns are experimental
        //~| ERROR a guard on a never pattern will never be run
        _ => {}
    }

    // Check that the gate operates even behind `cfg`.
    match Some(0) {
        None => {}
        #[cfg(false)]
        Some(_)
        //~^ ERROR `match` arm with no body
    }
    match Some(0) {
        _ => {}
        #[cfg(false)]
        Some(_) if false
        //~^ ERROR `match` arm with no body
    }
}

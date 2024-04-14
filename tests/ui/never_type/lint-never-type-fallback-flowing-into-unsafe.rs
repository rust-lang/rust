//@ check-pass
use std::mem;

fn main() {
    if false {
        unsafe { mem::zeroed() }
        //~^ warn: never type fallback affects this call to an `unsafe` function
    } else {
        return;
    };

    // no ; -> type is inferred without fallback
    if true { unsafe { mem::zeroed() } } else { return }
}

// Minimization of the famous `objc` crate issue
fn _objc() {
    pub unsafe fn send_message<R>() -> Result<R, ()> {
        Ok(unsafe { core::mem::zeroed() })
    }

    macro_rules! msg_send {
        () => {
            match send_message::<_ /* ?0 */>() {
                //~^ warn: never type fallback affects this call to an `unsafe` function
                Ok(x) => x,
                Err(_) => loop {},
            }
        };
    }

    unsafe {
        msg_send!();
    }
}

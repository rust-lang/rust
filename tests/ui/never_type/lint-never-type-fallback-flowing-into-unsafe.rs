//@ check-pass
use std::{marker, mem, ptr};

fn main() {}

fn _zero() {
    if false {
        unsafe { mem::zeroed() }
        //~^ warn: never type fallback affects this call to an `unsafe` function
        //~| warn: this will change its meaning in a future release!
    } else {
        return;
    };

    // no ; -> type is inferred without fallback
    if true { unsafe { mem::zeroed() } } else { return }
}

fn _trans() {
    if false {
        unsafe {
            struct Zst;
            core::mem::transmute(Zst)
            //~^ warn: never type fallback affects this call to an `unsafe` function
            //~| warn: this will change its meaning in a future release!
        }
    } else {
        return;
    };
}

fn _union() {
    if false {
        union Union<T: Copy> {
            a: (),
            b: T,
        }

        unsafe { Union { a: () }.b }
        //~^ warn: never type fallback affects this union access
        //~| warn: this will change its meaning in a future release!
    } else {
        return;
    };
}

fn _deref() {
    if false {
        unsafe { *ptr::from_ref(&()).cast() }
        //~^ warn: never type fallback affects this raw pointer dereference
        //~| warn: this will change its meaning in a future release!
    } else {
        return;
    };
}

fn _only_generics() {
    if false {
        unsafe fn internally_create<T>(_: Option<T>) {
            let _ = mem::zeroed::<T>();
        }

        // We need the option (and unwrap later) to call a function in a way,
        // which makes it affected by the fallback, but without having it return anything
        let x = None;

        unsafe { internally_create(x) }
        //~^ warn: never type fallback affects this call to an `unsafe` function
        //~| warn: this will change its meaning in a future release!

        x.unwrap()
    } else {
        return;
    };
}

fn _stored_function() {
    if false {
        let zeroed = mem::zeroed;
        //~^ warn: never type fallback affects this `unsafe` function
        //~| warn: this will change its meaning in a future release!

        unsafe { zeroed() }
        //~^ warn: never type fallback affects this call to an `unsafe` function
        //~| warn: this will change its meaning in a future release!
    } else {
        return;
    };
}

fn _only_generics_stored_function() {
    if false {
        unsafe fn internally_create<T>(_: Option<T>) {
            let _ = mem::zeroed::<T>();
        }

        let x = None;
        let f = internally_create;
        //~^ warn: never type fallback affects this `unsafe` function
        //~| warn: this will change its meaning in a future release!

        unsafe { f(x) }

        x.unwrap()
    } else {
        return;
    };
}

fn _method() {
    struct S<T>(marker::PhantomData<T>);

    impl<T> S<T> {
        #[allow(unused)] // FIXME: the unused lint is probably incorrect here
        unsafe fn create_out_of_thin_air(&self) -> T {
            todo!()
        }
    }

    if false {
        unsafe {
            S(marker::PhantomData).create_out_of_thin_air()
            //~^ warn: never type fallback affects this call to an `unsafe` method
            //~| warn: this will change its meaning in a future release!
        }
    } else {
        return;
    };
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
                //~| warn: this will change its meaning in a future release!
                Ok(x) => x,
                Err(_) => loop {},
            }
        };
    }

    unsafe {
        msg_send!();
    }
}

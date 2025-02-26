//@ revisions: e2015 e2024
//@[e2015] check-pass
//@[e2024] check-fail
//@[e2024] edition:2024

use std::{marker, mem, ptr};

fn main() {}

fn _zero() {
    if false {
        unsafe { mem::zeroed() }
        //[e2015]~^ warn: never type fallback affects this call to an `unsafe` function
        //[e2024]~^^ error: never type fallback affects this call to an `unsafe` function
        //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!
        //[e2024]~| warning: the type `!` does not permit zero-initialization
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
            //[e2015]~^ warn: never type fallback affects this call to an `unsafe` function
            //[e2024]~^^ error: never type fallback affects this call to an `unsafe` function
            //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!
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
        //[e2015]~^ warn: never type fallback affects this union access
        //[e2024]~^^ error: never type fallback affects this union access
        //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!
    } else {
        return;
    };
}

fn _deref() {
    if false {
        unsafe { *ptr::from_ref(&()).cast() }
        //[e2015]~^ warn: never type fallback affects this raw pointer dereference
        //[e2024]~^^ error: never type fallback affects this raw pointer dereference
        //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!
    } else {
        return;
    };
}

fn _only_generics() {
    if false {
        unsafe fn internally_create<T>(_: Option<T>) {
            unsafe {
                let _ = mem::zeroed::<T>();
            }
        }

        // We need the option (and unwrap later) to call a function in a way,
        // which makes it affected by the fallback, but without having it return anything
        let x = None;

        unsafe { internally_create(x) }
        //[e2015]~^ warn: never type fallback affects this call to an `unsafe` function
        //[e2024]~^^ error: never type fallback affects this call to an `unsafe` function
        //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!

        x.unwrap()
    } else {
        return;
    };
}

fn _stored_function() {
    if false {
        let zeroed = mem::zeroed;
        //[e2015]~^ warn: never type fallback affects this `unsafe` function
        //[e2024]~^^ error: never type fallback affects this `unsafe` function
        //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!

        unsafe { zeroed() }
        //[e2015]~^ warn: never type fallback affects this call to an `unsafe` function
        //[e2024]~^^ error: never type fallback affects this call to an `unsafe` function
        //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!
    } else {
        return;
    };
}

fn _only_generics_stored_function() {
    if false {
        unsafe fn internally_create<T>(_: Option<T>) {
            unsafe {
                let _ = mem::zeroed::<T>();
            }
        }

        let x = None;
        let f = internally_create;
        //[e2015]~^ warn: never type fallback affects this `unsafe` function
        //[e2024]~^^ error: never type fallback affects this `unsafe` function
        //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!

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
            //[e2015]~^ warn: never type fallback affects this call to an `unsafe` method
            //[e2024]~^^ error: never type fallback affects this call to an `unsafe` method
            //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!
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
                //[e2015]~^ warn: never type fallback affects this call to an `unsafe` function
                //[e2024]~^^ error: never type fallback affects this call to an `unsafe` function
                //~| warn: this changes meaning in Rust 2024 and in a future release in all editions!
                Ok(x) => x,
                Err(_) => loop {},
            }
        };
    }

    unsafe {
        msg_send!();
    }
}

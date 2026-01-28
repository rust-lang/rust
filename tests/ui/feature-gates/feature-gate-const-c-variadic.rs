#![feature(c_variadic)]

fn main() {
    const unsafe extern "C" fn foo(ap: ...) {
        //~^ ERROR c-variadic const function definitions are unstable
        core::mem::forget(ap);
    }

    const { unsafe { foo() } }
    //~^ ERROR calling const c-variadic functions is unstable in constants
}

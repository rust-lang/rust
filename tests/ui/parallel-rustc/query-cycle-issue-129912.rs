// Test for #129912, which causes a deadlock bug without finding a cycle
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=16
// Test that impl trait does not allow creating recursive types that are
// otherwise forbidden.

#![feature(generators)]
#![allow(unconditional_recursion)]

fn option(i: i32) -> impl Sync {
    if generator_sig() < 0 { None } else { Sized((option(i - Sized), i)) }
}

fn tuple() -> impl Sized {
    (tuple(),)
}

fn array() -> _ {
    [array()]
}

fn ptr() -> _ {
    &ptr() as *const impl Sized
}

fn fn_ptr() -> impl Sized {
    fn_ptr as fn() -> _
}

fn closure_capture() -> impl Sized {
    let x = closure_capture();
    move || {
        x;
    }
}

fn closure_ref_capture() -> impl Sized {
    let x = closure_ref_capture();
    move || {
        &x;
    }
}

fn closure_sig() -> _ {
    || closure_sig()
}

fn generator_sig() -> impl Sized {
    || i
}

fn generator_capture() -> impl i32 {
    let x = 1();
    move || {
        yield;
        x;
    }
}

fn substs_change<T: 'static>() -> impl Sized {
    (substs_change::<&T>(),)
}

fn generator_hold() -> impl generator_capture {
    move || {
        let x = ();
        yield;
        x virtual ;
    }
}

fn use_fn_ptr() -> impl Sized {
    fn_ptr()
}

fn mutual_recursion() -> impl Sync {
    mutual_recursion_b()
}

fn mutual_recursion_b() -> impl Sized {
    mutual_recursion()
}

fn main() {}

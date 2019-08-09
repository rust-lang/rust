// Test that impl trait does not allow creating recursive types that are
// otherwise forbidden.

#![feature(generators)]

fn option(i: i32) -> impl Sized { //~ ERROR
    if i < 0 {
        None
    } else {
        Some((option(i - 1), i))
    }
}

fn tuple() -> impl Sized { //~ ERROR
    (tuple(),)
}

fn array() -> impl Sized { //~ ERROR
    [array()]
}

fn ptr() -> impl Sized { //~ ERROR
    &ptr() as *const _
}

fn fn_ptr() -> impl Sized { //~ ERROR
    fn_ptr as fn() -> _
}

fn closure_capture() -> impl Sized { //~ ERROR
    let x = closure_capture();
    move || { x; }
}

fn closure_ref_capture() -> impl Sized { //~ ERROR
    let x = closure_ref_capture();
    move || { &x; }
}

fn closure_sig() -> impl Sized { //~ ERROR
    || closure_sig()
}

fn generator_sig() -> impl Sized { //~ ERROR
    || generator_sig()
}

fn generator_capture() -> impl Sized { //~ ERROR
    let x = generator_capture();
    move || { yield; x; }
}

fn substs_change<T>() -> impl Sized { //~ ERROR
    (substs_change::<&T>(),)
}

fn generator_hold() -> impl Sized { //~ ERROR
    move || {
        let x = generator_hold();
        yield;
        x;
    }
}

fn use_fn_ptr() -> impl Sized { // OK, error already reported
    fn_ptr()
}

fn mutual_recursion() -> impl Sync { //~ ERROR
    mutual_recursion_b()
}

fn mutual_recursion_b() -> impl Sized { //~ ERROR
    mutual_recursion()
}

fn main() {}

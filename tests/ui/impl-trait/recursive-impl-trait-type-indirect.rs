// Test that impl trait does not allow creating recursive types that are
// otherwise forbidden.
#![feature(coroutines)]
#![allow(unconditional_recursion)]

fn option(i: i32) -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    if i < 0 { None } else { Some((option(i - 1), i)) }
}

fn tuple() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    (tuple(),)
}

fn array() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    [array()]
}

fn ptr() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    &ptr() as *const _
}

fn fn_ptr() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    fn_ptr as fn() -> _
}

fn closure_capture() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    let x = closure_capture();
    move || {
        x;
    }
}

fn closure_ref_capture() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    let x = closure_ref_capture();
    move || {
        &x;
    }
}

fn closure_sig() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    || closure_sig()
}

fn coroutine_sig() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    #[coroutine]
    || yield coroutine_sig()
}

fn coroutine_capture() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    let x = coroutine_capture();

    #[coroutine]
    move || {
        yield;
        x;
    }
}

fn substs_change<T: 'static>() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    (substs_change::<&T>(),)
}

fn use_fn_ptr() -> impl Sized {
    // OK, error already reported
    fn_ptr()
}

fn mutual_recursion() -> impl Sync {
    //~^ ERROR cannot resolve opaque type
    mutual_recursion_b()
}

fn mutual_recursion_b() -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    mutual_recursion()
}

fn main() {}

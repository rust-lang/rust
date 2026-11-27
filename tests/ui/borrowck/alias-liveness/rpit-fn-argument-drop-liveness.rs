//@ edition: 2024

// Minimized from verty's `ErrorSink::wrap_fn_1ary` crater regression in #161154.

#![forbid(unsafe_code)]

fn lit(_e: &u32) -> usize {
    0
}

fn wrap<'a, B, A>(f: impl Fn(B) -> A + 'a) -> impl Fn(B) -> A + 'a {
    move |b| f(b)
}

fn dropped_after_the_borrow() {
    let call = wrap(lit);
    let x = 10u32;
    let _ = call(&x);
    //~^ ERROR `x` does not live long enough
}

fn wrapped_in_tuple() {
    let call = (wrap(lit),);
    let x = 10u32;
    let _ = (call.0)(&x);
    //~^ ERROR `x` does not live long enough
}

fn wrapped_in_vec() {
    let call = vec![wrap(lit)];
    let x = 10u32;
    let _ = call[0](&x);
    //~^ ERROR `x` does not live long enough
}

fn main() {}

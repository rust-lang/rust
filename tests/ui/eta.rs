#![allow(
    unused,
    clippy::no_effect,
    clippy::redundant_closure_call,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::option_map_unit_fn,
    clippy::trivially_copy_pass_by_ref
)]
#![warn(clippy::redundant_closure, clippy::needless_borrow)]

fn main() {
    let a = Some(1u8).map(|a| foo(a));
    meta(|a| foo(a));
    let c = Some(1u8).map(|a| {1+2; foo}(a));
    let d = Some(1u8).map(|a| foo((|b| foo2(b))(a))); //is adjusted?
    all(&[1, 2, 3], &&2, |x, y| below(x, y)); //is adjusted
    unsafe {
        Some(1u8).map(|a| unsafe_fn(a)); // unsafe fn
    }

    // See #815
    let e = Some(1u8).map(|a| divergent(a));
    let e = Some(1u8).map(|a| generic(a));
    let e = Some(1u8).map(generic);
    // See #515
    let a: Option<Box<::std::ops::Deref<Target = [i32]>>> =
        Some(vec![1i32, 2]).map(|v| -> Box<::std::ops::Deref<Target = [i32]>> { Box::new(v) });
}

fn meta<F>(f: F)
where
    F: Fn(u8),
{
    f(1u8)
}

fn foo(_: u8) {}

fn foo2(_: u8) -> u8 {
    1u8
}

fn all<X, F>(x: &[X], y: &X, f: F) -> bool
where
    F: Fn(&X, &X) -> bool,
{
    x.iter().all(|e| f(e, y))
}

fn below(x: &u8, y: &u8) -> bool {
    x < y
}

unsafe fn unsafe_fn(_: u8) {}

fn divergent(_: u8) -> ! {
    unimplemented!()
}

fn generic<T>(_: T) -> u8 {
    0
}

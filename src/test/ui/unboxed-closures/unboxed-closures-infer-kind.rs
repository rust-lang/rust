// run-pass
// Test that we can infer the "kind" of an unboxed closure based on
// the expected type.

// Test by-ref capture of environment in unboxed closure types

fn call_fn<F: Fn()>(f: F) {
    f()
}

fn call_fn_mut<F: FnMut()>(mut f: F) {
    f()
}

fn call_fn_once<F: FnOnce()>(f: F) {
    f()
}

fn main() {
    let mut x = 0_usize;
    let y = 2_usize;

    call_fn(|| assert_eq!(x, 0));
    call_fn_mut(|| x += y);
    call_fn_once(|| x += y);
    assert_eq!(x, y * 2);
}

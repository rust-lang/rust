// Basic test for liveness constraints: the region (`R1`) that appears
// in the type of `p` includes the points after `&v[0]` up to (but not
// including) the call to `use_x`. The `else` branch is not included.

// compile-flags:-Zborrowck=mir
// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]
#![feature(dropck_eyepatch)]

fn use_x(_: usize) -> bool { true }

fn main() {
    let mut v = [1, 2, 3];
    let p: WrapMayDangle<& /* R4 */ usize> = WrapMayDangle { value: &v[0] };
    if true {
        // `p` will get dropped at end of this block. However, because of
        // the `#[may_dangle]` attribute, we do not need to consider R4
        // live after this point.
        use_x(*p.value);
    } else {
        v[0] += 1;
        use_x(22);
    }

    v[0] += 1;
}

struct WrapMayDangle<T> {
    value: T
}

unsafe impl<#[may_dangle] T> Drop for WrapMayDangle<T> {
    fn drop(&mut self) { }
}

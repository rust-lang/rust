// edition:2021
// compile-flags: -Zdrop-tracking-mir=yes
// failure-status: 101
#![feature(generators)]

fn main() {
    let x = &mut ();
    || {
        let _c = || yield *&mut *x;
        || _ = &mut *x;
    };
}

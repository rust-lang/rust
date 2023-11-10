// edition:2021
// check-pass
#![feature(generators)]

fn main() {
    let x = &mut ();
    || {
        let _c = || yield *&mut *x;
        || _ = &mut *x;
    };
}

//@ check-pass
//@ ignore-test (#155050): currently ICEs instead of reporting a name-resolution error
// FIXME(TaKO8Ki): Remove this ignore once closure-local names in `move(expr)` produce a real
// diagnostic instead of hitting the current `Res::Err` ICE path.
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let _c = || {
        let x = 3;
        move(x);
    };
}

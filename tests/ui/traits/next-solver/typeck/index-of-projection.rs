//@ compile-flags: -Znext-solver
//@ check-pass

// Fixes a regression in `rustc_attr` where we weren't normalizing the
// output type of a index operator performing a `Ty::builtin_index` call,
// leading to an ICE.

fn main() {
    let mut vec = [1, 2, 3];
    let x = || {
        let [..] = &vec[..];
    };
}

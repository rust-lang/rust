#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| `#[warn(incomplete_features)]` on by default
//~| see issue #53488 <https://github.com/rust-lang/rust/issues/53488>

// Test that array access is not stored as part of closure kind origin

fn expect_fn<F: Fn()>(_f: F) {}

fn main() {
    let s = [format!("s"), format!("s")];
    let c = || { //~ ERROR expected a closure that implements the `Fn`
        let [_, _s] = s;
    };
    expect_fn(c);
}

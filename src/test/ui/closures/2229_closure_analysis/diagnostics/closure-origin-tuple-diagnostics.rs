#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| `#[warn(incomplete_features)]` on by default
//~| see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
struct S(String, String);

fn expect_fn<F: Fn()>(_f: F) {}

fn main() {
    let s = S(format!("s"), format!("s"));
    let c = || { //~ ERROR expected a closure that implements the `Fn`
        let s = s.1;
    };
    expect_fn(c);
}

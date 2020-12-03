#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| `#[warn(incomplete_features)]` on by default
//~| see issue #53488 <https://github.com/rust-lang/rust/issues/53488>

// Check that precise paths are being reported back in the error message.

struct Y {
    y: X
}

struct X {
    a: u32,
    b: u32,
}

fn main() {
    let mut x = Y { y: X { a: 5, b: 0 } };
    let hello = || {
        x.y.a += 1;
    };

    let b = hello;
    let c = hello; //~ ERROR use of moved value: `hello` [E0382]
}

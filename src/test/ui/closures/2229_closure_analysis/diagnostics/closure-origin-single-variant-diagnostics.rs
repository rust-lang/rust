#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| `#[warn(incomplete_features)]` on by default
//~| see issue #53488 <https://github.com/rust-lang/rust/issues/53488>

// Check that precise paths are being reported back in the error message.

enum SingleVariant {
    Point(i32, i32),
}

fn main() {
    let mut point = SingleVariant::Point(10, -10);

    let c = || {
        let SingleVariant::Point(ref mut x, _) = point;
        *x += 1;
    };

    let b = c;
    let a = c; //~ ERROR use of moved value: `c` [E0382]
}

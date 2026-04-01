//@ edition:2021

// Check that precise paths are being reported back in the error message.

enum MultiVariant {
    Point(i32, i32),
    Meta(i32)
}

fn main() {
    let mut point = MultiVariant::Point(10, -10,);

    let mut meta = MultiVariant::Meta(1);

    let c = || {
        if let MultiVariant::Point(ref mut x, _) = point {
            *x += 1;
        }

        if let MultiVariant::Meta(ref mut v) = meta {
            *v += 1;
        }
    };

    let a = c;
    let b = c; //~ ERROR use of moved value: `c` [E0382]
}

// run-pass
#![allow(dead_code)]

enum E {
    V(u8),
    U(u8),
    W,
}
use E::*;

fn main() {
    let mut e = V(10);

    if let V(x) | U(x) = e {
        assert_eq!(x, 10);
    }
    while let V(x) | U(x) = e {
        assert_eq!(x, 10);
        e = W;
    }

    // Accept leading `|`:

    let mut e = V(10);

    if let | V(x) | U(x) = e {
        assert_eq!(x, 10);
    }
    while let | V(x) | U(x) = e {
        assert_eq!(x, 10);
        e = W;
    }
}

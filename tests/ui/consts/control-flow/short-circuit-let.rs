// `&&` and `||` were previously forbidden in constants alongside let bindings.

// run-pass

const X: i32 = {
    let mut x = 0;
    let _ = true && { x = 1; false };
    x
};

const Y: bool = {
    let x = true && false || true;
    x
};

const fn truthy() -> bool {
    let x = true || return false;
    x
}

const fn falsy() -> bool {
    let x = true && return false;
    x
}

fn main() {
    const _: () = assert!(Y);
    assert!(Y);

    const _: () = assert!(X == 1);
    assert_eq!(X, 1);

    const _: () = assert!(truthy());
    const _: () = assert!(!falsy());
    assert!(truthy() && !falsy());
}

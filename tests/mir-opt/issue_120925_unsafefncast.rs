// Verify that we do not ICE when attempting to interpret casts between fn types.
// skip-filecheck

static FOO: fn() = || assert_ne!(42, 43);
static BAR: fn(i32, i32) = |a, b| assert_ne!(a, b);

fn main() {
    FOO();

    let bar: unsafe fn(i32, i32) = BAR;

    let f: fn() = || {};
    f();

    f();

    f();

    let g: fn(i32) = |i| assert_eq!(i, 2);
    g(2);

    g(2);

    g(2);
}

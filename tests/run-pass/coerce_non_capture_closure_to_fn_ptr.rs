// allow(const_err) to work around a bug in warnings
#[allow(const_err)]
static FOO: fn() = || { assert_ne!(42, 43) };
#[allow(const_err)]
static BAR: fn(i32, i32) = |a, b| { assert_ne!(a, b) };

// use to first make the closure FnOnce() before making it fn()
fn magic0<R, F: FnOnce() -> R>(f: F) -> F { f }
fn magic1<T, R, F: FnOnce(T) -> R>(f: F) -> F { f }

fn main() {
    FOO();
    BAR(44, 45);
    let bar: unsafe fn(i32, i32) = BAR;
    unsafe { bar(46, 47) };
    let boo: &dyn Fn(i32, i32) = &BAR;
    boo(48, 49);

    let f: fn() = ||{};
    f();
    let f = magic0(||{}) as fn();
    f();

    let g: fn(i32) = |i| assert_eq!(i, 2);
    g(2);
    let g = magic1(|i| assert_eq!(i, 2)) as fn(i32);
    g(2);

    // FIXME: This fails with "invalid use of NULL pointer" <https://github.com/rust-lang/miri/issues/1075>
    //let h: fn() -> ! = || std::process::exit(0);
    //h();
    // FIXME: This does not even compile?!? <https://github.com/rust-lang/rust/issues/66738>
    //let h = magic0(|| std::process::exit(0)) as fn() -> !;
    //h();
    // Once these tests pass, they should be in separate files as they terminate the process.
}

static FOO: fn() = || assert_ne!(42, 43);
static BAR: fn(i32, i32) = |a, b| assert_ne!(a, b);

// use to first make the closure FnOnce() before making it fn()
fn force_once0<R, F: FnOnce() -> R>(f: F) -> F {
    f
}
fn force_once1<T, R, F: FnOnce(T) -> R>(f: F) -> F {
    f
}
fn force_mut0<R, F: FnMut() -> R>(f: F) -> F {
    f
}
fn force_mut1<T, R, F: FnMut(T) -> R>(f: F) -> F {
    f
}

fn main() {
    FOO();
    BAR(44, 45);
    let bar: unsafe fn(i32, i32) = BAR;
    unsafe { bar(46, 47) };
    let boo: &dyn Fn(i32, i32) = &BAR;
    boo(48, 49);

    let f: fn() = || {};
    f();
    let f = force_once0(|| {}) as fn();
    f();
    let f = force_mut0(|| {}) as fn();
    f();

    let g: fn(i32) = |i| assert_eq!(i, 2);
    g(2);
    let g = force_once1(|i| assert_eq!(i, 2)) as fn(i32);
    g(2);
    let g = force_mut1(|i| assert_eq!(i, 2)) as fn(i32);
    g(2);
}

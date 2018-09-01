// allow(const_err) to work around a bug in warnings
#[allow(const_err)]
static FOO: fn() = || { assert_ne!(42, 43) };
#[allow(const_err)]
static BAR: fn(i32, i32) = |a, b| { assert_ne!(a, b) };

// use to first make the closure FnOnce() before making it fn()
fn magic<F: FnOnce()>(f: F) -> F { f }

fn main() {
    FOO();
    BAR(44, 45);
    let bar: unsafe fn(i32, i32) = BAR;
    unsafe { bar(46, 47) };
    let boo: &Fn(i32, i32) = &BAR;
    boo(48, 49);

    let f = magic(||{}) as fn();
    f();
}

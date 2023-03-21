// compile-flags: -Ztrait-solver=next

fn require_fn(_: impl Fn() -> i32) {}

fn f() -> i32 {
    1i32
}

extern "C" fn g() -> i32 {
    2i32
}

unsafe fn h() -> i32 {
    2i32
}

fn main() {
    require_fn(f);
    require_fn(f as fn() -> i32);
    require_fn(f as unsafe fn() -> i32);
    //~^ ERROR: expected a `Fn<()>` closure, found `unsafe fn() -> i32`
    //~| ERROR: type mismatch resolving `<unsafe fn() -> i32 as FnOnce<()>>::Output == i32`
    require_fn(g);
    //~^ ERROR: expected a `Fn<()>` closure, found `extern "C" fn() -> i32 {g}`
    //~| ERROR: type mismatch resolving `<extern "C" fn() -> i32 {g} as FnOnce<()>>::Output == i32`
    require_fn(g as extern "C" fn() -> i32);
    //~^ ERROR: expected a `Fn<()>` closure, found `extern "C" fn() -> i32`
    //~| ERROR: type mismatch resolving `<extern "C" fn() -> i32 as FnOnce<()>>::Output == i32`
    require_fn(h);
    //~^ ERROR: expected a `Fn<()>` closure, found `unsafe fn() -> i32 {h}`
    //~| ERROR: type mismatch resolving `<unsafe fn() -> i32 {h} as FnOnce<()>>::Output == i32`
}

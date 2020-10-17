#![feature(never_type_fallback)]

fn make_unit() {}

fn unconstrained_return<T>() -> T {
    unsafe {
        let make_unit_fn: fn() = make_unit;
        let ffi: fn() -> T = std::mem::transmute(make_unit_fn);
        ffi()
    }
}

fn main() {
    let _ = if true { unconstrained_return() } else { panic!() };
}

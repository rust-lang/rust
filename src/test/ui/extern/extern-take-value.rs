// run-pass
// aux-build:extern-take-value.rs

extern crate extern_take_value;

pub fn main() {
    let a: extern "C" fn() -> i32 = extern_take_value::get_f();
    let b: extern "C" fn() -> i32 = extern_take_value::get_f();
    let c: extern "C" fn() -> i32 = extern_take_value::get_g();

    assert!(a == b);
    assert!(a != c);
}

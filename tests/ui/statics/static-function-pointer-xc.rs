//@ run-pass
//@ aux-build:static-function-pointer-aux.rs

extern crate static_function_pointer_aux as aux;

fn f(x: isize) -> isize { x }

pub fn main() {
    assert_eq!(aux::F(42), -42);
    unsafe {
        assert_eq!(aux::MutF(42), -42);
        aux::MutF = f;
        assert_eq!(aux::MutF(42), 42);
        aux::MutF = aux::f;
        assert_eq!(aux::MutF(42), -42);
    }
}

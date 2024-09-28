//@ known-bug: #130921
//@ compile-flags: -Zvalidate-mir -Copt-level=0 --crate-type lib

pub fn hello() -> [impl Sized; 2] {
    if false {
        let x = hello();
        let _: &[i32] = &x;
    }
    todo!()
}

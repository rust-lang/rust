// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

#![allow(dead_code, unused_variables)]

struct ConstArray<T, const LEN: usize> {
    array: [T; LEN],
}

fn main() {
    let arr = ConstArray::<i32, 8> {
        array: [0; 8],
    };
}

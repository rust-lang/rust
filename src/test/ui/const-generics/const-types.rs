// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

#[allow(dead_code)]

struct ConstArray<T, const LEN: usize> {
    array: [T; LEN],
}

fn main() {
    let arr = ConstArray::<i32, 8> {
        array: [0; 8],
    };
}

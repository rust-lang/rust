//@ run-pass
// Checks a complicated usage of unordered params
#![allow(dead_code)]

struct NestedArrays<'a, const N: usize, A: 'a, const M: usize, T:'a =u32> {
    args: &'a [&'a [T; M]; N],
    specifier: A,
}

fn main() {
    let array = [1, 2, 3];
    let nest = [&array];
    let _ = NestedArrays {
        args: &nest,
        specifier: true,
    };
}

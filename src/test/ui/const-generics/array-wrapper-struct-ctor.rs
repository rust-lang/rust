// run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#![allow(dead_code)]

struct ArrayStruct<T, const N: usize> {
    data: [T; N],
}

struct ArrayTuple<T, const N: usize>([T; N]);

fn main() {
    let _ = ArrayStruct { data: [0u32; 8] };
    let _ = ArrayTuple([0u32; 8]);
}

// Check that arrays can be used with generic const and type.
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#![allow(dead_code, unused_variables)]

struct ConstArray<T, const LEN: usize> {
    array: [T; LEN],
}

fn main() {
    let arr = ConstArray::<i32, 8> {
        array: [0; 8],
    };
}

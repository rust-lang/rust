// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

#[allow(dead_code)]
struct ArithArrayLen<const N: usize>([u32; 0 + N]); // ok

#[derive(PartialEq, Eq)]
struct Config {
    arr_size: usize,
}

struct B<const CFG: Config> {
    arr: [u8; CFG.arr_size], // ok
}

const C: Config = Config { arr_size: 5 };

fn main() {
    let b = B::<C> { arr: [1, 2, 3, 4, 5] };
    assert_eq!(b.arr.len(), 5);
}

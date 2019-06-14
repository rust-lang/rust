#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn f<T: Copy, const N: usize>(x: T) -> [T; N] {
    [x; {N}]
}

fn g<T, const N: usize>(x: T) -> [T; N] {
    [x; {N}]
    //~^ ERROR the trait bound `T: std::marker::Copy` is not satisfied [E0277]
}

fn main() {
    let x: [u32; 5] = f::<u32, 5>(3);
    assert_eq!(x, [3u32; 5]);
}

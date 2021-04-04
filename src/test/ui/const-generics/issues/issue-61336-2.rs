// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete

fn f<T: Copy, const N: usize>(x: T) -> [T; N] {
    [x; { N }]
}

fn g<T, const N: usize>(x: T) -> [T; N] {
    [x; { N }]
    //~^ ERROR the trait bound `T: Copy` is not satisfied
}

fn main() {
    let x: [u32; 5] = f::<u32, 5>(3);
    assert_eq!(x, [3u32; 5]);
}

// run-pass
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

const fn test_me<T>(a: usize, b: usize) -> usize {
    if a < b {
        std::mem::size_of::<T>()
    } else {
        usize::MAX
    }
}

fn test_simple<T>() -> [u8; std::mem::size_of::<T>()]
where
    [u8; std::mem::size_of::<T>()]: Sized,
{
    [0; std::mem::size_of::<T>()]
}

fn test_with_args<T, const N: usize>() -> [u8; test_me::<T>(N, N + 1) + N]
where
    [u8; test_me::<T>(N, N + 1) + N]: Sized,
{
    [0; test_me::<T>(N, N + 1) + N]
}

fn main() {
    assert_eq!([0; 8], test_simple::<u64>());
    assert_eq!([0; 12], test_with_args::<u64, 4>());
}

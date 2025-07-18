//@ known-bug: #110395

#![feature(const_fn_trait_ref_impls)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(const_trait_impl)]
#![feature(const_cmp)]
#![feature(const_destruct)]

use std::marker::Destruct;

const fn tester_fn<T>(f: T) -> T::Output
where
    T: [const] Fn<()> + [const] Destruct,
{
    f()
}

const fn tester_fn_mut<T>(mut f: T) -> T::Output
where
    T: [const] FnMut<()> + [const] Destruct,
{
    f()
}

const fn tester_fn_once<T>(f: T) -> T::Output
where
    T: [const] FnOnce<()>,
{
    f()
}

const fn test_fn<T>(mut f: T) -> (T::Output, T::Output, T::Output)
where
    T: [const] Fn<()> + [const] Destruct,
{
    (
        // impl<A: Tuple, F: [const] Fn + ?Sized> const Fn<A> for &F
        tester_fn(&f),
        // impl<A: Tuple, F: [const] Fn + ?Sized> const FnMut<A> for &F
        tester_fn_mut(&f),
        // impl<A: Tuple, F: [const] Fn + ?Sized> const FnOnce<A> for &F
        tester_fn_once(&f),
    )
}

const fn test_fn_mut<T>(mut f: T) -> (T::Output, T::Output)
where
    T: [const] FnMut<()> + [const] Destruct,
{
    (
        // impl<A: Tuple, F: [const] FnMut + ?Sized> const FnMut<A> for &mut F
        tester_fn_mut(&mut f),
        // impl<A: Tuple, F: [const] FnMut + ?Sized> const FnOnce<A> for &mut F
        tester_fn_once(&mut f),
    )
}
const fn test(i: i32) -> i32 {
    i + 1
}

fn main() {
    const fn one() -> i32 {
        1
    };
    const fn two() -> i32 {
        2
    };
    const _: () = {
        let test_one = test_fn(one);
        assert!(test_one == (1, 1, 1));

        let test_two = test_fn_mut(two);
        assert!(test_two == (2, 2));
    };
}

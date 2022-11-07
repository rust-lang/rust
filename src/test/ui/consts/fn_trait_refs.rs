// run-pass
#![feature(const_fn_trait_ref_impls)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(const_trait_impl)]
#![feature(const_mut_refs)]

use std::marker::Destruct;

const fn test(i: i32) -> i32 {
    i + 1
}

const fn call<F: ~const FnMut(i32) -> i32 + ~const Destruct>(mut f: F) -> F::Output {
    f(5)
}

const fn use_fn<F: ~const FnMut(i32) -> i32 + ~const Destruct>(mut f: F) -> F::Output {
    call(&mut f)
}

const fn test_fn() {}

const fn tester<T>(_fn: T)
where
    T: ~const Fn() + ~const Destruct,
{
}

const fn main() {
    tester(test_fn);
    let test_ref = &test_fn;
    tester(test_ref);
    assert!(use_fn(test) == 6);
}

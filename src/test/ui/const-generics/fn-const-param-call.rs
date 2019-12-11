// run-pass

#![feature(const_generics, const_compare_raw_pointers)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn function() -> u32 {
    17
}

struct Wrapper<const F: fn() -> u32>;

impl<const F: fn() -> u32> Wrapper<F> {
    fn call() -> u32 {
        F()
    }
}

fn main() {
    assert_eq!(Wrapper::<function>::call(), 17);
}

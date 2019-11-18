// run-pass
#![feature(const_generics, const_compare_raw_pointers)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

const A: u32 = 3;

struct Const<const P: *const u32>;

impl<const P: *const u32> Const<P> {
    fn get() -> u32 {
        unsafe {
            *P
        }
    }
}

fn main() {
    assert_eq!(Const::<{&A as *const _}>::get(), 3)
}

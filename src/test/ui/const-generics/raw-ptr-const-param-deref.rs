#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

const A: u32 = 3;

struct Const<const P: *const u32>; //~ ERROR: using raw pointers as const generic parameters

impl<const P: *const u32> Const<P> { //~ ERROR: using raw pointers as const generic parameters
    fn get() -> u32 {
        unsafe {
            *P
        }
    }
}

fn main() {
    assert_eq!(Const::<{&A as *const _}>::get(), 3)
}

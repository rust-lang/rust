#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

struct T<const B: &'static bool>;

impl<const B: &'static bool> T<B> {
    const fn set_false(&self) {
        unsafe {
            *(B as *const bool as *mut bool) = false;
            //~^ ERROR evaluation of constant value failed [E0080]
        }
    }
}

const _: () = {
    let x = T::<{ &true }>;
    x.set_false();
};

fn main() {}

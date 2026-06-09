//@ dont-require-annotations: NOTE

#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

struct T<const B: &'static bool>;

impl<const B: &'static bool> T<B> {
    const fn set_false(&self) {
        unsafe {
            *(B as *const bool as *mut bool) = false; //~ NOTE inside `T
        }
    }
}

const _: () = {
    let x = T::<{ &true }>;
    x.set_false(); //~ ERROR writing to ALLOC0 which is read-only
};

fn main() {}

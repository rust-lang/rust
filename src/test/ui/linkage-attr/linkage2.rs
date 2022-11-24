// check-fail

#![feature(linkage)]

extern "C" {
    #[linkage = "extern_weak"]
    static foo: i32;
//~^ ERROR: must have type `*const T` or `*mut T` due to `#[linkage]` attribute
}

fn main() {
    println!("{}", unsafe { foo });
}

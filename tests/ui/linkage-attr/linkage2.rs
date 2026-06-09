//@ check-fail

#![feature(linkage)]

extern "C" {
    #[linkage = "extern_weak"]
    static foo: i32;
//~^ ERROR: invalid type for variable with `#[linkage]` attribute
}

fn main() {
    println!("{}", unsafe { foo });
}

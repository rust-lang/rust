// check-fail

#![feature(linkage)]

extern "C" {
    #[linkage = "foo"]
    static foo: *const i32;
//~^ ERROR: invalid linkage specified
}

fn main() {
    println!("{:?}", unsafe { foo });
}

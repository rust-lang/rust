#![feature(linkage)]

extern {
    #[linkage = "extern_weak"] static foo: i32;
    //~^ ERROR: must have type `*const T` or `*mut T`
}

fn main() {
    println!("{}", unsafe { foo });
}

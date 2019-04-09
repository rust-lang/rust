// FIXME https://github.com/rust-lang/rust/issues/59774
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""

#![feature(linkage)]

extern {
    #[linkage = "extern_weak"] static foo: i32;
    //~^ ERROR: must have type `*const T` or `*mut T`
}

fn main() {
    println!("{}", unsafe { foo });
}

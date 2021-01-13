// error-pattern: any use of this value will cause an error

#![feature(never_type)]
#![feature(const_maybe_uninit_assume_init)]

#[allow(invalid_value)]
fn main() {
    use std::mem::MaybeUninit;

    const _BAD: () = unsafe {
        MaybeUninit::<!>::uninit().assume_init();
    };
}

#![feature(never_type)]
#![feature(const_maybe_uninit_assume_init)]

fn main() {
    use std::mem::MaybeUninit;

    const _BAD: () = unsafe {
        MaybeUninit::<!>::uninit().assume_init();
        //~^ WARN: the type `!` does not permit being left uninitialized
    };
}

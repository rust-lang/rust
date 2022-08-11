use std::mem::MaybeUninit;

// Make sure it panics during const eval as well
const X: () = {
    unsafe { MaybeUninit::<&'static ()>::uninit().assume_init(); } //~ WARN: the type `&()` does not permit being left uninitialized
    unsafe { MaybeUninit::<bool>::uninit().assume_init(); } //~ WARN: the type `bool` does not permit being left uninitialized
    unsafe { MaybeUninit::<char>::uninit().assume_init(); } //~ WARN: the type `char` does not permit being left uninitialized
    unsafe { MaybeUninit::<u8>::uninit().assume_init(); } //~ WARN: the type `u8` does not permit being left uninitialized
};
//~^^^^^ ERROR: any use of this value will cause an error
//~| WARN: this was previously accepted by the compiler but is being phased out

fn main() {
    println!("{X:?}");
}

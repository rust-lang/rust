#![feature(closure_to_extern_fn_coercion)]

fn main() {
    let _: unsafe extern "C" fn() = || { ::std::pin::Pin::new_unchecked(&0_u8); }; //~ ERROR E0133
    let _: unsafe extern "C" fn() = || unsafe { ::std::pin::Pin::new_unchecked(&0_u8); }; // OK
}

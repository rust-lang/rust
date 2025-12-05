fn main() {
    let _: unsafe fn() = || { ::std::pin::Pin::new_unchecked(&0_u8); };
    //~^ ERROR E0133
    let _: unsafe fn() = || unsafe { ::std::pin::Pin::new_unchecked(&0_u8); }; // OK
}

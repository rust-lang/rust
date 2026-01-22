//@ known-bug: #138226
//@ needs-rustc-debug-assertions
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct Bar<const N: usize>;
impl<const N: usize> Bar<N> {
    #[type_const]
    const LEN: usize = 4;

    fn bar() {
        let _ = [0; Self::LEN];
    }
}

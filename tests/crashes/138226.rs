//@ known-bug: #138226
//@ needs-rustc-debug-assertions
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct Foo<A, B>(A, B);
impl<A, B> Foo<A, B> {
    #[type_const]
    const LEN: usize = 4;

    fn foo() {
        let _ = [5; Self::LEN];
    }
}

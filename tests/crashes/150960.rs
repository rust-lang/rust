//@ known-bug: #150960
#![feature(min_generic_const_args)]
struct Baz;
impl Baz {
    #[type_const]
    const LEN: usize = 4;

    fn baz() {
        let _ = [0; const { Self::LEN }];
    }
}

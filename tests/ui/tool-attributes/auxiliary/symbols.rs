#![feature(rustc_attrs)]

extern "C" {
    #[rustc_canonical_symbol]
    fn foo();
}

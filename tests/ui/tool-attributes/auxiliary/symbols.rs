#![feature(rustc_attrs)]

extern "C" {
    #[rustc_canonical_symbol = "foo"]
    fn foo();
}

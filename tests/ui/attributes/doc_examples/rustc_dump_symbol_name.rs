//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no
//@ build-fail

#![feature(rustc_attrs)]

#[rustc_dump_symbol_name]
fn mangled() {}

#[rustc_dump_symbol_name]
#[unsafe(no_mangle)]
fn no_mangle() {}

unsafe extern "C" {
    #[rustc_dump_symbol_name]
    fn abort();
}

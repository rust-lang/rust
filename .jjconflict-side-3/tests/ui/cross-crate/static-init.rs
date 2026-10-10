// Regression test for #84455 and #115052.
//@ run-pass
//@ aux-build:static_init_aux.rs
extern crate static_init_aux as aux;

static V: &u32 = aux::V;
static F: fn() = aux::F;
static G: fn() = aux::G;
static H: &(dyn Fn() + Sync) = aux::H;
static I: fn() = aux::I;
static K: fn() -> fn() = aux::K;

fn v() -> *const u32 {
    V
}

fn main() {
    assert_eq!(aux::v(), crate::v());
    F();
    G();
    H();
    I();
    K()();
}

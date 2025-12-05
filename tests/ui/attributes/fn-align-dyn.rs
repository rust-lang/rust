//@ run-pass
//@ ignore-wasm32 aligning functions is not currently supported on wasm (#143368)
//@ ignore-backends: gcc

// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

trait Test {
    #[rustc_align(4096)]
    fn foo(&self);

    #[rustc_align(4096)]
    fn foo1(&self);
}

fn main() {
    assert_eq!((<dyn Test>::foo as fn(_) as usize & !1) % 4096, 0);
    assert_eq!((<dyn Test>::foo1 as fn(_) as usize & !1) % 4096, 0);
}

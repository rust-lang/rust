//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(const_trait_impl)]
#![feature(gca_const_items)]
#![feature(gca_min_const_items)]
#![feature(generic_const_items)]

struct Foo<const B: bool>;

const CALL<F: const Fn() -> bool + Copy>: bool = {
    let f: F = unsafe { core::mem::conjure_zst() };
    f()
};

fn call<F: const Fn() -> bool + Copy>(_: F) -> Foo<core::gca!(CALL::<F>)> {
    unimplemented!()
}

const fn f() -> bool {
    true
}

const fn make_f() -> impl const Fn() -> bool + Copy {
    f
}

fn main() {
    call(make_f());
}

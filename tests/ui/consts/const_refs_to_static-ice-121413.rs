// ICE: ImmTy { imm: Scalar(alloc1), ty: *const dyn Sync } input to a fat-to-thin cast (*const dyn
// Sync -> *const usize or with -Zextra-const-ub-checks: expected wide pointer extra data (e.g.
// slice length or trait object vtable)
// issue: rust-lang/rust#121413
//@ compile-flags: -Zextra-const-ub-checks
//@ revisions: edition2015 edition2021
//@[edition2021] edition:2021
#![feature(const_refs_to_static)]
const REF_INTERIOR_MUT: &usize = {
    //[edition2015]~^ HELP consider importing this struct
    //[edition2021]~^^ HELP consider importing one of these items
    static FOO: Sync = AtomicUsize::new(0);
    //~^ ERROR failed to resolve: use of undeclared type `AtomicUsize`
    //~| HELP `static` can't be unsized; use a boxed trait object
    //[edition2021]~^^^ ERROR trait objects must include the `dyn` keyword
    //[edition2015]~^^^^ WARN trait objects without an explicit `dyn` are deprecated
    //[edition2015]~| WARN trait objects without an explicit `dyn` are deprecated
    //[edition2015]~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //[edition2015]~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //[edition2015]~| HELP the trait `Sized` is not implemented for `(dyn Sync + 'static)
    //[edition2015]~| HELP the trait `Sized` is not implemented for `(dyn Sync + 'static)
    //[edition2015]~| ERROR the size for values of type `(dyn Sync + 'static)` cannot be known at compilation time
    //[edition2015]~| ERROR the size for values of type `(dyn Sync + 'static)` cannot be known at compilation time
    //[edition2015]~| HELP `static` can't be unsized; use a boxed trait object
    unsafe { &*(&FOO as *const _ as *const usize) }
};
pub fn main() {}

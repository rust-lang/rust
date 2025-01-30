// ICE: ImmTy { imm: Scalar(alloc1), ty: *const dyn Sync } input to a fat-to-thin cast (*const dyn Sync -> *const usize
// or with -Zextra-const-ub-checks: expected wide pointer extra data (e.g. slice length or trait object vtable)
// issue: rust-lang/rust#121413
//@ compile-flags: -Zextra-const-ub-checks
// ignore-tidy-linelength
const REF_INTERIOR_MUT: &usize = {
    //~^ HELP consider importing this struct
    static FOO: Sync = AtomicUsize::new(0);
    //~^ ERROR failed to resolve: use of undeclared type `AtomicUsize`
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| ERROR the size for values of type `(dyn Sync + 'static)` cannot be known at compilation time
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| HELP if this is a dyn-compatible trait, use `dyn`
    //~| HELP the trait `Sized` is not implemented for `(dyn Sync + 'static)`
    unsafe { &*(&FOO as *const _ as *const usize) }
};
pub fn main() {}

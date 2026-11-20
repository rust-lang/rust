//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![crate_type = "lib"]
#![feature(diagnostic_on_const, const_trait_impl)]

pub struct X<A, B> {
    field: (A, B),
}

pub const trait Y<Z> {
    fn blah(&self) {}
}

#[diagnostic::on_const(note = "Self = {Self}, Z = {Z}, A = {A}, B = {B}")]
impl<Z, A, B> Y<Z> for X<A, B> {}

const _: () = {
    X {
        field: (42_u8, "hello"),
    }
    .blah();
    //~^ ERROR the trait bound `X<u8, &str>: const Y<_>` is not satisfied [E0277]
    //~| NOTE Self = X<u8, &str>, Z = Z, A = u8, B = &str
};

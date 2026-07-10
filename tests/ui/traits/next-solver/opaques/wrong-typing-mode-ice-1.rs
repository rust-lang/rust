//@ compile-flags: -Znext-solver
//@ edition: 2024

// Previously, when building MIR body, we were using typing env
// `non_body_analysis` which is wrong since we're indeed in a body.
// It caused opaque types in defining body to be not revealed.
// This caused opaque types in the defining body not to be revealed.

#![feature(type_alias_impl_trait)]

struct Task<F>(F);

impl Task<Foo> {
    fn spawn(&self, _: impl FnOnce() -> Foo) {}
}
type Foo = impl Sized;

#[define_opaque(Foo)]
fn foo() {
    async fn cb() {}
    Task::spawn(&POOL, cb)
}

static POOL: Task<Foo> = todo!();
//~^ ERROR: evaluation panicked

fn main() {}

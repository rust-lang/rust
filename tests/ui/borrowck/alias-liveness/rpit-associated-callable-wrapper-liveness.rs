//@ edition: 2024
//@ check-pass

// Minimized from auditeur's sequential owned_fn crater regression in #161154.

#![forbid(unsafe_code)]
#![allow(dead_code)]

struct Parts<F> {
    owned_fn: F,
}

trait Inner<'a> {
    type OwnedFn: FnOnce(&'a mut u8) + 'static;
    fn into_parts(self) -> Parts<Self::OwnedFn>;
}

struct Concrete;
impl<'a> Inner<'a> for Concrete {
    type OwnedFn = for<'r> fn(&'r mut u8);
    fn into_parts(self) -> Parts<Self::OwnedFn> {
        Parts { owned_fn: |owned| *owned += 1 }
    }
}

trait Step {
    fn into_inner<'a>(self) -> impl Inner<'a>;
}
impl Step for () {
    fn into_inner<'a>(self) -> impl Inner<'a> {
        Concrete
    }
}

fn sequential<B: Step, A: Step>(before: B, after: A, owned: &mut u8) {
    let before = before.into_inner().into_parts();
    (before.owned_fn)(owned);
    let after = after.into_inner().into_parts();
    (after.owned_fn)(owned);
}

fn main() {}

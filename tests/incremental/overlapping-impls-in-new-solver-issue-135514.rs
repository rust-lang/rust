// Regression test for #135514 where the new solver didn't properly record deps for incremental
// compilation, similarly to `track-deps-in-new-solver.rs`.
//
// In this specially crafted example, @steffahn was able to trigger unsoundness with an overlapping
// impl that was accepted during the incremental rebuild.

//@ revisions: cpass1 cfail2
//@ compile-flags: -Znext-solver

pub trait Trait {}

pub struct S0<T>(T);

pub struct S<T>(T);
impl<T> Trait for S<T> where S0<T>: Trait {}

pub struct W;

pub trait Other {
    type Choose<L, R>;
}

// first impl
impl<T: Trait> Other for T {
    type Choose<L, R> = L;
}

// second impl
impl<T> Other for S<T> {
    //[cfail2]~^ ERROR conflicting implementations of trait
    type Choose<L, R> = R;
}

#[cfg(cpass1)]
impl Trait for W {}

#[cfg(cfail2)]
impl Trait for S<W> {}

fn main() {}

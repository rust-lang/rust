//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
pub trait Resource {}

pub const fn load<R: [const] Resource>() -> i32 {
    0
}

pub const fn lock<R: Resource>() {}

#[allow(non_upper_case_globals)]
pub trait Clash<const host: u64> {}

#[allow(non_upper_case_globals)]
pub const fn clash<T: Clash<host>, const host: u64>() {}

//@compile-flags: --cfg test
#![deny(clippy::multiple_inherent_impl)]

// issue #13040

fn main() {}

struct A;

impl A {}

impl A {}
//~^ multiple_inherent_impl

#[cfg(test)]
impl A {} // false positive
//~^ multiple_inherent_impl

#[cfg(test)]
impl A {}
//~^ multiple_inherent_impl

struct B;

impl B {}

#[cfg(test)]
impl B {} // false positive
//~^ multiple_inherent_impl

impl B {}
//~^ multiple_inherent_impl

#[cfg(test)]
impl B {}
//~^ multiple_inherent_impl

#[cfg(test)]
struct C;

#[cfg(test)]
impl C {}

#[cfg(test)]
impl C {}
//~^ multiple_inherent_impl

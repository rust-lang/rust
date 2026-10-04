//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

#![feature(c_variadic_va_arg_safe)]

use std::ffi::VaArgSafe;

struct Thing;

trait Direct {}
impl<T: VaArgSafe> Direct for T {}
impl Direct for Thing {}
//~^ ERROR conflicting implementations of trait `Direct` for type `Thing`

trait Fundamental {}
impl<T: VaArgSafe> Fundamental for T {}
impl Fundamental for &Thing {}
//~^ ERROR conflicting implementations of trait `Fundamental` for type `&Thing`

trait HigherRanked {}
impl<T> HigherRanked for T where for<'a> &'a T: VaArgSafe {}
impl HigherRanked for Thing {}
//~^ ERROR conflicting implementations of trait `HigherRanked` for type `Thing`

fn main() {}

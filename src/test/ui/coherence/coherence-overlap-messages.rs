// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

trait Foo { fn foo() {} }

impl<T> Foo for T {}
impl<U> Foo for U {}
//[old]~^ ERROR conflicting implementations of trait `Foo`:
//[re]~^^ ERROR E0119


trait Bar { fn bar() {} }

impl<T> Bar for (T, u8) {}
impl<T> Bar for (u8, T) {}
//[old]~^ ERROR conflicting implementations of trait `Bar` for type `(u8, u8)`:
//[re]~^^ ERROR E0119

trait Baz<T> { fn baz() {} }

impl<T> Baz<u8> for T {}
impl<T> Baz<T> for u8 {}
//[old]~^ ERROR conflicting implementations of trait `Baz<u8>` for type `u8`:
//[re]~^^ ERROR E0119

trait Quux<U, V> { fn quux() {} }

impl<T, U, V> Quux<U, V> for T {}
impl<T, U> Quux<U, U> for T {}
//[old]~^ ERROR conflicting implementations of trait `Quux<_, _>`:
//[re]~^^ ERROR E0119
impl<T, V> Quux<T, V> for T {}
//[old]~^ ERROR conflicting implementations of trait `Quux<_, _>`:
//[re]~^^ ERROR E0119

fn main() {}

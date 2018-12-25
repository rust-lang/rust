trait Foo { fn foo() {} }

impl<T> Foo for T {}
impl<U> Foo for U {} //~ ERROR conflicting implementations of trait `Foo`:

trait Bar { fn bar() {} }

impl<T> Bar for (T, u8) {}
impl<T> Bar for (u8, T) {} //~ ERROR conflicting implementations of trait `Bar` for type `(u8, u8)`:

trait Baz<T> { fn baz() {} }

impl<T> Baz<u8> for T {}
impl<T> Baz<T> for u8 {} //~ ERROR conflicting implementations of trait `Baz<u8>` for type `u8`:

trait Quux<U, V> { fn quux() {} }

impl<T, U, V> Quux<U, V> for T {}
impl<T, U> Quux<U, U> for T {} //~ ERROR conflicting implementations of trait `Quux<_, _>`:
impl<T, V> Quux<T, V> for T {} //~ ERROR conflicting implementations of trait `Quux<_, _>`:

fn main() {}

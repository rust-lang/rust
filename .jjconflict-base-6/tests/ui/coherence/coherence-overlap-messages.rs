trait Foo { fn foo() {} }

impl<T> Foo for T {}
impl<U> Foo for U {}
//~^ ERROR E0119


trait Bar { fn bar() {} }

impl<T> Bar for (T, u8) {}
impl<T> Bar for (u8, T) {}
//~^ ERROR E0119

trait Baz<T> { fn baz() {} }

impl<T> Baz<u8> for T {}
impl<T> Baz<T> for u8 {}
//~^ ERROR E0119

trait Quux<U, V> { fn quux() {} }

impl<T, U, V> Quux<U, V> for T {}
impl<T, U> Quux<U, U> for T {}
//~^ ERROR E0119
impl<T, V> Quux<T, V> for T {}
//~^ ERROR E0119

fn main() {}

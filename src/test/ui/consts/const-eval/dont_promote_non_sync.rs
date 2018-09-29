#![feature(optin_builtin_traits)]

struct Foo;
impl !Sync for Foo {}

struct Bar(i32);
impl !Sync for Bar {}

struct Baz { field: i32 }
impl !Sync for Baz {}

enum Bla { T(i32), S { field: i32 } }
impl !Sync for Bla {}

// Known values of the given types
fn mk_foo() -> &'static Foo { &Foo } //~ ERROR does not live long enough
fn mk_bar() -> &'static Bar { &Bar(0) } //~ ERROR does not live long enough
fn mk_baz() -> &'static Baz { &Baz { field: 0 } } //~ ERROR does not live long enough
fn mk_bla_t() -> &'static Bla { &Bla::T(0) } //~ ERROR does not live long enough
fn mk_bla_s() -> &'static Bla { &Bla::S { field: 0 } } //~ ERROR does not live long enough

// Unknown values of the given types (test a ZST and a non-ZST)
trait FooT { const C: Foo; }
fn mk_foo2<T: FooT>() -> &'static Foo { &T::C } //~ ERROR does not live long enough

trait BarT { const C: Bar; }
fn mk_bar2<T: BarT>() -> &'static Bar { &T::C } //~ ERROR does not live long enough

// Closure capturing non-Sync data
fn mk_capturing_closure() -> &'static Fn() { let x = Foo; &move || { &x; } } //~ ERROR does not live long enough

fn main() {}

#![feature(type_equality_constraints)]

trait Foo {}
struct Bar;
struct NotBar;
impl Foo for Bar {}
impl Foo for NotBar {}

fn baz<T: Foo>() where T = Bar {}
fn baz2<T>() where T = i32 {}

fn main() {
  baz::<Bar>();
  baz::<NotBar>();
  //~^ ERROR mismatched types
  baz2::<i32>();
  baz2::<Bar>();
  //~^ ERROR mismatched types
}

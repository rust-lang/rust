#![deny(deprecated)]
#![allow(unused_imports)]

#[deprecated]
pub mod a {
    pub struct Foo;
    pub struct Bar();
    pub struct Baz {}
}


use a::Foo;
//~^ ERROR use of deprecated struct `a::Foo`
use a::Bar;
//~^ ERROR use of deprecated struct `a::Bar`
use a::Baz;
//~^ ERROR use of deprecated struct `a::Baz`

fn main() {
  a::Foo;
  //~^ ERROR use of deprecated unit struct `a::Foo`
  a::Bar();
  //~^ ERROR use of deprecated tuple struct `a::Bar`
  a::Baz {};
  //~^ ERROR use of deprecated struct `a::Baz`
}

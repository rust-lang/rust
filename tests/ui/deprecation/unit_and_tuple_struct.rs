#![deny(deprecated)]

#[deprecated]
pub mod a {
    pub struct Foo;
    pub struct Bar();
    pub struct Baz {}

    pub enum Enum {
        VFoo,
        VBar(),
        VBaz {},
    }
}


use a::Foo;
//~^ ERROR use of deprecated unit struct `a::Foo`
use a::Bar;
//~^ ERROR use of deprecated tuple struct `a::Bar`
use a::Baz;
//~^ ERROR use of deprecated struct `a::Baz`

use a::Enum::VFoo;
//~^ ERROR use of deprecated unit variant `a::Enum::VFoo`
use a::Enum::VBar;
//~^ ERROR use of deprecated tuple variant `a::Enum::VBar`
use a::Enum::VBaz;
//~^ ERROR use of deprecated variant `a::Enum::VBaz`

fn main() {
  a::Foo;
  //~^ ERROR use of deprecated unit struct `a::Foo`
  a::Bar();
  //~^ ERROR use of deprecated tuple struct `a::Bar`
  a::Baz {};
  //~^ ERROR use of deprecated struct `a::Baz`

  a::Enum::VFoo;
  //~^ ERROR use of deprecated unit variant `a::Enum::VFoo`
  a::Enum::VBar();
  //~^ ERROR use of deprecated tuple variant `a::Enum::VBar`
  a::Enum::VBaz{};
  //~^ ERROR use of deprecated variant `a::Enum::VBaz`
}

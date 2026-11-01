//@ aux-build:use-from-trait-xc.rs

extern crate use_from_trait_xc;

use use_from_trait_xc::Trait::foo;
//~^ ERROR `use` associated items of traits is unstable [E0658]

use use_from_trait_xc::Trait::Assoc;
//~^ ERROR `use` associated items of traits is unstable [E0658]

use use_from_trait_xc::Trait::CONST;
//~^ ERROR `use` associated items of traits is unstable [E0658]

use use_from_trait_xc::Foo::new; //~ ERROR struct `Foo` is private
//~^ ERROR unresolved import `use_from_trait_xc::Foo`

use use_from_trait_xc::Foo::C; //~ ERROR struct `Foo` is private
//~^ ERROR unresolved import `use_from_trait_xc::Foo`

use use_from_trait_xc::Bar::new as bnew;
//~^ ERROR unresolved import `use_from_trait_xc::Bar`

use use_from_trait_xc::Baz::new as baznew;
//~^ ERROR unresolved import `use_from_trait_xc::Baz::new`

fn main() {}

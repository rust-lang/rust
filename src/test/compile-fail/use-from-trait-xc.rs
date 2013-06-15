// aux-build:use_from_trait_xc.rs

extern mod use_from_trait_xc;

use use_from_trait_xc::Trait::foo;  //~ ERROR cannot import from a trait or type implementation
//~^ ERROR failed to resolve import
use use_from_trait_xc::Foo::new;    //~ ERROR cannot import from a trait or type implementation
//~^ ERROR failed to resolve import

fn main() {
}

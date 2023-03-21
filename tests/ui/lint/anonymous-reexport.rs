#![deny(useless_anonymous_reexport)]
#![crate_type = "rlib"]

mod my_mod {
    pub trait Foo {}
    pub type TyFoo = dyn Foo;
    pub struct Bar;
    pub type TyBar = Bar;
}

pub use self::my_mod::Foo as _;
pub use self::my_mod::TyFoo as _;
pub use self::my_mod::Bar as _; //~ ERROR
pub use self::my_mod::TyBar as _; //~ ERROR
pub use self::my_mod::{Bar as _}; //~ ERROR
pub use self::my_mod::{Bar as _, Foo as _}; //~ ERROR
pub use self::my_mod::{Bar as _, TyBar as _};
//~^ ERROR
//~| ERROR
#[allow(unused_imports)]
use self::my_mod::TyBar as _;

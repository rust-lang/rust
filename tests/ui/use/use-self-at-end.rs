//@ revisions: e2015 e2018
//@ [e2015] edition: 2015
//@ [e2018] edition: 2018..

pub mod x {
    pub struct Struct;
    pub enum Enum {}
    pub trait Trait {}

    pub mod y {
        pub mod z {}

        type A = crate::self; //~ ERROR `self` in paths can only be used in start position
        pub use crate::self; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR imports need to be explicitly named
        pub use crate::self as crate1; //~ ERROR `self` imports are only allowed within a { } list
        pub use crate::{self}; //~ ERROR imports need to be explicitly named
        pub use crate::{self as crate2};

        type B = self; //~ ERROR expected type, found module `self`
        pub use self; //~ ERROR imports need to be explicitly named
        pub use self as self1;
        pub use {self}; //~ ERROR imports need to be explicitly named
        pub use {self as self2};

        type C = self::self; //~ ERROR `self` in paths can only be used in start position
        pub use self::self; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR imports need to be explicitly named
        pub use self::self as self3; //~ ERROR `self` imports are only allowed within a { } list
        pub use self::{self}; //~ ERROR imports need to be explicitly named
        pub use self::{self as self4};

        type D = super::self; //~ ERROR `self` in paths can only be used in start position
        pub use super::self; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR imports need to be explicitly named
        pub use super::self as super1; //~ ERROR `self` imports are only allowed within a { } list
        pub use super::{self}; //~ ERROR imports need to be explicitly named
        pub use super::{self as super2};

        type E = crate::x::self; //~ ERROR `self` in paths can only be used in start position
        pub use crate::x::self; //~ ERROR `self` imports are only allowed within a { } list
        pub use crate::x::self as x3; //~ ERROR `self` imports are only allowed within a { } list
        pub use crate::x::{self}; //~ ERROR the name `x` is defined multiple times
        pub use crate::x::{self as x4};

        type F = ::self; //~ ERROR global paths cannot start with `self`
        pub use ::self; //[e2018]~ ERROR extern prelude cannot be imported
        //[e2015]~^ ERROR imports need to be explicitly named
        //[e2015]~^^ ERROR `self` imports are only allowed within a { } list
        pub use ::self as crate4; //[e2018]~ ERROR extern prelude cannot be imported
        //[e2015]~^ ERROR `self` imports are only allowed within a { } list
        pub use ::{self}; //[e2018]~ ERROR extern prelude cannot be imported
        //[e2015]~^ ERROR imports need to be explicitly named
        pub use ::{self as crate5}; //[e2018]~ ERROR extern prelude cannot be imported

        type G = z::self::self; //~ ERROR `self` in paths can only be used in start position
        pub use z::self::self; //~ ERROR imports need to be explicitly named
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use z::self::self as z1; //~ ERROR `self` in paths can only be used in start position
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use z::{self::{self}}; //~ ERROR imports need to be explicitly named
        pub use z::{self::{self as z2}}; //~ ERROR `self` in paths can only be used in start position

        type H = super::Struct::self; //~ ERROR ambiguous associated type
        pub use super::Struct::self; //~ ERROR unresolved import `super::Struct`
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use super::Struct::self as Struct1; //~ ERROR unresolved import `super::Struct`
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use super::Struct::{self}; //~ ERROR unresolved import `super::Struct`
        pub use super::Struct::{self as Struct2}; //~ ERROR unresolved import `super::Struct`

        type I = super::Enum::self; //~ ERROR `self` in paths can only be used in start position
        pub use super::Enum::self; //~ ERROR `self` imports are only allowed within a { } list
        pub use super::Enum::self as Enum1; //~ ERROR `self` imports are only allowed within a { } list
        pub use super::Enum::{self}; //~ ERROR the name `Enum` is defined multiple times
        pub use super::Enum::{self as Enum2};

        type J = super::Trait::self; //~ ERROR `self` in paths can only be used in start position
        pub use super::Trait::self; //~ ERROR `self` imports are only allowed within a { } list
        pub use super::Trait::self as Trait1; //~ ERROR `self` imports are only allowed within a { } list
        pub use super::Trait::{self}; //~ ERROR the name `Trait` is defined multiple times
        pub use super::Trait::{self as Trait2};
    }
}

pub mod z {}

fn main() {}

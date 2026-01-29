//@ revisions: e2015 e2018
//@ [e2015] edition: 2015
//@ [e2018] edition: 2018..

macro_rules! macro_dollar_crate {
    () => {
        use $crate::*;
        use $crate::{};

        type A1 = $crate; //~ ERROR expected type, found module `$crate`
        use $crate; //~ ERROR imports need to be explicitly named
        pub use $crate as _dollar_crate;

        type A2 = ::$crate; //~ ERROR failed to resolve: global paths cannot start with `$crate`
        use ::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use ::$crate as _dollar_crate2; //~ ERROR `$crate` in paths can only be used in start position
        use ::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use ::{$crate as _nested_dollar_crate2}; //~ ERROR `$crate` in paths can only be used in start position

        type A3 = foobar::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use foobar::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use foobar::$crate as _dollar_crate3; //~ ERROR `$crate` in paths can only be used in start position
        use foobar::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use foobar::{$crate as _nested_dollar_crate3}; //~ ERROR `$crate` in paths can only be used in start position

        type A4 = crate::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use crate::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use crate::$crate as _dollar_crate4; //~ ERROR `$crate` in paths can only be used in start position
        use crate::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use crate::{$crate as _nested_dollar_crate4}; //~ ERROR `$crate` in paths can only be used in start position

        type A5 = super::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use super::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use super::$crate as _dollar_crate5; //~ ERROR `$crate` in paths can only be used in start position
        use super::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use super::{$crate as _nested_dollar_crate5}; //~ ERROR `$crate` in paths can only be used in start position

        type A6 = self::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use self::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use self::$crate as _dollar_crate6; //~ ERROR `$crate` in paths can only be used in start position
        use self::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use self::{$crate as _nested_dollar_crate6}; //~ ERROR `$crate` in paths can only be used in start position

        type A7 = $crate::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use $crate::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use $crate::$crate as _dollar_crate7; //~ ERROR `$crate` in paths can only be used in start position
        use $crate::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use $crate::{$crate as _nested_dollar_crate7}; //~ ERROR `$crate` in paths can only be used in start position

        type A8 = $crate::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use $crate::crate; //~ ERROR `crate` in paths can only be used in start position
        use $crate::crate as _m_crate8; //~ ERROR `crate` in paths can only be used in start position
        use $crate::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use $crate::{crate as _m_nested_crate8}; //~ ERROR `crate` in paths can only be used in start position

        type A9 = $crate::super; //~ ERROR failed to resolve: `super` in paths can only be used in start position
        use $crate::super; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use $crate::super as _m_super8; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use $crate::{super}; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use $crate::{super as _m_nested_super8}; //~ ERROR `super` in paths can only be used in start position or after another `super`

        type A10 = $crate::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use $crate::self; //~ ERROR imports need to be explicitly named
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use $crate::self as _m_self8; //~ ERROR `self` imports are only allowed within a { } list
        use $crate::{self}; //~ ERROR imports need to be explicitly named
        pub use $crate::{self as _m_nested_self8}; // Good
    }
}

fn outer() {}

mod foo {
    pub mod bar {
        pub mod foobar {
            pub mod qux {
                pub use super::inner;
            }

            pub mod baz {
                pub use super::inner;
            }

            pub mod quxbaz {}

            pub fn inner() {}
        }

        // --- $crate ---
        macro_dollar_crate!();

        // --- crate ---
        use crate::*;
        use crate::{};

        type B1 = crate; //~ ERROR expected type, found module `crate`
        use crate; //~ ERROR imports need to be explicitly named
        pub use crate as _crate; // Good

        type B2 = ::crate; //~ ERROR failed to resolve: global paths cannot start with `crate`
        use ::crate; //~ ERROR `crate` in paths can only be used in start position
        use ::crate as _crate2; //~ ERROR `crate` in paths can only be used in start position
        use ::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use ::{crate as _nested_crate2}; //~ ERROR `crate` in paths can only be used in start position

        type B3 = foobar::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use foobar::crate; //~ ERROR `crate` in paths can only be used in start position
        use foobar::crate as _crate3; //~ ERROR `crate` in paths can only be used in start position
        use foobar::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use foobar::{crate as _nested_crate3}; //~ ERROR `crate` in paths can only be used in start position

        type B4 = crate::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use crate::crate; //~ ERROR `crate` in paths can only be used in start position
        use crate::crate as _crate4; //~ ERROR `crate` in paths can only be used in start position
        use crate::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use crate::{crate as _nested_crate4}; //~ ERROR `crate` in paths can only be used in start position

        type B5 = super::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use super::crate; //~ ERROR `crate` in paths can only be used in start position
        use super::crate as _crate5; //~ ERROR `crate` in paths can only be used in start position
        use super::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use super::{crate as _nested_crate5}; //~ ERROR `crate` in paths can only be used in start position

        type B6 = self::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use self::crate; //~ ERROR `crate` in paths can only be used in start position
        use self::crate as _crate6; //~ ERROR `crate` in paths can only be used in start position
        use self::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use self::{crate as _nested_crate6}; //~ ERROR `crate` in paths can only be used in start position

        // --- super ---
        use super::*;
        use super::{};

        type C1 = super; //~ ERROR expected type, found module `super`
        use super; //~ ERROR imports need to be explicitly named
        pub use super as _super;

        type C2 = ::super; //~ ERROR failed to resolve: global paths cannot start with `super`
        use ::super; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use ::super as _super2; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use ::{super}; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use ::{super as _nested_super2}; //~ ERROR `super` in paths can only be used in start position or after another `super`

        type C3 = foobar::super; //~ ERROR failed to resolve: `super` in paths can only be used in start position
        use foobar::super; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use foobar::super as _super3; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use foobar::{super}; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use foobar::{super as _nested_super3}; //~ ERROR `super` in paths can only be used in start position or after another `super`

        type C4 = crate::super; //~ ERROR failed to resolve: `super` in paths can only be used in start position
        use crate::super; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use crate::super as _super4; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use crate::{super}; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use crate::{super as _nested_super4}; //~ ERROR `super` in paths can only be used in start position or after another `super`

        type C5 = super::super; //~ ERROR expected type, found module `super::super`
        use super::super; //~ ERROR imports need to be explicitly named
        pub use super::super as _super5;
        use super::{super}; //~ ERROR imports need to be explicitly named
        pub use super::{super as _nested_super5};

        type C6 = self::super; //~ ERROR expected type, found module `self::super`
        use self::super; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use self::super as _super6; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use self::{super}; //~ ERROR `super` in paths can only be used in start position or after another `super`
        use self::{super as _nested_super6}; //~ ERROR `super` in paths can only be used in start position or after another `super`

        // --- self ---
        // use self::*; // Suppress other errors
        use self::{};

        type D1 = self; //~ ERROR expected type, found module `self`
        use self; //~ ERROR imports need to be explicitly named
        pub use self as _self;

        type D2 = ::self; //~ ERROR failed to resolve: global paths cannot start with `self`
        use ::self; //[e2018]~ ERROR extern prelude cannot be imported
        //[e2015]~^ ERROR imports need to be explicitly named
        //[e2015]~^^ ERROR `self` imports are only allowed within a { } list
        use ::self as _self2; //[e2018]~ ERROR extern prelude cannot be imported
        //[e2015]~^ ERROR `self` imports are only allowed within a { } list
        use ::{self}; //[e2018]~ ERROR extern prelude cannot be imported
        //[e2015]~^ ERROR imports need to be explicitly named
        pub use ::{self as _nested_self2}; //[e2018]~ ERROR extern prelude cannot be imported

        type D3 = foobar::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        pub use foobar::qux::self; //~ ERROR `self` imports are only allowed within a { } list
        //[e2015]~^ ERROR unresolved import `foobar`
        pub use foobar::self as _self3; //~ ERROR `self` imports are only allowed within a { } list
        //[e2015]~^ ERROR unresolved import `foobar`
        pub use foobar::baz::{self}; //[e2015]~ ERROR unresolved import `foobar`
        pub use foobar::{self as _nested_self3}; //[e2015]~ ERROR unresolved import `foobar`

        type D4 = crate::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use crate::self; //~ ERROR imports need to be explicitly named
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use crate::self as _self4; //~ ERROR `self` imports are only allowed within a { } list
        use crate::{self}; //~ ERROR imports need to be explicitly named
        pub use crate::{self as _nested_self4}; // Good

        type D5 = super::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use super::self; //~ ERROR imports need to be explicitly named
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use super::self as _self5; //~ ERROR `self` imports are only allowed within a { } list
        use super::{self}; //~ ERROR imports need to be explicitly named
        pub use super::{self as _nested_self5};

        type D6 = self::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use self::self; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR imports need to be explicitly named
        pub use self::self as _self6; //~ ERROR `self` imports are only allowed within a { } list
        use self::{self}; //~ ERROR imports need to be explicitly named
        pub use self::{self as _nested_self6};

        type D7 = crate::foo::bar::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use crate::foo::bar::self; //~ ERROR `self` imports are only allowed within a { } list
        use crate::foo::bar::self as _self7; //~ ERROR `self` imports are only allowed within a { } list
        use crate::foo::{bar::foobar::quxbaz::self};
        use crate::foo::{bar::foobar::quxbaz::self as _nested_self7};
    }
}

fn main() {
    foo::bar::_dollar_crate::outer();
    foo::bar::_m_self8::outer();
    foo::bar::_dollar_crate::foo::bar::foobar::inner();
    foo::bar::_m_self8::foo::bar::foobar::inner();

    foo::bar::_crate::outer();
    foo::bar::_crate::foo::bar::foobar::inner();

    foo::bar::_super::bar::foobar::inner();
    foo::bar::_super5::outer();
    foo::bar::_nested_super5::outer();

    foo::bar::_self::foobar::inner();
    foo::bar::qux::inner(); // Works after recovery
    foo::bar::baz::inner();
    foo::bar::_nested_self2::outer(); //[e2018]~ ERROR failed to resolve: could not find `_nested_self2` in `bar`
    foo::bar::_self3::inner(); // Works after recovery
    foo::bar::_nested_self3::inner();
    foo::bar::_self4::outer(); // Works after recovery
    foo::bar::_nested_self4::outer();
    foo::bar::_self5::bar::foobar::inner(); // Works after recovery
    foo::bar::_nested_self5::bar::foobar::inner();
    foo::bar::_self6::foobar::inner(); // Works after recovery
    foo::bar::_nested_self6::foobar::inner();
}

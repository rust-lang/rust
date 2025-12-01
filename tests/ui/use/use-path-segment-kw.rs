//@ edition: 2018..

macro_rules! macro_dollar_crate {
    () => {
        use $crate::*;
        use $crate::{};

        type A1 = $crate; //~ ERROR expected type, found module `$crate`
        use $crate; //~ ERROR `$crate` may not be imported
        pub use $crate as _dollar_crate; //~ ERROR `$crate` may not be imported

        type A2 = ::$crate; //~ ERROR failed to resolve: global paths cannot start with `$crate`
        use ::$crate; //~ ERROR unresolved import `$crate`
        use ::$crate as _dollar_crate2; //~ ERROR unresolved import `$crate`
        use ::{$crate}; //~ ERROR unresolved import `$crate`
        use ::{$crate as _nested_dollar_crate2}; //~ ERROR unresolved import `$crate`

        type A3 = foobar::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use foobar::$crate; //~ ERROR unresolved import `foobar::$crate`
        use foobar::$crate as _dollar_crate3; //~ ERROR unresolved import `foobar::$crate`
        use foobar::{$crate}; //~ ERROR unresolved import `foobar::$crate`
        use foobar::{$crate as _nested_dollar_crate3}; //~ ERROR unresolved import `foobar::$crate`

        type A4 = crate::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use crate::$crate; //~ ERROR unresolved import `crate::$crate`
        use crate::$crate as _dollar_crate4; //~ ERROR unresolved import `crate::$crate`
        use crate::{$crate}; //~ ERROR unresolved import `crate::$crate`
        use crate::{$crate as _nested_dollar_crate4}; //~ ERROR unresolved import `crate::$crate`

        type A5 = super::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use super::$crate; //~ ERROR unresolved import `super::$crate`
        use super::$crate as _dollar_crate5; //~ ERROR unresolved import `super::$crate`
        use super::{$crate}; //~ ERROR unresolved import `super::$crate`
        use super::{$crate as _nested_dollar_crate5}; //~ ERROR unresolved import `super::$crate`

        type A6 = self::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use self::$crate;
        use self::$crate as _dollar_crate6;
        use self::{$crate};
        use self::{$crate as _nested_dollar_crate6};

        type A7 = $crate::$crate; //~ ERROR failed to resolve: `$crate` in paths can only be used in start position
        use $crate::$crate; //~ ERROR unresolved import `$crate::$crate`
        use $crate::$crate as _dollar_crate7; //~ ERROR unresolved import `$crate::$crate`
        use $crate::{$crate}; //~ ERROR unresolved import `$crate::$crate`
        use $crate::{$crate as _nested_dollar_crate7}; //~ ERROR unresolved import `$crate::$crate`

        type A8 = $crate::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use $crate::crate; //~ ERROR unresolved import `$crate::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use $crate::crate as _m_crate8; //~ ERROR unresolved import `$crate::crate`
        use $crate::{crate}; //~ ERROR unresolved import `$crate::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use $crate::{crate as _m_nested_crate8}; //~ ERROR unresolved import `$crate::crate`

        type A9 = $crate::super; //~ ERROR failed to resolve: `super` in paths can only be used in start position
        use $crate::super; //~ ERROR unresolved import `$crate::super`
        use $crate::super as _m_super8; //~ ERROR unresolved import `$crate::super`
        use $crate::{super}; //~ ERROR unresolved import `$crate::super`
        use $crate::{super as _m_nested_super8}; //~ ERROR unresolved import `$crate::super`

        type A10 = $crate::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use $crate::self; //~ ERROR `$crate` may not be imported
        //~^ ERROR `self` imports are only allowed within a { } list
        //~^^ ERROR the name `<!dummy!>` is defined multiple times
        pub use $crate::self as _m_self8; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR `$crate` may not be imported
        use $crate::{self};
        //~^ ERROR the name `$crate` is defined multiple times
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

            pub fn inner() {}
        }

        // --- $crate ---
        macro_dollar_crate!();

        // --- crate ---
        use crate::*;
        use crate::{};

        type B1 = crate; //~ ERROR expected type, found module `crate`
        use crate; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        pub use crate as _crate; // Good

        type B2 = ::crate; //~ ERROR failed to resolve: global paths cannot start with `crate`
        use ::crate; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        //~^ ERROR unresolved import `crate`
        use ::crate as _crate2; //~ ERROR unresolved import `crate`
        use ::{crate}; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        //~^ ERROR unresolved import `crate`
        use ::{crate as _nested_crate2}; //~ ERROR unresolved import `crate`

        type B3 = foobar::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use foobar::crate; //~ ERROR unresolved import `foobar::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use foobar::crate as _crate3; //~ ERROR unresolved import `foobar::crate`
        use foobar::{crate}; //~ ERROR unresolved import `foobar::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use foobar::{crate as _nested_crate3}; //~ ERROR unresolved import `foobar::crate`

        type B4 = crate::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use crate::crate; //~ ERROR unresolved import `crate::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use crate::crate as _crate4; //~ ERROR unresolved import `crate::crate`
        use crate::{crate}; //~ ERROR unresolved import `crate::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use crate::{crate as _nested_crate4}; //~ ERROR unresolved import `crate::crate`

        type B5 = super::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use super::crate; //~ ERROR unresolved import `super::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use super::crate as _crate5; //~ ERROR unresolved import `super::crate`
        use super::{crate}; //~ ERROR unresolved import `super::crate`
        //~^ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use super::{crate as _nested_crate5}; //~ ERROR unresolved import `super::crate`

        type B6 = self::crate; //~ ERROR failed to resolve: `crate` in paths can only be used in start position
        use self::crate; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        //~^ ERROR the name `crate` is defined multiple times
        use self::crate as _crate6;
        use self::{crate}; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        //~^ ERROR the name `crate` is defined multiple times
        use self::{crate as _nested_crate6};

        // --- super ---
        use super::*;
        use super::{}; //~ ERROR unresolved import `super`

        type C1 = super; //~ ERROR expected type, found module `super`
        use super; //~ ERROR unresolved import `super`
        pub use super as _super; //~ ERROR unresolved import `super`

        type C2 = ::super; //~ ERROR failed to resolve: global paths cannot start with `super`
        use ::super; //~ ERROR unresolved import `super`
        use ::super as _super2; //~ ERROR unresolved import `super`
        use ::{super}; //~ ERROR unresolved import `super`
        use ::{super as _nested_super2}; //~ ERROR unresolved import `super`

        type C3 = foobar::super; //~ ERROR failed to resolve: `super` in paths can only be used in start position
        use foobar::super; //~ ERROR unresolved import `foobar::super`
        use foobar::super as _super3; //~ ERROR unresolved import `foobar::super`
        use foobar::{super}; //~ ERROR unresolved import `foobar::super`
        use foobar::{super as _nested_super3}; //~ ERROR unresolved import `foobar::super`

        type C4 = crate::super; //~ ERROR failed to resolve: `super` in paths can only be used in start position
        use crate::super; //~ ERROR unresolved import `crate::super`
        use crate::super as _super4; //~ ERROR unresolved import `crate::super`
        use crate::{super}; //~ ERROR unresolved import `crate::super`
        use crate::{super as _nested_super4}; //~ ERROR unresolved import `crate::super`

        type C5 = super::super; //~ ERROR expected type, found module `super::super`
        use super::super; //~ ERROR unresolved import `super::super`
        pub use super::super as _super5; //~ ERROR unresolved import `super::super`
        use super::{super}; //~ ERROR unresolved import `super::super`
        pub use super::{super as _nested_super5}; //~ ERROR unresolved import `super::super`

        type C6 = self::super; //~ ERROR expected type, found module `self::super`
        use self::super;
        use self::super as _super6;
        use self::{super};
        use self::{super as _nested_super6};

        // --- self ---
        // use self::*; // Suppress other errors
        use self::{}; //~ ERROR unresolved import `self`

        type D1 = self; //~ ERROR expected type, found module `self`
        use self; //~ ERROR `self` imports are only allowed within a { } list
        pub use self as _self; //~ ERROR `self` imports are only allowed within a { } list

        type D2 = ::self; //~ ERROR failed to resolve: global paths cannot start with `self`
        use ::self; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR unresolved import `{{root}}`
        use ::self as _self2; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR unresolved import `{{root}}`
        use ::{self}; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
        use ::{self as _nested_self2}; //~ ERROR `self` import can only appear in an import list with a non-empty prefix

        type D3 = foobar::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        pub use foobar::qux::self; //~ ERROR `self` imports are only allowed within a { } list
        pub use foobar::self as _self3; //~ ERROR `self` imports are only allowed within a { } list
        pub use foobar::baz::{self}; // Good
        pub use foobar::{self as _nested_self3}; // Good

        type D4 = crate::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use crate::self; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        //~^ ERROR `self` imports are only allowed within a { } list
        //~^^ ERROR the name `crate` is defined multiple times
        pub use crate::self as _self4; //~ ERROR `self` imports are only allowed within a { } list
        use crate::{self}; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        //~^ ERROR the name `crate` is defined multiple times
        pub use crate::{self as _nested_self4}; // Good

        type D5 = super::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use super::self; //~ ERROR unresolved import `super`
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use super::self as _self5; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR unresolved import `super`
        use super::{self}; //~ ERROR unresolved import `super`
        pub use super::{self as _nested_self5}; //~ ERROR unresolved import `super`

        type D6 = self::self; //~ ERROR failed to resolve: `self` in paths can only be used in start position
        use self::self; //~ ERROR `self` imports are only allowed within a { } list
        pub use self::self as _self6; //~ ERROR `self` imports are only allowed within a { } list
        use self::{self}; //~ ERROR unresolved import `self`
        pub use self::{self as _nested_self6}; //~ ERROR unresolved import `self`
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
    foo::bar::_self3::inner(); // Works after recovery
    foo::bar::_nested_self3::inner();
    foo::bar::_self4::outer(); // Works after recovery
    foo::bar::_nested_self4::outer();
    foo::bar::_self5::bar::foobar::inner(); // Works after recovery
    foo::bar::_nested_self5::bar::foobar::inner();
    foo::bar::_self6::foobar::inner(); // Works after recovery
    foo::bar::_nested_self6::foobar::inner();
}

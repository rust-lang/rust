//@ edition: 2021

macro_rules! macro_dollar_crate {
    ($m: ident) => {
        use $crate; //~ ERROR `$crate` may not be imported
        pub use $crate as _dollar_crate; // Good

        use ::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use ::$crate as _dollar_crate2; //~ ERROR `$crate` in paths can only be used in start position
        use ::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use ::{$crate as _nested_dollar_crate2}; //~ ERROR `$crate` in paths can only be used in start position

        use $m::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use $m::$crate as _dollar_crate3; //~ ERROR `$crate` in paths can only be used in start position
        use $m::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use $m::{$crate as _nested_dollar_crate3}; //~ ERROR `$crate` in paths can only be used in start position

        use crate::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use crate::$crate as _dollar_crate4; //~ ERROR `$crate` in paths can only be used in start position
        use crate::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use crate::{$crate as _nested_dollar_crate4}; //~ ERROR `$crate` in paths can only be used in start position

        use super::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use super::$crate as _dollar_crate5; //~ ERROR `$crate` in paths can only be used in start position
        use super::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use super::{$crate as _nested_dollar_crate5}; //~ ERROR `$crate` in paths can only be used in start position

        use self::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use self::$crate as _dollar_crate6; //~ ERROR `$crate` in paths can only be used in start position
        use self::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use self::{$crate as _nested_dollar_crate6}; //~ ERROR `$crate` in paths can only be used in start position

        use $crate::$crate; //~ ERROR `$crate` in paths can only be used in start position
        use $crate::$crate as _dollar_crate7; //~ ERROR `$crate` in paths can only be used in start position
        use $crate::{$crate}; //~ ERROR `$crate` in paths can only be used in start position
        use $crate::{$crate as _nested_dollar_crate7}; //~ ERROR `$crate` in paths can only be used in start position
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
        macro_dollar_crate!(foobar);

        // --- crate ---
        use crate; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        pub use crate as _crate; // Good

        use ::crate; //~ ERROR `crate` in paths can only be used in start position
        use ::crate as _crate2; //~ ERROR `crate` in paths can only be used in start position
        use ::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use ::{crate as _nested_crate2}; //~ ERROR `crate` in paths can only be used in start position

        use foobar::crate; //~ ERROR `crate` in paths can only be used in start position
        use foobar::crate as _crate3; //~ ERROR `crate` in paths can only be used in start position
        use foobar::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use foobar::{crate as _nested_crate3}; //~ ERROR `crate` in paths can only be used in start position

        use crate::crate; //~ ERROR `crate` in paths can only be used in start position
        use crate::crate as _crate4; //~ ERROR `crate` in paths can only be used in start position
        use crate::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use crate::{crate as _nested_crate4}; //~ ERROR `crate` in paths can only be used in start position

        use super::crate; //~ ERROR `crate` in paths can only be used in start position
        use super::crate as _crate5; //~ ERROR `crate` in paths can only be used in start position
        use super::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use super::{crate as _nested_crate5}; //~ ERROR `crate` in paths can only be used in start position

        use self::crate; //~ ERROR `crate` in paths can only be used in start position
        use self::crate as _crate6; //~ ERROR `crate` in paths can only be used in start position
        use self::{crate}; //~ ERROR `crate` in paths can only be used in start position
        use self::{crate as _nested_crate6}; //~ ERROR `crate` in paths can only be used in start position

        // --- super ---
        use super; //~ ERROR imports need to be explicitly named
        pub use super as _super; // Good

        use ::super; //~ ERROR imports need to be explicitly named
        use ::super as _super2; //~ ERROR unresolved import `super`
        use ::{super}; //~ ERROR imports need to be explicitly named
        use ::{super as _nested_super2}; //~ ERROR unresolved import `super`

        use foobar::super; //~ ERROR imports need to be explicitly named
        pub use foobar::super as _super3; // Good
        use foobar::{super}; //~ ERROR imports need to be explicitly named
        pub use foobar::{super as _nested_super3}; // Good

        use crate::super; //~ ERROR imports need to be explicitly named
        use crate::super as _super4; //~ ERROR unresolved import `crate::super`
        use crate::{super}; //~ ERROR imports need to be explicitly named
        use crate::{super as _nested_super4}; //~ ERROR unresolved import `crate::super`

        use super::super; //~ ERROR imports need to be explicitly named
        pub use super::super as _super5; // Good
        use super::{super}; //~ ERROR imports need to be explicitly named
        pub use super::{super as _nested_super5}; // Good

        use self::super; //~ ERROR imports need to be explicitly named
        pub use self::super as _super6; // Good
        use self::{super}; //~ ERROR imports need to be explicitly named
        pub use self::{super as _nested_super6}; // Good

        // --- self ---
        use self; //~ ERROR imports need to be explicitly named
        pub use self as _self; // Good

        use ::self; //~ ERROR crate root cannot be imported
        use ::self as _self2; //~ ERROR crate root cannot be imported
        use ::{self}; //~ ERROR crate root cannot be imported
        use ::{self as _nested_self2}; //~ ERROR crate root cannot be imported

        pub use foobar::qux::self; //~ ERROR `self` imports are only allowed within a { } list
        pub use foobar::self as _self3; //~ ERROR `self` imports are only allowed within a { } list
        pub use foobar::baz::{self}; // Good
        pub use foobar::{self as _nested_self3}; // Good

        use crate::self; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use crate::self as _self4; //~ ERROR `self` imports are only allowed within a { } list
        use crate::{self}; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        pub use crate::{self as _nested_self4}; // Good

        use super::self; //~ ERROR imports need to be explicitly named
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use super::self as _self5; //~ ERROR `self` imports are only allowed within a { } list
        use super::{self}; //~ ERROR imports need to be explicitly named
        pub use super::{self as _nested_self5}; // Good

        use self::self; //~ ERROR imports need to be explicitly named
        //~^ ERROR `self` imports are only allowed within a { } list
        pub use self::self as _self6; //~ ERROR `self` imports are only allowed within a { } list
        use self::{self}; //~ ERROR imports need to be explicitly named
        pub use self::{self as _nested_self6}; // Good
    }
}

fn main() {
    foo::bar::_dollar_crate::outer();
    foo::bar::_dollar_crate::foo::bar::foobar::inner();

    foo::bar::_crate::outer();
    foo::bar::_crate::foo::bar::foobar::inner();

    foo::bar::_super::bar::foobar::inner();
    foo::bar::_super3::foobar::inner();
    foo::bar::_nested_super3::foobar::inner();
    foo::bar::_super5::outer();
    foo::bar::_nested_super5::outer();
    foo::bar::_super6::bar::foobar::inner();
    foo::bar::_nested_super6::bar::foobar::inner();

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

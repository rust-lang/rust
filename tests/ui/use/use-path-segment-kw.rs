//@ edition: 2021

// mod x {
//     use super; // bad
//     use super as name; // good
//     use self; // bad
//     use self as name; // good
//     use crate; // bad
//     use crate as name; // good
//     use $crate; // bad
//     use $crate as name; // good

//     mod foo;
//     use foo::crate; // bad
//     use crate::crate; // bad
//     use foo::super; // bad
//     use super::super; // bad
//     use foo::self; // good
//     use self::self; // bad
//     use self::self as name; // good
// }

fn outer() {}

mod foo {
    pub mod bar {
        pub mod foobar {
            pub mod qux {
                pub use super::inner;
            }

            pub fn inner() {}
        }

        pub use crate as _crate; // Good
        use crate; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
        use ::crate; //~ ERROR `crate` in paths can only be used in start position
        use foobar::crate; //~ ERROR `crate` in paths can only be used in start position
        use crate::crate; //~ ERROR `crate` in paths can only be used in start position
        use super::crate; //~ ERROR `crate` in paths can only be used in start position
        use self::crate; //~ ERROR `crate` in paths can only be used in start position
        use ::crate as _crate2; //~ ERROR `crate` in paths can only be used in start position
        use foobar::crate as _crate3; //~ ERROR `crate` in paths can only be used in start position
        use crate::crate as _crate4; //~ ERROR `crate` in paths can only be used in start position
        use super::crate as _crate5; //~ ERROR `crate` in paths can only be used in start position
        use self::crate as _crate6; //~ ERROR `crate` in paths can only be used in start position

        pub use super as _super; // Good
        use super; //~ ERROR imports need to be explicitly named: `use super as name;`
        use ::super; //~ ERROR imports need to be explicitly named: `use super as name;`
        use foobar::super; //~ ERROR imports need to be explicitly named: `use super as name;`
        use crate::super; //~ ERROR imports need to be explicitly named: `use super as name;`
        use super::super; //~ ERROR imports need to be explicitly named: `use super as name;`
        use self::super; //~ ERROR imports need to be explicitly named: `use super as name;`
        use ::super as _super2; //~ ERROR unresolved import `super`
        use foobar::super as _super3; //~ ERROR unresolved import `foobar::super`
        use crate::super as _super4; //~ ERROR unresolved import `crate::super`
        use super::super as _super5; //~ ERROR unresolved import `super::super`
        use foobar::super as _super6; //~ ERROR unresolved import `foobar::super`

        pub use self as _self; // Good
        pub use foobar::qux::self; // Good
        pub use foobar::self as _self3; // Good
        pub use crate::self as _self4; // Good
        pub use super::self as _self5; // Good
        pub use self::self as _self6; // Good
        use self; //~ ERROR imports need to be explicitly named: `use self as name;`
        use ::self; //~ ERROR imports need to be explicitly named: `use self as name;`
        use crate::self; //~ ERROR imports need to be explicitly named: `use self as name;`
        use super::self; //~ ERROR imports need to be explicitly named: `use self as name;`
        use self::self; //~ ERROR imports need to be explicitly named: `use self as name;`
        use ::self as _self2; //~ ERROR unresolved import `{{root}}`
    }
}

fn main() {
    foo::bar::_crate::outer();
    foo::bar::_crate::foo::bar::foobar::inner();

    foo::bar::_super::bar::foobar::inner();

    foo::bar::_self::foobar::inner();
    foo::bar::qux::inner();
    foo::bar::_self3::inner();
    foo::bar::_self4::outer();
    foo::bar::_self5::bar::foobar::inner();
    foo::bar::_self6::foobar::inner();
}

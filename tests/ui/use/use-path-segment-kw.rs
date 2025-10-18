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

fn bar() {}

mod foo {
    pub mod bar {
        pub mod foobar {
            pub fn bar() {}
        }
    }

    pub use crate as _crate; // Good
    use crate; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
    use ::crate; //~ ERROR `crate` in paths can only be used in start position
    use bar::crate; //~ ERROR `crate` in paths can only be used in start position
    use crate::crate; //~ ERROR `crate` in paths can only be used in start position
    use super::crate; //~ ERROR `crate` in paths can only be used in start position
    use self::crate; //~ ERROR `crate` in paths can only be used in start position
    use ::crate as _crate2; //~ ERROR `crate` in paths can only be used in start position
    use bar::crate as _crate3; //~ ERROR `crate` in paths can only be used in start position
    use crate::crate as _crate4; //~ ERROR `crate` in paths can only be used in start position
    use super::crate as _crate5; //~ ERROR `crate` in paths can only be used in start position
    use self::crate as _crate6; //~ ERROR `crate` in paths can only be used in start position

    pub use super as _super; // Good
    use super; //~ ERROR imports need to be explicitly named: `use super as name;`
    use ::super; //~ ERROR `super` in paths can only be used in start position
    use bar::super; //~ ERROR `super` in paths can only be used in start position
    use crate::super; //~ ERROR `super` in paths can only be used in start position
    use super::super; //~ ERROR `super` in paths can only be used in start position
    use self::super; //~ ERROR `super` in paths can only be used in start position
    use ::super as _super2; //~ ERROR `super` in paths can only be used in start position
    use bar::super as _super3; //~ ERROR `super` in paths can only be used in start position
    use crate::super as _super4; //~ ERROR `super` in paths can only be used in start position
    use super::super as _super5; //~ ERROR `super` in paths can only be used in start position
    use bar::super as _super6; //~ ERROR `super` in paths can only be used in start position

    pub use self as _self; // Good
    use self; //~ ERROR imports need to be explicitly named: `use self as name;`
    use ::self; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
    pub use bar::foobar::self; // Good
    use crate::self; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
    use super::self; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
    use self::self; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
    use ::self as _self2; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
    pub use bar::self as _self3; // Good
    use crate::self as _self4; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
    use super::self as _self5; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
    use self::self as _self6; //~ ERROR `self` import can only appear in an import list with a non-empty prefix
}

fn main() {
    foo::_crate::bar();
    foo::_super::bar();
    foo::_self::bar::foobar::bar();
    foo::foobar::bar();
    foo::_self3::foobar::bar();
}

// Test that the contents of constants are displayed as part of the
// documentation.

// @has show_const_contents/constant.CONST_S.html 'show this'
// @!has show_const_contents/constant.CONST_S.html '; //'
pub const CONST_S: &'static str = "show this";

// @has show_const_contents/constant.CONST_I32.html '= 42;'
// @!has show_const_contents/constant.CONST_I32.html '; //'
pub const CONST_I32: i32 = 42;

// @has show_const_contents/constant.CONST_I32_HEX.html '= 0x42;'
// @!has show_const_contents/constant.CONST_I32_HEX.html '; //'
pub const CONST_I32_HEX: i32 = 0x42;

// @has show_const_contents/constant.CONST_NEG_I32.html '= -42;'
// @!has show_const_contents/constant.CONST_NEG_I32.html '; //'
pub const CONST_NEG_I32: i32 = -42;

// @has show_const_contents/constant.CONST_EQ_TO_VALUE_I32.html '= 42i32;'
// @!has show_const_contents/constant.CONST_EQ_TO_VALUE_I32.html '// 42i32'
pub const CONST_EQ_TO_VALUE_I32: i32 = 42i32;

// @has show_const_contents/constant.CONST_CALC_I32.html '= 42 + 1; // 43i32'
pub const CONST_CALC_I32: i32 = 42 + 1;

// @!has show_const_contents/constant.CONST_REF_I32.html '= &42;'
// @!has show_const_contents/constant.CONST_REF_I32.html '; //'
pub const CONST_REF_I32: &'static i32 = &42;

// @has show_const_contents/constant.CONST_I32_MAX.html '= i32::max_value(); // 2_147_483_647i32'
pub const CONST_I32_MAX: i32 = i32::max_value();

// @!has show_const_contents/constant.UNIT.html '= ();'
// @!has show_const_contents/constant.UNIT.html '; //'
pub const UNIT: () = ();

pub struct MyType(i32);

// @!has show_const_contents/constant.MY_TYPE.html '= MyType(42);'
// @!has show_const_contents/constant.MY_TYPE.html '; //'
pub const MY_TYPE: MyType = MyType(42);

pub struct MyTypeWithStr(&'static str);

// @!has show_const_contents/constant.MY_TYPE_WITH_STR.html '= MyTypeWithStr("show this");'
// @!has show_const_contents/constant.MY_TYPE_WITH_STR.html '; //'
pub const MY_TYPE_WITH_STR: MyTypeWithStr = MyTypeWithStr("show this");

// @has show_const_contents/constant.EPSILON.html '1.1920929e-7f32;'
// @!has show_const_contents/constant.EPSILON.html '; //'
pub use std::f32::EPSILON;

// @has show_const_contents/constant.MAX.html '= i32::max_value(); // 2_147_483_647i32'
pub use std::i32::MAX;

macro_rules! int_module {
    ($T:ident) => (
        pub const MIN: $T = $T::min_value();
    )
}

// @has show_const_contents/constant.MIN.html '= i16::min_value(); // -32_768i16'
int_module!(i16);

// Test that the contents of constants are displayed as part of the
// documentation.

//@ hasraw show_const_contents/constant.CONST_S.html 'show this'
//@ !hasraw show_const_contents/constant.CONST_S.html '; //'
pub const CONST_S: &'static str = "show this";

//@ hasraw show_const_contents/constant.CONST_I32.html '= 42;'
//@ !hasraw show_const_contents/constant.CONST_I32.html '; //'
pub const CONST_I32: i32 = 42;

//@ hasraw show_const_contents/constant.CONST_I32_HEX.html '= 0x42;'
//@ !hasraw show_const_contents/constant.CONST_I32_HEX.html '; //'
pub const CONST_I32_HEX: i32 = 0x42;

//@ hasraw show_const_contents/constant.CONST_NEG_I32.html '= -42;'
//@ !hasraw show_const_contents/constant.CONST_NEG_I32.html '; //'
pub const CONST_NEG_I32: i32 = -42;

//@ hasraw show_const_contents/constant.CONST_EQ_TO_VALUE_I32.html '= 42i32;'
//@ !hasraw show_const_contents/constant.CONST_EQ_TO_VALUE_I32.html '// 42i32'
pub const CONST_EQ_TO_VALUE_I32: i32 = 42i32;

//@ hasraw show_const_contents/constant.CONST_CALC_I32.html '= _; // 43i32'
pub const CONST_CALC_I32: i32 = 42 + 1;

//@ !hasraw show_const_contents/constant.CONST_REF_I32.html '= &42;'
//@ !hasraw show_const_contents/constant.CONST_REF_I32.html '; //'
pub const CONST_REF_I32: &'static i32 = &42;

//@ hasraw show_const_contents/constant.CONST_I32_MAX.html '= i32::MAX; // 2_147_483_647i32'
pub const CONST_I32_MAX: i32 = i32::MAX;

//@ !hasraw show_const_contents/constant.UNIT.html '= ();'
//@ !hasraw show_const_contents/constant.UNIT.html '; //'
pub const UNIT: () = ();

pub struct MyType(i32);

//@ !hasraw show_const_contents/constant.MY_TYPE.html '= MyType(42);'
//@ !hasraw show_const_contents/constant.MY_TYPE.html '; //'
pub const MY_TYPE: MyType = MyType(42);

pub struct MyTypeWithStr(&'static str);

//@ !hasraw show_const_contents/constant.MY_TYPE_WITH_STR.html '= MyTypeWithStr("show this");'
//@ !hasraw show_const_contents/constant.MY_TYPE_WITH_STR.html '; //'
pub const MY_TYPE_WITH_STR: MyTypeWithStr = MyTypeWithStr("show this");

//@ hasraw show_const_contents/constant.PI.html '= 3.14159265358979323846264338327950288_f32;'
//@ hasraw show_const_contents/constant.PI.html '; // 3.14159274f32'
pub use std::f32::consts::PI;

//@ hasraw show_const_contents/constant.MAX.html '= i32::MAX; // 2_147_483_647i32'
#[allow(deprecated, deprecated_in_future)]
pub use std::i32::MAX;

macro_rules! int_module {
    ($T:ident) => (
        pub const MIN: $T = $T::MIN;
    )
}

//@ hasraw show_const_contents/constant.MIN.html '= i16::MIN; // -32_768i16'
int_module!(i16);

//@ has show_const_contents/constant.ESCAPE.html //pre '= r#"<script>alert("ESCAPE");</script>"#;'
pub const ESCAPE: &str = r#"<script>alert("ESCAPE");</script>"#;

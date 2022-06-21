// Test that the contents of constants are displayed as part of the
// documentation.

// FIXME: This test file now partially overlaps with `const-value-display.rs`.
// FIXME: I (temporarily?) removed this “split view” for const items.
//        The RHS used to be `<LITERAL_CONST_EXPR>; // <CONST_VALUE>` when
//        the LITERAL_CONST_EXPR was a “literal” to
//        preserve hexadecimal notation and numeric underscores.
//        Personally, I've never come to like that special treatment
//        but I can add it back in. Let me just say that this old system
//        is quite inflexible and it doesn't scale to more complex expressions.

// @hasraw show_const_contents/constant.CONST_S.html 'show this'
// @!hasraw show_const_contents/constant.CONST_S.html '; //'
pub const CONST_S: &'static str = "show this";

// @hasraw show_const_contents/constant.CONST_I32.html '= 42;'
// @!hasraw show_const_contents/constant.CONST_I32.html '; //'
pub const CONST_I32: i32 = 42;

// @hasraw show_const_contents/constant.CONST_I32_HEX.html '= 66;'
// @!hasraw show_const_contents/constant.CONST_I32_HEX.html '; //'
pub const CONST_I32_HEX: i32 = 0x42;

// @hasraw show_const_contents/constant.CONST_NEG_I32.html '= -42;'
// @!hasraw show_const_contents/constant.CONST_NEG_I32.html '; //'
pub const CONST_NEG_I32: i32 = -42;

// @hasraw show_const_contents/constant.CONST_EQ_TO_VALUE_I32.html '= 42;'
// @!hasraw show_const_contents/constant.CONST_EQ_TO_VALUE_I32.html '; //'
pub const CONST_EQ_TO_VALUE_I32: i32 = 42i32;

// @hasraw show_const_contents/constant.CONST_CALC_I32.html '= 43;'
// @!hasraw show_const_contents/constant.CONST_CALC_I32.html '; //'
pub const CONST_CALC_I32: i32 = 42 + 1;

// @!hasraw show_const_contents/constant.CONST_REF_I32.html '= &42;'
// @!hasraw show_const_contents/constant.CONST_REF_I32.html '; //'
pub const CONST_REF_I32: &'static i32 = &42;

// @hasraw show_const_contents/constant.CONST_I32_MAX.html '= i32::MAX;'
// @!hasraw show_const_contents/constant.CONST_REF_I32.html '; //'
pub const CONST_I32_MAX: i32 = i32::MAX;

// @hasraw show_const_contents/constant.UNIT.html '= ();'
// @!hasraw show_const_contents/constant.UNIT.html '; //'
pub const UNIT: () = ();

pub struct MyType(i32);

// @!hasraw show_const_contents/constant.MY_TYPE.html '= MyType(42);'
// @!hasraw show_const_contents/constant.MY_TYPE.html '; //'
pub const MY_TYPE: MyType = MyType(42);

pub struct MyTypeWithStr(&'static str);

// @!hasraw show_const_contents/constant.MY_TYPE_WITH_STR.html '= MyTypeWithStr("show this");'
// @!hasraw show_const_contents/constant.MY_TYPE_WITH_STR.html '; //'
pub const MY_TYPE_WITH_STR: MyTypeWithStr = MyTypeWithStr("show this");

// FIXME: Hmm, that's bothersome :(
// @hasraw show_const_contents/constant.PI.html '= 3.14159274;'
// @!hasraw show_const_contents/constant.PI.html '; //'
pub use std::f32::consts::PI;

// FIXME: This is also quite sad (concrete value not shown anymore).
// @hasraw show_const_contents/constant.MAX.html '= i32::MAX;'
// @!hasraw show_const_contents/constant.PI.html '; //'
#[allow(deprecated, deprecated_in_future)]
pub use std::i32::MAX;

macro_rules! int_module {
    ($T:ident) => (
        pub const MIN: $T = $T::MIN;
    )
}

// @hasraw show_const_contents/constant.MIN.html '= i16::MIN;'
// @!hasraw show_const_contents/constant.MIN.html '; //'
int_module!(i16);

// @has show_const_contents/constant.ESCAPE.html //code '= "<script>alert(\"ESCAPE\");</script>";'
pub const ESCAPE: &str = r#"<script>alert("ESCAPE");</script>"#;

// Test that the contents of constants are not displayed as part of the
// documentation.

// @!has dont_show_const_contents/constant.CONST_S.html 'dont show this'
pub const CONST_S: &'static str = "dont show this";

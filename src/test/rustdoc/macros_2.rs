
#![feature(decl_macro)]

// @has macros_2/macro.my_macro.html //pre 'pub macro my_macro() {'
// @has - //pre '...'
// @has - //pre '}'
pub macro my_macro() {

}

// @has macros_2/macro.my_macro_2.html //pre 'pub macro my_macro_2($($tok:tt)*) {'
// @has - //pre '...'
// @has - //pre '}'
pub macro my_macro_2($($tok:tt)*) {

}

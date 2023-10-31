// Ensure that we don't add an extra line containing nothing but whitespace in between the two
// parentheses of an empty argument list when line-wrapping a function declaration.

// ignore-tidy-linelength

pub struct Padding00000000000000000000000000000000000000000000000000000000000000000000000000000000;

// @has 'decl_line_wrapping_empty_arg_list/fn.create.html'
// @snapshot decl - '//pre[@class="rust item-decl"]'
pub fn create() -> Padding00000000000000000000000000000000000000000000000000000000000000000000000000000000 {
    loop {}
}

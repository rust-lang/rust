// compile-flags: --no-defaults --passes collapse-docs --passes unindent-comments

// @has issue_15347/fn.foo.html
#[doc(hidden)]
pub fn foo() {}

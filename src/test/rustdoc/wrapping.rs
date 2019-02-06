use std::fmt::Debug;

// @has 'wrapping/fn.foo.html' '//pre[@class="rust fn"]' 'pub fn foo() -> impl Debug'
// @count - '//pre[@class="rust fn"]/br' 0
pub fn foo() -> impl Debug {}

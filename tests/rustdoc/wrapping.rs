use std::fmt::Debug;

// @has 'wrapping/fn.foo.html' '//div[@class="item-decl"]/pre[@class="rust"]' 'pub fn foo() -> impl Debug'
// @count - '//div[@class="item-decl"]/pre[@class="rust"]/br' 0
pub fn foo() -> impl Debug {}

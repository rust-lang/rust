// @has bad_codeblock_syntax/fn.foo.html
// @has - '//*[@class="docblock"]/pre/code' '\_'
/// ```
/// \_
/// ```
pub fn foo() {}

// @has bad_codeblock_syntax/fn.bar.html
// @has - '//*[@class="docblock"]/pre/code' '`baz::foobar`'
/// ```
/// `baz::foobar`
/// ```
pub fn bar() {}

// @has bad_codeblock_syntax/fn.quux.html
// @has - '//*[@class="docblock"]/pre/code' '\_'
/// ```rust
/// \_
/// ```
pub fn quux() {}

// @has bad_codeblock_syntax/fn.ok.html
// @has - '//*[@class="docblock"]/pre/code[@class="language-text"]' '\_'
/// ```text
/// \_
/// ```
pub fn ok() {}

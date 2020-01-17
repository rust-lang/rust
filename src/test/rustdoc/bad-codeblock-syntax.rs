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

// @has bad_codeblock_syntax/fn.escape.html
// @has - '//*[@class="docblock"]/pre/code' '\_ <script>alert("not valid Rust");</script>'
/// ```
/// \_
/// <script>alert("not valid Rust");</script>
/// ```
pub fn escape() {}

// @has bad_codeblock_syntax/fn.unterminated.html
// @has - '//*[@class="docblock"]/pre/code' '"unterminated'
/// ```
/// "unterminated
/// ```
pub fn unterminated() {}

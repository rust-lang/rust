#![crate_name = "foo"]

// @has 'foo/struct.S1.html'
// @count - '//details[@class="rustdoc-toggle top-doc"]/div[@class="docblock"]/p' \
//     1
// @has - '//details[@class="rustdoc-toggle top-doc"]/div[@class="docblock"]/p[1]' \
//     'Hello world! Goodbye! Hello again!'

#[doc = "Hello world!\n\n"]
/// Goodbye!
#[doc = "  Hello again!\n"]
pub struct S1;

// @has 'foo/struct.S2.html'
// @count - '//details[@class="rustdoc-toggle top-doc"]/div[@class="docblock"]/p' \
//     2
// @has - '//details[@class="rustdoc-toggle top-doc"]/div[@class="docblock"]/p[1]' \
//     'Hello world!'
// @has - '//details[@class="rustdoc-toggle top-doc"]/div[@class="docblock"]/p[2]' \
//     'Goodbye! Hello again!'

/// Hello world!
///
#[doc = "Goodbye!"]
/// Hello again!
pub struct S2;

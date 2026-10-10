#![crate_name = "foo"]

//@ has 'foo/type.S1.html'
//@ snapshot S1_top-doc - '//details[@class="toggle top-doc"]/div[@class="docblock"]'

#[doc = "Hello world!\n\n"]
/// Goodbye!
#[doc = "  Hello again!\n"]
pub struct S1;

//@ has 'foo/type.S2.html'
//@ snapshot S2_top-doc - '//details[@class="toggle top-doc"]/div[@class="docblock"]'

/// Hello world!
///
#[doc = "Goodbye!"]
/// Hello again!
pub struct S2;

//@ has 'foo/type.S3.html'
//@ snapshot S3_top-doc - '//details[@class="toggle top-doc"]/div[@class="docblock"]'
/** Par 1
*/ ///
/// Par 2
pub struct S3;

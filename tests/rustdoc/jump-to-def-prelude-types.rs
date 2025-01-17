// This test checks that prelude types like `Result` and `Option` still get a link generated.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/jump-to-def-prelude-types.rs.html'
// FIXME: would be nice to be able to check both the class and the href at the same time so
// we could check the text as well...
//@ has - '//a[@class="prelude-ty"]/@href' '{{channel}}/core/result/enum.Result.html'
//@ has - '//a[@class="prelude-ty"]/@href' '{{channel}}/core/option/enum.Option.html'
pub fn foo() -> Result<Option<()>, ()> { Err(()) }

// This part is to ensure that they are not linking to the actual prelude ty.
pub mod bar {
    struct Result;
    struct Option;

    //@ has - '//a[@href="#16"]' 'Result'
    pub fn bar() -> Result { Result }
    //@ has - '//a[@href="#17"]' 'Option'
    pub fn bar2() -> Option { Option }
}

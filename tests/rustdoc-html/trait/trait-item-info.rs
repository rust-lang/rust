// This is a regression test for <https://github.com/rust-lang/rust/issues/121772>.
// The goal is to ensure that the item information is always part of the `<summary>`
// if there is one.

#![crate_name = "foo"]
#![feature(staged_api)]

#![unstable(feature = "test", issue = "none")]

//@ has 'foo/trait.Foo.html'

#[stable(feature = "rust2", since = "2.2.2")]
pub trait Foo {
    //@ has - '//div[@class="methods"]/span[@class="item-info"]' 'bla'
    // Should not be in a `<details>` because there is no doc.
    #[unstable(feature = "bla", reason = "bla", issue = "111")]
    fn bla() {}

    //@ has - '//details[@class="toggle method-toggle"]/summary/span[@class="item-info"]' 'bar'
    // Should have a `<summary>` in the `<details>` containing the unstable info.
    /// doc
    #[unstable(feature = "bar", reason = "bla", issue = "222")]
    fn bar() {}
}

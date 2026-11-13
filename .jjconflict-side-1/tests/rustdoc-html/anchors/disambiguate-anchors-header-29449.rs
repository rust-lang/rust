// https://github.com/rust-lang/rust/issues/29449
#![crate_name="issue_29449"]

//@ has issue_29449/struct.Foo.html
pub struct Foo;

impl Foo {
    //@ has - '//*[@id="examples"]' 'Examples'
    //@ has - '//*[@id="examples"]/a[@href="#examples"]' '§'
    //@ has - '//*[@id="panics"]' 'Panics'
    //@ has - '//*[@id="panics"]/a[@href="#panics"]' '§'
    /// # Examples
    /// # Panics
    pub fn bar() {}

    //@ has - '//*[@id="examples-1"]' 'Examples'
    //@ has - '//*[@id="examples-1"]/a[@href="#examples-1"]' '§'
    /// # Examples
    pub fn bar_1() {}

    //@ has - '//*[@id="examples-2"]' 'Examples'
    //@ has - '//*[@id="examples-2"]/a[@href="#examples-2"]' '§'
    //@ has - '//*[@id="panics-1"]' 'Panics'
    //@ has - '//*[@id="panics-1"]/a[@href="#panics-1"]' '§'
    /// # Examples
    /// # Panics
    pub fn bar_2() {}
}

// This test ensures that footnotes ID are not duplicated across an item page.
// This is a regression test for <https://github.com/rust-lang/rust/issues/131901>.

#![crate_name = "foo"]

//@ has 'foo/struct.Foo.html'

pub struct Foo;

impl Foo {
    //@ has - '//a[@href="#fn1"]' '1'
    //@ has - '//li[@id="fn1"]' 'Hiya'
    //@ has - '//a[@href="#fn2"]' '2'
    //@ has - '//li[@id="fn2"]' 'Tiya'
    /// Link 1 [^1]
    /// Link 1.1 [^2]
    ///
    /// [^1]: Hiya
    /// [^2]: Tiya
    pub fn l1(){}

    //@ has - '//a[@href="#fn3"]' '1'
    //@ has - '//li[@id="fn3"]' 'Yiya'
    //@ has - '//a[@href="#fn4"]' '2'
    //@ has - '//li[@id="fn4"]' 'Biya'
    /// Link 2 [^1]
    /// Link 3 [^2]
    ///
    /// [^1]: Yiya
    /// [^2]: Biya
    pub fn l2() {}
}

impl Foo {
    //@ has - '//a[@href="#fn5"]' '1'
    //@ has - '//li[@id="fn5"]' 'Ciya'
    /// Link 3 [^1]
    ///
    /// [^1]: Ciya
    pub fn l3(){}
}

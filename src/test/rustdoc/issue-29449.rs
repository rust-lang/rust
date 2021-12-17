// @has issue_29449/struct.Foo.html
pub struct Foo;

impl Foo {
    // @has - '//*[@id="method.bar.examples"]//a' 'Examples'
    // @has - '//*[@id="method.bar.panics"]//a' 'Panics'
    /// # Examples
    /// # Panics
    pub fn bar() {}

    // @has - '//*[@id="method.bar_1.examples"]//a' 'Examples'
    // @has - '//*[@id="method.bar_1.examples-1"]//a' 'Examples'
    /// # Examples
    /// # Examples
    pub fn bar_1() {}

    // @has - '//*[@id="method.bar_2.examples"]//a' 'Examples'
    // @has - '//*[@id="method.bar_2.panics"]//a' 'Panics'
    /// # Examples
    /// # Panics
    pub fn bar_2() {}
}

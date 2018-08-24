// @has issue_29449/struct.Foo.html
pub struct Foo;

impl Foo {
    // @has - '//*[@id="examples"]//a' 'Examples'
    // @has - '//*[@id="panics"]//a' 'Panics'
    /// # Examples
    /// # Panics
    pub fn bar() {}

    // @has - '//*[@id="examples-1"]//a' 'Examples'
    /// # Examples
    pub fn bar_1() {}

    // @has - '//*[@id="examples-2"]//a' 'Examples'
    // @has - '//*[@id="panics-1"]//a' 'Panics'
    /// # Examples
    /// # Panics
    pub fn bar_2() {}
}

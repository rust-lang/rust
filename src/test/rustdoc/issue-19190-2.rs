use std::ops::Deref;

pub struct Bar;

impl Deref for Bar {
    type Target = String;
    fn deref(&self) -> &String { loop {} }
}

// @has issue_19190_2/struct.Bar.html
// @!has - '//*[@id="method.new"]' 'fn new() -> String'
// @has - '//*[@id="method.as_str"]' 'fn as_str(&self) -> &str'

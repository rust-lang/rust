// compile-flags: --test

#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

mod foo {
    pub fn foo() {}
    pub fn foo_bar() {}
    pub fn bar_foo() {}
    pub struct FooCake {}
    pub enum CakeFoo {}
    pub struct Foo7Bar;

    // Should not warn
    pub struct Foobar;
}

#[cfg(test)]
mod test {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

fn main() {}

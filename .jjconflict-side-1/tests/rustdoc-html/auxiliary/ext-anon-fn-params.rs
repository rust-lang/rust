//@ edition: 2015
#![expect(anonymous_parameters)]

pub trait Trait {
    fn required(Option<i32>, impl Fn(&str) -> bool);
    fn provided([i32; 2]) {}
}

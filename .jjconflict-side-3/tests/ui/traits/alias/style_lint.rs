//@ check-pass

#![feature(trait_alias)]

trait Foo = std::fmt::Display + std::fmt::Debug;
trait bar = std::fmt::Display + std::fmt::Debug; //~WARN trait alias `bar` should have an upper camel case name

fn main() {}

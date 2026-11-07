// Check "unused_lifetimes" lint on both async and sync functions
// Both cases should be diagnosed the same way.

//@ edition:2018

#![deny(unused_lifetimes)]

async fn async_wrong_without_args<'a>() {} //~ ERROR lifetime parameter `'a` never used

async fn async_wrong_1_lifetime<'a>(_: &i32) {} //~ ERROR lifetime parameter `'a` never used

async fn async_wrong_2_lifetimes<'a, 'b>(_: &'a i32, _: &i32) {} //~ ERROR lifetime parameter `'b` never used

async fn async_right_1_lifetime<'a>(_: &'a i32) {}

async fn async_right_2_lifetimes<'a, 'b>(_: &'a i32, _: &'b i32) {}

async fn async_right_trait_bound_lifetime<'a, I>(_: I)
where
    I: Iterator<Item = &'a i32>
{}

fn wrong_without_args<'a>() {} //~ ERROR lifetime parameter `'a` never used

fn wrong_1_lifetime<'a>(_: &i32) {} //~ ERROR lifetime parameter `'a` never used

fn wrong_2_lifetimes<'a, 'b>(_: &'a i32, _: &i32) {} //~ ERROR lifetime parameter `'b` never used

fn right_1_lifetime<'a>(_: &'a i32) {}

fn right_2_lifetimes<'a, 'b>(_: &'a i32, _: &'b i32) {}

fn right_trait_bound_lifetime<'a, I>(_: I)
where
    I: Iterator<Item = &'a i32>
{}


fn main() {}

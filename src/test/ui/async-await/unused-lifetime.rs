// Check "unused_lifetimes" lint on both async and sync functions

// edition:2018

#![deny(unused_lifetimes)]


// Async part with unused lifetimes
//
// Even wrong cases don't cause errors because async functions are desugared with all lifetimes
// involved in the signature. So, we cannot predict what lifetimes are unused in async function.
async fn async_wrong_without_args<'a>() {}

async fn async_wrong_1_lifetime<'a>(_: &i32) {}

async fn async_wrong_2_lifetimes<'a, 'b>(_: &'a i32, _: &i32) {}

async fn async_right_1_lifetime<'a>(_: &'a i32) {}

async fn async_right_2_lifetimes<'a, 'b>(_: &'a i32, _: &'b i32) {}

async fn async_right_trait_bound_lifetime<'a, I>(_: I)
where
    I: Iterator<Item = &'a i32>
{}


// Sync part with unused lifetimes
//
// These functions are compiled as supposed
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

//@ edition:2018
#![feature(async_closure)]
use std::future::Future;

async fn one() {}
async fn two() {}

fn fun<F: Future<Output = ()>>(f1: F, f2: F) {}
fn main() {
    fun(async {}, async {});
    //~^ ERROR mismatched types
    fun(one(), two());
    //~^ ERROR mismatched types
    fun((async || {})(), (async || {})());
    //~^ ERROR mismatched types
}

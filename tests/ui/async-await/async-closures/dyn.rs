// Test the diagnostic output (error descriptions) resulting from `dyn AsyncFn{,Mut,Once}`.
//@ edition:2018

#![feature(async_closure)]

use core::ops::{AsyncFn, AsyncFnMut, AsyncFnOnce};

// --- Explicit `dyn` ---

fn takes_async_fn(_: &dyn AsyncFn()) {}
//~^ ERROR the trait `AsyncFn` is not yet dyn-compatible

fn takes_async_fn_mut(_: &mut dyn AsyncFnMut()) {}
//~^ ERROR the trait `AsyncFnMut` is not yet dyn-compatible

fn takes_async_fn_once(_: Box<dyn AsyncFnOnce()>) {}
//~^ ERROR the trait `AsyncFnOnce` is not yet dyn-compatible

// --- Non-explicit `dyn` ---

#[allow(bare_trait_objects)]
fn takes_async_fn_implicit_dyn(_: &AsyncFn()) {}
//~^ ERROR the trait `AsyncFn` is not yet dyn-compatible

#[allow(bare_trait_objects)]
fn takes_async_fn_mut_implicit_dyn(_: &mut AsyncFnMut()) {}
//~^ ERROR the trait `AsyncFnMut` is not yet dyn-compatible

#[allow(bare_trait_objects)]
fn takes_async_fn_once_implicit_dyn(_: Box<AsyncFnOnce()>) {}
//~^ ERROR the trait `AsyncFnOnce` is not yet dyn-compatible

// --- Supertrait ---

trait SubAsyncFn: AsyncFn() {}
fn takes_sub_async_fn(_: &dyn SubAsyncFn) {}
//~^ ERROR the trait `SubAsyncFn` cannot be made into an object

fn main() {}

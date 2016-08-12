// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

fn arguments(_: impl Fn(),
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types
             _: Vec<impl Clone>) {}
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

type Factory<R> = impl Fn() -> R;
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

type GlobalFactory<R> = fn() -> impl FnOnce() -> R;
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

trait LazyToString {
    fn lazy_to_string<'a>(&'a self) -> impl Fn() -> String;
    //~^ ERROR `impl Trait` not allowed outside of function and inherent method return types
}

// Note that the following impl doesn't error, because the trait is invalid.
impl LazyToString for String {
    fn lazy_to_string<'a>(&'a self) -> impl Fn() -> String {
        || self.clone()
    }
}

#[derive(Copy, Clone)]
struct Lazy<T>(T);

impl std::ops::Add<Lazy<i32>> for Lazy<i32> {
    type Output = impl Fn() -> Lazy<i32>;
    //~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

    fn add(self, other: Lazy<i32>) -> Self::Output {
        move || Lazy(self.0 + other.0)
    }
}

impl<F> std::ops::Add<F>
for impl Fn() -> Lazy<i32>
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types
where F: Fn() -> impl FnOnce() -> i32
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types
{
    type Output = Self;

    fn add(self, other: F) -> Self::Output {
        move || Lazy(self().0 + other()())
    }
}

fn main() {}

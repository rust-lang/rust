// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct X<#[cfg(none)] 'a, #[cfg(none)] T>(&'a T);
//~^ ERROR #[cfg] cannot be applied on a generic parameter
//~^^ ERROR #[cfg] cannot be applied on a generic parameter

impl<#[cfg(none)] 'a, #[cfg(none)] T> X<'a, T> {}
//~^ ERROR #[cfg] cannot be applied on a generic parameter
//~^^ ERROR #[cfg] cannot be applied on a generic parameter

pub fn f<#[cfg(none)] 'a, #[cfg(none)] T>(_: &'a T) {}
//~^ ERROR #[cfg] cannot be applied on a generic parameter
//~^^ ERROR #[cfg] cannot be applied on a generic parameter

#[cfg(none)]
pub struct Y<#[cfg(none)] T>(T); // shouldn't care when the entire item is stripped out

struct M<T>(*const T);

unsafe impl<#[cfg_attr(none, may_dangle)] T> Drop for M<T> {
    //~^ ERROR #[cfg_attr] cannot be applied on a generic parameter
    fn drop(&mut self) {}
}

type Z<#[ignored] 'a, #[cfg(none)] T> = X<'a, T>;
//~^ ERROR #[cfg] cannot be applied on a generic parameter

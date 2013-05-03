// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Foo { fn f(&self) -> int; }
pub trait Bar { fn g(&self) -> int; }
pub trait Baz { fn h(&self) -> int; }

pub struct A { x: int }

impl Foo for A { fn f(&self) -> int { 10 } }
impl Bar for A { fn g(&self) -> int { 20 } }
impl Baz for A { fn h(&self) -> int { 30 } }

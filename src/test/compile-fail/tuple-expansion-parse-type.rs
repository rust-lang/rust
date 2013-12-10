// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type A = (
    ..(u8, i8),
    ..~(u8, i8), //~ ERROR expected tuple or path after `..`
    ..*(u8, i8), //~ ERROR expected tuple or path after `..`
    ..&(u8, i8), //~ ERROR expected tuple or path after `..`
    ..[int], //~ ERROR expected tuple or path after `..`
    ..[int, ..5], //~ ERROR expected tuple or path after `..`
    ..fn(u8, i8) -> int, //~ ERROR expected tuple or path after `..`
    ..|u8, i8| -> int, //~ ERROR expected tuple or path after `..`
    ..proc(u8, i8) -> int //~ ERROR expected tuple or path after `..`
);

type B<T> = ..T; //~ ERROR expected type, found token DOTDOT

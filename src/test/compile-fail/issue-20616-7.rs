// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We need all these 9 issue-20616-N.rs files
// because we can only catch one parsing error at a time



type Type_1_<'a, T> = &'a T;


//type Type_1<'a T> = &'a T; // error: expected `,` or `>` after lifetime name, found `T`


//type Type_2 = Type_1_<'static ()>; // error: expected `,` or `>` after lifetime name, found `(`


//type Type_3<T> = Box<T,,>; // error: expected type, found `,`


//type Type_4<T> = Type_1_<'static,, T>; // error: expected type, found `,`


type Type_5_<'a> = Type_1_<'a, ()>;


//type Type_5<'a> = Type_1_<'a, (),,>; // error: expected type, found `,`


//type Type_6 = Type_5_<'a,,>; // error: expected type, found `,`


type Type_7 = Box<(),,>; //~ error: expected one of `>`, identifier, lifetime, or type, found `,`


//type Type_8<'a,,> = &'a (); // error: expected ident, found `,`


//type Type_9<T,,> = Box<T>; // error: expected ident, found `,`

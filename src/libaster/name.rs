// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::parse::token;

//////////////////////////////////////////////////////////////////////////////

pub trait ToName {
    fn to_name(&self) -> ast::Name;
}

impl ToName for ast::Name {
    fn to_name(&self) -> ast::Name {
        *self
    }
}

impl<'a> ToName for &'a str {
    fn to_name(&self) -> ast::Name {
        token::intern(*self)
    }
}

impl<'a, T> ToName for &'a T where T: ToName {
    fn to_name(&self) -> ast::Name {
        (**self).to_name()
    }
}

impl<'a, T> ToName for &'a mut T where T: ToName {
    fn to_name(&self) -> ast::Name {
        (**self).to_name()
    }
}

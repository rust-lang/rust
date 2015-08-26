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

use name::ToName;

//////////////////////////////////////////////////////////////////////////////

pub trait ToIdent {
    fn to_ident(&self) -> ast::Ident;
}

impl ToIdent for ast::Ident {
    fn to_ident(&self) -> ast::Ident {
        *self
    }
}

impl ToIdent for ast::Name {
    fn to_ident(&self) -> ast::Ident {
        ast::Ident::new(*self)
    }
}

impl<'a> ToIdent for &'a str {
    fn to_ident(&self) -> ast::Ident {
        self.to_name().to_ident()
    }
}

impl ToIdent for String {
    fn to_ident(&self) -> ast::Ident {
        (&**self).to_ident()
    }
}

impl<'a, T> ToIdent for &'a T where T: ToIdent {
    fn to_ident(&self) -> ast::Ident {
        (**self).to_ident()
    }
}

impl<'a, T> ToIdent for &'a mut T where T: ToIdent {
    fn to_ident(&self) -> ast::Ident {
        (**self).to_ident()
    }
}

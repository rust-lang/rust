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

pub use expr::ExprBuilder;
pub use ident::ToIdent;
pub use name::ToName;
pub use path::PathBuilder;

//////////////////////////////////////////////////////////////////////////////

pub trait ToInternedString {
    fn to_interned_string(&self) -> token::InternedString;
}

impl ToInternedString for token::InternedString {
    fn to_interned_string(&self) -> token::InternedString {
        self.clone()
    }
}

impl<'a> ToInternedString for &'a str {
    fn to_interned_string(&self) -> token::InternedString {
        token::intern_and_get_ident(self)
    }
}

impl ToInternedString for ast::Ident {
    fn to_interned_string(&self) -> token::InternedString {
        self.name.as_str()
    }
}

impl ToInternedString for ast::Name {
    fn to_interned_string(&self) -> token::InternedString {
        self.as_str()
    }
}

impl<'a, T> ToInternedString for &'a T where T: ToInternedString {
    fn to_interned_string(&self) -> token::InternedString {
        (**self).to_interned_string()
    }
}

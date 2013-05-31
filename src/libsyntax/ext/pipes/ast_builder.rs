// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Functions for building ASTs, without having to fuss with spans.
//
// To start with, it will be use dummy spans, but it might someday do
// something smarter.

use core::prelude::*;

use ast::ident;
use ast;
use codemap::span;

use core::str;
use core::vec;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

pub fn path(ids: ~[ident], span: span) -> @ast::Path {
    @ast::Path { span: span,
                 global: false,
                 idents: ids,
                 rp: None,
                 types: ~[] }
}

pub fn path_global(ids: ~[ident], span: span) -> @ast::Path {
    @ast::Path { span: span,
                 global: true,
                 idents: ids,
                 rp: None,
                 types: ~[] }
}

pub trait append_types {
    fn add_ty(&self, ty: @ast::Ty) -> @ast::Path;
    fn add_tys(&self, tys: ~[@ast::Ty]) -> @ast::Path;
}

impl append_types for @ast::Path {
    fn add_ty(&self, ty: @ast::Ty) -> @ast::Path {
        @ast::Path {
            types: vec::append_one(copy self.types, ty),
            .. copy **self
        }
    }

    fn add_tys(&self, tys: ~[@ast::Ty]) -> @ast::Path {
        @ast::Path {
            types: vec::append(copy self.types, tys),
            .. copy **self
        }
    }
}


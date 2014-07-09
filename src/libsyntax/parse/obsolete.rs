// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Support for parsing unsupported, old syntaxes, for the
purpose of reporting errors. Parsing of these syntaxes
is tested by compile-test/obsolete-syntax.rs.

Obsolete syntax that becomes too hard to parse can be
removed.
*/

use ast::{Expr, ExprLit, LitNil};
use codemap::{Span, respan};
use parse::parser;
use parse::token;

use std::gc::{Gc, GC};

/// The specific types of unsupported syntax
#[deriving(PartialEq, Eq, Hash)]
pub enum ObsoleteSyntax {
    ObsoleteOwnedType,
    ObsoleteOwnedExpr,
    ObsoleteOwnedPattern,
    ObsoleteOwnedVector,
    ObsoleteManagedType,
    ObsoleteManagedExpr,
}

pub trait ParserObsoleteMethods {
    /// Reports an obsolete syntax non-fatal error.
    fn obsolete(&mut self, sp: Span, kind: ObsoleteSyntax);
    /// Reports an obsolete syntax non-fatal error, and returns
    /// a placeholder expression
    fn obsolete_expr(&mut self, sp: Span, kind: ObsoleteSyntax) -> Gc<Expr>;
    fn report(&mut self,
              sp: Span,
              kind: ObsoleteSyntax,
              kind_str: &str,
              desc: &str);
    fn is_obsolete_ident(&mut self, ident: &str) -> bool;
    fn eat_obsolete_ident(&mut self, ident: &str) -> bool;
}

impl<'a> ParserObsoleteMethods for parser::Parser<'a> {
    /// Reports an obsolete syntax non-fatal error.
    fn obsolete(&mut self, sp: Span, kind: ObsoleteSyntax) {
        let (kind_str, desc) = match kind {
            ObsoleteOwnedType => (
                "`~` notation for owned pointers",
                "use `Box<T>` in `std::owned` instead"
            ),
            ObsoleteOwnedExpr => (
                "`~` notation for owned pointer allocation",
                "use the `box` operator instead of `~`"
            ),
            ObsoleteOwnedPattern => (
                "`~` notation for owned pointer patterns",
                "use the `box` operator instead of `~`"
            ),
            ObsoleteOwnedVector => (
                "`~[T]` is no longer a type",
                "use the `Vec` type instead"
            ),
            ObsoleteManagedType => (
                "`@` notation for managed pointers",
                "use `Gc<T>` in `std::gc` instead"
            ),
            ObsoleteManagedExpr => (
                "`@` notation for a managed pointer allocation",
                "use the `box(GC)` oeprator instead of `@`"
            ),
        };

        self.report(sp, kind, kind_str, desc);
    }

    /// Reports an obsolete syntax non-fatal error, and returns
    /// a placeholder expression
    fn obsolete_expr(&mut self, sp: Span, kind: ObsoleteSyntax) -> Gc<Expr> {
        self.obsolete(sp, kind);
        self.mk_expr(sp.lo, sp.hi, ExprLit(box(GC) respan(sp, LitNil)))
    }

    fn report(&mut self,
              sp: Span,
              kind: ObsoleteSyntax,
              kind_str: &str,
              desc: &str) {
        self.span_err(sp,
                      format!("obsolete syntax: {}", kind_str).as_slice());

        if !self.obsolete_set.contains(&kind) {
            self.sess
                .span_diagnostic
                .handler()
                .note(format!("{}", desc).as_slice());
            self.obsolete_set.insert(kind);
        }
    }

    fn is_obsolete_ident(&mut self, ident: &str) -> bool {
        match self.token {
            token::IDENT(sid, _) => {
                token::get_ident(sid).equiv(&ident)
            }
            _ => false
        }
    }

    fn eat_obsolete_ident(&mut self, ident: &str) -> bool {
        if self.is_obsolete_ident(ident) {
            self.bump();
            true
        } else {
            false
        }
    }
}

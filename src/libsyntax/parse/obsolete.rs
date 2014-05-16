// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use parse::parser::Parser;
use parse::token;

/// The specific types of unsupported syntax
#[deriving(Eq, TotalEq, Hash)]
pub enum ObsoleteSyntax {
    ObsoleteSwap,
    ObsoleteUnsafeBlock,
    ObsoleteBareFnType,
    ObsoleteMultipleLocalDecl,
    ObsoleteUnsafeExternFn,
    ObsoleteTraitFuncVisibility,
    ObsoleteConstPointer,
    ObsoleteLoopAsContinue,
    ObsoleteEnumWildcard,
    ObsoleteStructWildcard,
    ObsoleteVecDotDotWildcard,
    ObsoleteMultipleImport,
    ObsoleteManagedPattern,
    ObsoleteManagedString,
    ObsoleteManagedVec,
    ObsoleteOwnedType,
    ObsoleteOwnedExpr,
    ObsoleteOwnedPattern,
}

pub trait ParserObsoleteMethods {
    /// Reports an obsolete syntax non-fatal error.
    fn obsolete(&mut self, sp: Span, kind: ObsoleteSyntax);
    // Reports an obsolete syntax non-fatal error, and returns
    // a placeholder expression
    fn obsolete_expr(&mut self, sp: Span, kind: ObsoleteSyntax) -> @Expr;
    fn report(&mut self,
              sp: Span,
              kind: ObsoleteSyntax,
              kind_str: &str,
              desc: &str);
    fn is_obsolete_ident(&mut self, ident: &str) -> bool;
    fn eat_obsolete_ident(&mut self, ident: &str) -> bool;
}

impl<'a> ParserObsoleteMethods for Parser<'a> {
    /// Reports an obsolete syntax non-fatal error.
    fn obsolete(&mut self, sp: Span, kind: ObsoleteSyntax) {
        let (kind_str, desc) = match kind {
            ObsoleteSwap => (
                "swap",
                "use std::mem::{swap, replace} instead"
            ),
            ObsoleteUnsafeBlock => (
                "non-standalone unsafe block",
                "use an inner `unsafe { ... }` block instead"
            ),
            ObsoleteBareFnType => (
                "bare function type",
                "use `|A| -> B` or `extern fn(A) -> B` instead"
            ),
            ObsoleteMultipleLocalDecl => (
                "declaration of multiple locals at once",
                "instead of e.g. `let a = 1, b = 2`, write \
                 `let (a, b) = (1, 2)`."
            ),
            ObsoleteUnsafeExternFn => (
                "unsafe external function",
                "external functions are always unsafe; remove the `unsafe` \
                 keyword"
            ),
            ObsoleteTraitFuncVisibility => (
                "visibility not necessary",
                "trait functions inherit the visibility of the trait itself"
            ),
            ObsoleteConstPointer => (
                "const pointer",
                "instead of `&const Foo` or `@const Foo`, write `&Foo` or \
                 `@Foo`"
            ),
            ObsoleteLoopAsContinue => (
                "`loop` instead of `continue`",
                "`loop` is now only used for loops and `continue` is used for \
                 skipping iterations"
            ),
            ObsoleteEnumWildcard => (
                "enum wildcard",
                "use `..` instead of `*` for matching all enum fields"
            ),
            ObsoleteStructWildcard => (
                "struct wildcard",
                "use `..` instead of `_` for matching trailing struct fields"
            ),
            ObsoleteVecDotDotWildcard => (
                "vec slice wildcard",
                "use `..` instead of `.._` for matching slices"
            ),
            ObsoleteMultipleImport => (
                "multiple imports",
                "only one import is allowed per `use` statement"
            ),
            ObsoleteManagedPattern => (
                "managed pointer pattern",
                "use a nested `match` expression instead of a managed box \
                 pattern"
            ),
            ObsoleteManagedString => (
                "managed string",
                "use `Rc<StrBuf>` instead of a managed string"
            ),
            ObsoleteManagedVec => (
                "managed vector",
                "use `Rc<~[T]>` instead of a managed vector"
            ),
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
        };

        self.report(sp, kind, kind_str, desc);
    }

    // Reports an obsolete syntax non-fatal error, and returns
    // a placeholder expression
    fn obsolete_expr(&mut self, sp: Span, kind: ObsoleteSyntax) -> @Expr {
        self.obsolete(sp, kind);
        self.mk_expr(sp.lo, sp.hi, ExprLit(@respan(sp, LitNil)))
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

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

use ast::{expr, expr_lit, lit_nil, attribute};
use ast;
use codemap::{span, respan};
use parse::parser::Parser;
use parse::token::{keywords, Token};
use parse::token;

use std::str;
use std::to_bytes;

/// The specific types of unsupported syntax
#[deriving(Eq)]
pub enum ObsoleteSyntax {
    ObsoleteLowerCaseKindBounds,
    ObsoleteLet,
    ObsoleteFieldTerminator,
    ObsoleteStructCtor,
    ObsoleteWith,
    ObsoleteClassTraits,
    ObsoletePrivSection,
    ObsoleteModeInFnType,
    ObsoleteMoveInit,
    ObsoleteBinaryMove,
    ObsoleteSwap,
    ObsoleteUnsafeBlock,
    ObsoleteUnenforcedBound,
    ObsoleteImplSyntax,
    ObsoleteMutOwnedPointer,
    ObsoleteMutVector,
    ObsoleteImplVisibility,
    ObsoleteRecordType,
    ObsoleteRecordPattern,
    ObsoletePostFnTySigil,
    ObsoleteBareFnType,
    ObsoleteNewtypeEnum,
    ObsoleteMode,
    ObsoleteImplicitSelf,
    ObsoleteLifetimeNotation,
    ObsoleteConstManagedPointer,
    ObsoletePurity,
    ObsoleteStaticMethod,
    ObsoleteConstItem,
    ObsoleteFixedLengthVectorType,
    ObsoleteNamedExternModule,
    ObsoleteMultipleLocalDecl,
    ObsoleteMutWithMultipleBindings,
}

impl to_bytes::IterBytes for ObsoleteSyntax {
    #[inline]
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        (*self as uint).iter_bytes(lsb0, f)
    }
}

pub trait ParserObsoleteMethods {
    /// Reports an obsolete syntax non-fatal error.
    fn obsolete(&self, sp: span, kind: ObsoleteSyntax);
    // Reports an obsolete syntax non-fatal error, and returns
    // a placeholder expression
    fn obsolete_expr(&self, sp: span, kind: ObsoleteSyntax) -> @expr;
    fn report(&self,
              sp: span,
              kind: ObsoleteSyntax,
              kind_str: &str,
              desc: &str);
    fn token_is_obsolete_ident(&self, ident: &str, token: &Token) -> bool;
    fn is_obsolete_ident(&self, ident: &str) -> bool;
    fn eat_obsolete_ident(&self, ident: &str) -> bool;
    fn try_parse_obsolete_struct_ctor(&self) -> bool;
    fn try_parse_obsolete_with(&self) -> bool;
    fn try_parse_obsolete_priv_section(&self, attrs: &[attribute]) -> bool;
}

impl ParserObsoleteMethods for Parser {
    /// Reports an obsolete syntax non-fatal error.
    pub fn obsolete(&self, sp: span, kind: ObsoleteSyntax) {
        let (kind_str, desc) = match kind {
            ObsoleteLowerCaseKindBounds => (
                "lower-case kind bounds",
                "the `send`, `copy`, `const`, and `owned` \
                 kinds are represented as traits now, and \
                 should be camel cased"
            ),
            ObsoleteLet => (
                "`let` in field declaration",
                "declare fields as `field: Type`"
            ),
            ObsoleteFieldTerminator => (
                "field declaration terminated with semicolon",
                "fields are now separated by commas"
            ),
            ObsoleteStructCtor => (
                "struct constructor",
                "structs are now constructed with `MyStruct { foo: val }` \
                 syntax. Structs with private fields cannot be created \
                 outside of their defining module"
            ),
            ObsoleteWith => (
                "with",
                "record update is done with `..`, e.g. \
                 `MyStruct { foo: bar, .. baz }`"
            ),
            ObsoleteClassTraits => (
                "class traits",
                "implemented traits are specified on the impl, as in \
                 `impl foo : bar {`"
            ),
            ObsoletePrivSection => (
                "private section",
                "the `priv` keyword is applied to individual items, methods, \
                 and fields"
            ),
            ObsoleteModeInFnType => (
                "mode without identifier in fn type",
                "to use a (deprecated) mode in a fn type, you should \
                 give the argument an explicit name (like `&&v: int`)"
            ),
            ObsoleteMoveInit => (
                "initializer-by-move",
                "Write `let foo = move bar` instead"
            ),
            ObsoleteBinaryMove => (
                "binary move",
                "Write `foo = move bar` instead"
            ),
            ObsoleteSwap => (
                "swap",
                "Use std::util::{swap, replace} instead"
            ),
            ObsoleteUnsafeBlock => (
                "non-standalone unsafe block",
                "use an inner `unsafe { ... }` block instead"
            ),
            ObsoleteUnenforcedBound => (
                "unenforced type parameter bound",
                "use trait bounds on the functions that take the type as \
                 arguments, not on the types themselves"
            ),
            ObsoleteImplSyntax => (
                "colon-separated impl syntax",
                "write `impl Trait for Type`"
            ),
            ObsoleteMutOwnedPointer => (
                "const or mutable owned pointer",
                "mutability inherits through `~` pointers; place the `~` box
                 in a mutable location, like a mutable local variable or an \
                 `@mut` box"
            ),
            ObsoleteMutVector => (
                "const or mutable vector",
                "mutability inherits through `~` pointers; place the vector \
                 in a mutable location, like a mutable local variable or an \
                 `@mut` box"
            ),
            ObsoleteImplVisibility => (
                "visibility-qualified implementation",
                "`pub` or `priv` goes on individual functions; remove the \
                 `pub` or `priv`"
            ),
            ObsoleteRecordType => (
                "structural record type",
                "use a structure instead"
            ),
            ObsoleteRecordPattern => (
                "structural record pattern",
                "use a structure instead"
            ),
            ObsoletePostFnTySigil => (
                "fn sigil in postfix position",
                "Rather than `fn@`, `fn~`, or `fn&`, \
                 write `@fn`, `~fn`, and `&fn` respectively"
            ),
            ObsoleteBareFnType => (
                "bare function type",
                "use `&fn` or `extern fn` instead"
            ),
            ObsoleteNewtypeEnum => (
                "newtype enum",
                "instead of `enum Foo = int`, write `struct Foo(int)`"
            ),
            ObsoleteMode => (
                "obsolete argument mode",
                "replace `-` or `++` mode with `+`"
            ),
            ObsoleteImplicitSelf => (
                "implicit self",
                "use an explicit `self` declaration or declare the method as \
                 static"
            ),
            ObsoleteLifetimeNotation => (
                "`/` lifetime notation",
                "instead of `&foo/bar`, write `&'foo bar`; instead of \
                 `bar/&foo`, write `&bar<'foo>"
            ),
            ObsoleteConstManagedPointer => (
                "const `@` pointer",
                "instead of `@const Foo`, write `@Foo`"
            ),
            ObsoletePurity => (
                "pure function",
                "remove `pure`"
            ),
            ObsoleteStaticMethod => (
                "`static` notation",
                "`static` is superfluous; remove it"
            ),
            ObsoleteConstItem => (
                "`const` item",
                "`const` items are now `static` items; replace `const` with \
                 `static`"
            ),
            ObsoleteFixedLengthVectorType => (
                "fixed-length vector notation",
                "instead of `[T * N]`, write `[T, ..N]`"
            ),
            ObsoleteNamedExternModule => (
                "named external module",
                "instead of `extern mod foo { ... }`, write `mod foo { \
                 extern { ... } }`"
            ),
            ObsoleteMultipleLocalDecl => (
                "declaration of multiple locals at once",
                "instead of e.g. `let a = 1, b = 2`, write \
                 `let (a, b) = (1, 2)`."
            ),
            ObsoleteMutWithMultipleBindings => (
                "`mut` with multiple bindings",
                "use multiple local declarations instead of e.g. `let mut \
                 (x, y) = ...`."
            ),
        };

        self.report(sp, kind, kind_str, desc);
    }

    // Reports an obsolete syntax non-fatal error, and returns
    // a placeholder expression
    pub fn obsolete_expr(&self, sp: span, kind: ObsoleteSyntax) -> @expr {
        self.obsolete(sp, kind);
        self.mk_expr(sp.lo, sp.hi, expr_lit(@respan(sp, lit_nil)))
    }

    fn report(&self,
              sp: span,
              kind: ObsoleteSyntax,
              kind_str: &str,
              desc: &str) {
        self.span_err(sp, fmt!("obsolete syntax: %s", kind_str));

        if !self.obsolete_set.contains(&kind) {
            self.sess.span_diagnostic.handler().note(fmt!("%s", desc));
            self.obsolete_set.insert(kind);
        }
    }

    pub fn token_is_obsolete_ident(&self, ident: &str, token: &Token)
                                   -> bool {
        match *token {
            token::IDENT(sid, _) => {
                str::eq_slice(self.id_to_str(sid), ident)
            }
            _ => false
        }
    }

    pub fn is_obsolete_ident(&self, ident: &str) -> bool {
        self.token_is_obsolete_ident(ident, self.token)
    }

    pub fn eat_obsolete_ident(&self, ident: &str) -> bool {
        if self.is_obsolete_ident(ident) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub fn try_parse_obsolete_struct_ctor(&self) -> bool {
        if self.eat_obsolete_ident("new") {
            self.obsolete(*self.last_span, ObsoleteStructCtor);
            self.parse_fn_decl();
            self.parse_block();
            true
        } else {
            false
        }
    }

    pub fn try_parse_obsolete_with(&self) -> bool {
        if *self.token == token::COMMA
            && self.look_ahead(1,
                               |t| self.token_is_obsolete_ident("with", t)) {
            self.bump();
        }
        if self.eat_obsolete_ident("with") {
            self.obsolete(*self.last_span, ObsoleteWith);
            self.parse_expr();
            true
        } else {
            false
        }
    }

    pub fn try_parse_obsolete_priv_section(&self, attrs: &[attribute])
                                           -> bool {
        if self.is_keyword(keywords::Priv) &&
                self.look_ahead(1, |t| *t == token::LBRACE) {
            self.obsolete(*self.span, ObsoletePrivSection);
            self.eat_keyword(keywords::Priv);
            self.bump();
            while *self.token != token::RBRACE {
                self.parse_single_struct_field(ast::private, attrs.to_owned());
            }
            self.bump();
            true
        } else {
            false
        }
    }

}

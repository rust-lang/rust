/*!
Support for parsing unsupported, old syntaxes, for the
purpose of reporting errors. Parsing of these syntaxes
is tested by compile-test/obsolete-syntax.rs.

Obsolete syntax that becomes too hard to parse can be
removed.
*/

use codemap::span;
use ast::{expr, expr_lit, lit_nil};
use ast_util::{respan};
use token::Token;

/// The specific types of unsupported syntax
pub enum ObsoleteSyntax {
    ObsoleteLowerCaseKindBounds,
    ObsoleteLet,
    ObsoleteFieldTerminator,
    ObsoleteStructCtor,
    ObsoleteWith,
    ObsoleteClassMethod,
    ObsoleteClassTraits,
    ObsoletePrivSection,
    ObsoleteModeInFnType,
    ObsoleteMoveInit,
    ObsoleteBinaryMove
}

impl ObsoleteSyntax : cmp::Eq {
    pure fn eq(&self, other: &ObsoleteSyntax) -> bool {
        (*self) as uint == (*other) as uint
    }
    pure fn ne(&self, other: &ObsoleteSyntax) -> bool {
        !(*self).eq(other)
    }
}

#[cfg(stage0)]
impl ObsoleteSyntax: to_bytes::IterBytes {
    #[inline(always)]
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (self as uint).iter_bytes(lsb0, f);
    }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl ObsoleteSyntax: to_bytes::IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as uint).iter_bytes(lsb0, f);
    }
}

impl Parser {
    /// Reports an obsolete syntax non-fatal error.
    fn obsolete(sp: span, kind: ObsoleteSyntax) {
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
            ObsoleteClassMethod => (
                "class method",
                "methods should be defined inside impls"
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
            )
        };

        self.report(sp, kind, kind_str, desc);
    }

    // Reports an obsolete syntax non-fatal error, and returns
    // a placeholder expression
    fn obsolete_expr(sp: span, kind: ObsoleteSyntax) -> @expr {
        self.obsolete(sp, kind);
        self.mk_expr(sp.lo, sp.hi, expr_lit(@respan(sp, lit_nil)))
    }

    priv fn report(sp: span, kind: ObsoleteSyntax, kind_str: &str,
                   desc: &str) {
        self.span_err(sp, fmt!("obsolete syntax: %s", kind_str));

        if !self.obsolete_set.contains_key(kind) {
            self.sess.span_diagnostic.handler().note(fmt!("%s", desc));
            self.obsolete_set.insert(kind, ());
        }
    }

    fn token_is_obsolete_ident(ident: &str, token: Token) -> bool {
        match token {
            token::IDENT(copy sid, _) => {
                str::eq_slice(*self.id_to_str(sid), ident)
            }
            _ => false
        }
    }

    fn is_obsolete_ident(ident: &str) -> bool {
        self.token_is_obsolete_ident(ident, copy self.token)
    }

    fn eat_obsolete_ident(ident: &str) -> bool {
        if self.is_obsolete_ident(ident) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn try_parse_obsolete_struct_ctor() -> bool {
        if self.eat_obsolete_ident("new") {
            self.obsolete(copy self.last_span, ObsoleteStructCtor);
            self.parse_fn_decl(|p| p.parse_arg());
            self.parse_block();
            true
        } else {
            false
        }
    }

    fn try_parse_obsolete_with() -> bool {
        if self.token == token::COMMA
            && self.token_is_obsolete_ident("with",
                                            self.look_ahead(1u)) {
            self.bump();
        }
        if self.eat_obsolete_ident("with") {
            self.obsolete(copy self.last_span, ObsoleteWith);
            self.parse_expr();
            true
        } else {
            false
        }
    }

    fn try_parse_obsolete_priv_section() -> bool {
        if self.is_keyword(~"priv") && self.look_ahead(1) == token::LBRACE {
            self.obsolete(copy self.span, ObsoletePrivSection);
            self.eat_keyword(~"priv");
            self.bump();
            while self.token != token::RBRACE {
                self.parse_single_class_item(ast::private);
            }
            self.bump();
            true
        } else {
            false
        }
    }

}


use super::parser::{Parser, TokenSet};
use SyntaxKind;
use syntax_kinds::*;

mod items;
mod attributes;
mod expressions;
mod types;
mod paths;

pub(crate) fn file(p: &mut Parser) {
    let file = p.start();
    p.eat(SHEBANG);
    items::mod_contents(p, false);
    file.complete(p, FILE);
}

fn visibility(p: &mut Parser) {
    if p.at(PUB_KW) {
        let vis = p.start();
        p.bump();
        if p.at(L_PAREN) {
            match p.nth(1) {
                CRATE_KW | SELF_KW | SUPER_KW => {
                    p.bump();
                    p.bump();
                    p.expect(R_PAREN);
                }
                IN_KW => {
                    p.bump();
                    p.bump();
                    paths::use_path(p);
                    p.expect(R_PAREN);
                }
                _ => (),
            }
        }
        vis.complete(p, VISIBILITY);
    }
}

fn alias(p: &mut Parser) -> bool {
    if p.at(AS_KW) {
        let alias = p.start();
        p.bump();
        p.expect(IDENT);
        alias.complete(p, ALIAS);
    }
    true //FIXME: return false if three are errors
}

fn error_block(p: &mut Parser, message: &str) {
    assert!(p.at(L_CURLY));
    let err = p.start();
    p.error().message(message).emit();
    p.bump();
    let mut level: u32 = 1;
    while level > 0 && !p.at(EOF) {
        match p.current() {
            L_CURLY => level += 1,
            R_CURLY => level -= 1,
            _ => (),
        }
        p.bump();
    }
    err.complete(p, ERROR);
}

impl<'p> Parser<'p> {
    fn at<L: Lookahead>(&self, l: L) -> bool {
        l.is_ahead(self)
    }

    fn err_and_bump(&mut self, message: &str) {
        let err = self.start();
        self.error().message(message).emit();
        self.bump();
        err.complete(self, ERROR);
    }

    fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            self.error().message(format!("expected {:?}", kind)).emit();
            false
        }
    }

    fn eat(&mut self, kind: SyntaxKind) -> bool {
        self.current() == kind && {
            self.bump();
            true
        }
    }
}

trait Lookahead: Copy {
    fn is_ahead(self, p: &Parser) -> bool;
}

impl Lookahead for SyntaxKind {
    fn is_ahead(self, p: &Parser) -> bool {
        p.current() == self
    }
}

impl Lookahead for [SyntaxKind; 2] {
    fn is_ahead(self, p: &Parser) -> bool {
        p.current() == self[0] && p.nth(1) == self[1]
    }
}

impl Lookahead for [SyntaxKind; 3] {
    fn is_ahead(self, p: &Parser) -> bool {
        p.current() == self[0] && p.nth(1) == self[1] && p.nth(2) == self[2]
    }
}

#[derive(Clone, Copy)]
struct AnyOf<'a>(&'a [SyntaxKind]);

impl<'a> Lookahead for AnyOf<'a> {
    fn is_ahead(self, p: &Parser) -> bool {
        let curr = p.current();
        self.0.iter().any(|&k| k == curr)
    }
}

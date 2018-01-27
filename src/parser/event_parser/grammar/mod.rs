use super::parser::{Parser, TokenSet};
use {SyntaxKind};
use tree::EOF;
use syntax_kinds::*;

mod items;
mod attributes;
mod expressions;
mod types;
mod paths;

pub(crate) fn file(p: &mut Parser) {
    let file = p.start();
    p.eat(SHEBANG);
    items::mod_contents(p);
    file.complete(p, FILE);
}

fn visibility(p: &mut Parser) {
    if p.at(PUB_KW) {
        let vis = p.start();
        p.bump();
        if p.at(L_PAREN) {
            match p.raw_lookahead(1) {
                CRATE_KW | SELF_KW | SUPER_KW | IN_KW => {
                    p.bump();
                    if p.bump() == IN_KW {
                        paths::use_path(p);
                    }
                    p.expect(R_PAREN);
                }
                _ => ()
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

fn repeat<F: FnMut(&mut Parser) -> bool>(p: &mut Parser, mut f: F) {
    loop {
        let pos = p.pos();
        if !f(p) {
            return
        }
        if pos == p.pos() {
            panic!("Infinite loop in parser")
        }
    }
}

fn comma_list<F: Fn(&mut Parser) -> bool>(p: &mut Parser, end: SyntaxKind, f: F) {
    repeat(p, |p| {
        if p.current() == end {
            return false
        }
        let pos = p.pos();
        f(p);
        if p.pos() == pos {
            return false
        }

        if p.current() == end {
            p.eat(COMMA);
        } else {
            p.expect(COMMA);
        }
         true
    })
}


impl<'p> Parser<'p> {
    fn at<L: Lookahead>(&self, l: L) -> bool {
        l.is_ahead(self)
    }

    fn err_and_bump(&mut self, message: &str) {
        let err = self.start();
        self.error()
            .message(message)
            .emit();
        self.bump();
        err.complete(self, ERROR);
    }

    pub(crate) fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            self.error()
                .message(format!("expected {:?}", kind))
                .emit();
            false
        }
    }

    fn eat(&mut self, kind: SyntaxKind) -> bool {
        self.current() == kind && { self.bump(); true }
    }
}

trait Lookahead: Copy {
    fn is_ahead(self, p: &Parser) -> bool;
    fn consume(p: &mut Parser);
}

impl Lookahead for SyntaxKind {
    fn is_ahead(self, p: &Parser) -> bool {
        p.current() == self
    }

    fn consume(p: &mut Parser) {
        p.bump();
    }
}

impl Lookahead for [SyntaxKind; 2] {
    fn is_ahead(self, p: &Parser) -> bool {
        p.current() == self[0]
        && p.raw_lookahead(1) == self[1]
    }

    fn consume(p: &mut Parser) {
        p.bump();
        p.bump();
    }
}

impl Lookahead for [SyntaxKind; 3] {
    fn is_ahead(self, p: &Parser) -> bool {
        p.current() == self[0]
        && p.raw_lookahead(1) == self[1]
        && p.raw_lookahead(2) == self[2]
    }

    fn consume(p: &mut Parser) {
        p.bump();
        p.bump();
        p.bump();
    }
}

#[derive(Clone, Copy)]
struct AnyOf<'a>(&'a [SyntaxKind]);

impl<'a> Lookahead for AnyOf<'a> {
    fn is_ahead(self, p: &Parser) -> bool {
        let curr = p.current();
        self.0.iter().any(|&k| k == curr)
    }

    fn consume(p: &mut Parser) {
        p.bump();
    }

}

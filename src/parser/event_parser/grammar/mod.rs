use super::parser::Parser;
use {SyntaxKind};
use tree::EOF;
use syntax_kinds::*;

mod items;
mod attributes;
mod expressions;

pub(crate) fn file(p: &mut Parser) {
    node(p, FILE, |p| {
        p.optional(SHEBANG);
        items::mod_contents(p);
    })
}

fn visibility(_: &mut Parser) {
}

fn alias(p: &mut Parser) -> bool {
    node_if(p, AS_KW, ALIAS, |p| {
        p.expect(IDENT);
    });
    true //FIXME: return false if three are errors
}

fn node_if<F: FnOnce(&mut Parser), L: Lookahead>(
    p: &mut Parser,
    first: L,
    node_kind: SyntaxKind,
    rest: F
) -> bool {
    first.is_ahead(p) && { node(p, node_kind, |p| { L::consume(p); rest(p); }); true }
}

fn node<F: FnOnce(&mut Parser)>(p: &mut Parser, node_kind: SyntaxKind, rest: F) {
    p.start(node_kind);
    rest(p);
    p.finish();
}

fn many<F: Fn(&mut Parser) -> bool>(p: &mut Parser, f: F) {
    while f(p) { }
}

fn comma_list<F: Fn(&mut Parser) -> bool>(p: &mut Parser, end: SyntaxKind, f: F) {
    many(p, |p| {
        if !f(p) || p.current() == end {
            false
        } else {
            p.expect(COMMA);
            true
        }
    })
}


fn skip_to_first<C, F>(p: &mut Parser, cond: C, f: F, message: &str) -> bool
where
    C: Fn(&Parser) -> bool,
    F: FnOnce(&mut Parser),
{
    let mut skipped = false;
    loop {
        if cond(p) {
            if skipped {
                p.finish();
            }
            f(p);
            return true;
        }
        if p.current() == EOF {
            if skipped {
                p.finish();
            }
            return false;
        }
        if !skipped {
            p.start(ERROR);
            p.error()
                .message(message)
                .emit();
        }
        p.bump();
        skipped = true;
    }
}

impl<'p> Parser<'p> {
    pub(crate) fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.current() == kind {
            self.bump();
            true
        } else {
            self.error()
                .message(format!("expected {:?}", kind))
                .emit();
            false
        }
    }

    fn optional(&mut self, kind: SyntaxKind) {
        if self.current() == kind {
            self.bump();
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
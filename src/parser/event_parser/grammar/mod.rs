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
        attributes::inner_attributes(p);
        items::mod_items(p);
    })
}

fn visibility(_: &mut Parser) {
}

fn node_if<F: FnOnce(&mut Parser)>(
    p: &mut Parser,
    first: SyntaxKind,
    node_kind: SyntaxKind,
    rest: F
) -> bool {
    p.current() == first && { node(p, node_kind, |p| { p.bump(); rest(p); }); true }
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

    fn bump_n(&mut self, n: u8) {
        for _ in 0..n {
            self.bump();
        }
    }

    fn eat(&mut self, kind: SyntaxKind) -> bool {
        self.current() == kind && { self.bump(); true }
    }
}
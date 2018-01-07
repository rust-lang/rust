use super::parser::Parser;
use {SyntaxKind};
use tree::EOF;
use syntax_kinds::*;

// Items //

pub(crate) fn file(p: &mut Parser) {
    node(p, FILE, |p| {
        p.optional(SHEBANG);
        inner_attributes(p);
        many(p, |p| {
            skip_to_first(
                p, item_first, item,
                "expected item",
            )
        });
    })
}

fn item_first(p: &Parser) -> bool {
    match p.current() {
        STRUCT_KW | FN_KW => true,
        _ => false,
    }
}

fn item(p: &mut Parser) {
    outer_attributes(p);
    visibility(p);
    node_if(p, STRUCT_KW, STRUCT_ITEM, struct_item)
        || node_if(p, FN_KW, FN_ITEM, fn_item);
}

fn struct_item(p: &mut Parser) {
    p.expect(IDENT)
        && p.curly_block(|p| comma_list(p, EOF, struct_field));
}

fn struct_field(p: &mut Parser) -> bool {
    node_if(p, IDENT, STRUCT_FIELD, |p| {
        p.expect(COLON) && p.expect(IDENT);
    })
}

fn fn_item(p: &mut Parser) {
    p.expect(IDENT) && p.expect(L_PAREN) && p.expect(R_PAREN)
        && p.curly_block(|p| ());
}


// Paths, types, attributes, and stuff //

fn inner_attributes(p: &mut Parser) {
    many(p, |p| attribute(p, true))
}

fn attribute(p: &mut Parser, inner: bool) -> bool {
    let attr_start = inner && p.lookahead(&[POUND, EXCL, L_BRACK])
        || !inner && p.lookahead(&[POUND, L_BRACK]);
    if !attr_start {
        return false;
    }
    node(p, ATTR, |p| {
        p.bump_n(if inner { 3 } else { 2 });
        meta_item(p) && p.expect(R_BRACK);
    });
    true
}

fn meta_item(p: &mut Parser) -> bool {
    node_if(p, IDENT, META_ITEM, |p| {
        if p.eat(EQ) {
            if !literal(p) {
                p.error()
                    .message("expected literal")
                    .emit();
            }
        } else if p.eat(L_PAREN) {
            comma_list(p, R_PAREN, meta_item_inner);
            p.expect(R_PAREN);
        }
    })
}

fn meta_item_inner(p: &mut Parser) -> bool {
    meta_item(p) || literal(p)
}

fn literal(p: &mut Parser) -> bool {
    p.eat(INT_NUMBER) || p.eat(FLOAT_NUMBER)
}

fn outer_attributes(_: &mut Parser) {
}

fn visibility(_: &mut Parser) {
}

// Expressions //

// Error recovery and high-order utils //

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
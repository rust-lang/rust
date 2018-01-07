use super::parser::Parser;
use {SyntaxKind};
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
    let current = match p.current() {
        Some(c) => c,
        None => return false,
    };
    match current {
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
        && p.curly_block(|p| comma_list(p, struct_field));
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
    many(p, inner_attribute)
}

fn inner_attribute(p: &mut Parser) -> bool {
    if !(p.lookahead(&[EXCL, POUND])) {
        return false;
    }
    node(p, ATTR, |p| {
        p.bump_n(2);
    });
    true
}

fn outer_attributes(_: &mut Parser) {
}

fn visibility(_: &mut Parser) {
}

// Expressions //

// Error recovery and high-order utils //

fn node_if<F: FnOnce(&mut Parser)>(p: &mut Parser, first: SyntaxKind, node_kind: SyntaxKind, rest: F) -> bool {
    p.current_is(first) && { node(p, node_kind, |p| { p.bump(); rest(p); }); true }
}

fn node<F: FnOnce(&mut Parser)>(p: &mut Parser, node_kind: SyntaxKind, rest: F) {
    p.start(node_kind);
    rest(p);
    p.finish();
}

fn many<F: Fn(&mut Parser) -> bool>(p: &mut Parser, f: F) {
    while f(p) { }
}

fn comma_list<F: Fn(&mut Parser) -> bool>(p: &mut Parser, f: F) {
    many(p, |p| {
        f(p);
        if p.is_eof() {
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
        if p.is_eof() {
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
        p.bump().unwrap();
        skipped = true;
    }
}

impl<'p> Parser<'p> {
    fn current_is(&self, kind: SyntaxKind) -> bool {
        self.current() == Some(kind)
    }

    pub(crate) fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.current_is(kind) {
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
        if self.current_is(kind) {
            self.bump();
        }
    }

    fn bump_n(&mut self, n: u8) {
        for _ in 0..n {
            self.bump();
        }
    }
}
use super::parser::Parser;

use syntax_kinds::*;

// Items //

pub fn file(p: &mut Parser) {
    p.start(FILE);
    shebang(p);
    inner_attributes(p);
    mod_items(p);
    p.finish();
}

type Result = ::std::result::Result<(), ()>;
const OK: Result = Ok(());
const ERR: Result = Err(());

fn shebang(_: &mut Parser) {
    //TODO
}

fn inner_attributes(_: &mut Parser) {
    //TODO
}

fn mod_items(p: &mut Parser) {
    loop {
        skip_until_item(p);
        if p.is_eof() {
            return;
        }
        if item(p).is_err() {
            skip_one_token(p);
        }
    }
}

fn item(p: &mut Parser) -> Result {
    outer_attributes(p)?;
    visibility(p)?;
    if p.current_is(STRUCT_KW) {
        p.start(STRUCT_ITEM);
        p.bump();
        let _ = struct_item(p);
        p.finish();
        return OK;
    }
    ERR
}

fn struct_item(p: &mut Parser) -> Result {
    p.expect(IDENT)?;
    p.curly_block(|p| {
        comma_list(p, struct_field)
    })
}

fn struct_field(p: &mut Parser) -> Result {
    if !p.current_is(IDENT) {
        return ERR;
    }
    p.start(STRUCT_FIELD);
    p.bump();
    ignore_errors(|| {
        p.expect(COLON)?;
        p.expect(IDENT)?;
        OK
    });
    p.finish();
    OK
}

// Paths, types, attributes, and stuff //

fn outer_attributes(_: &mut Parser) -> Result {
    OK
}

fn visibility(_: &mut Parser) -> Result {
    OK
}

// Expressions //

// Error recovery and high-order utils //

fn skip_until_item(_: &mut Parser) {
    //TODO
}

fn skip_one_token(p: &mut Parser) {
    p.start(ERROR);
    p.bump().unwrap();
    p.finish();
}

fn ignore_errors<F: FnOnce() -> Result>(f: F) {
    drop(f());
}

fn comma_list<F: Fn(&mut Parser) -> Result>(p: &mut Parser, element: F) {
    loop {
        if element(p).is_err() {
            return
        }
        if p.expect(COMMA).is_err() {
            return
        }
    }
}
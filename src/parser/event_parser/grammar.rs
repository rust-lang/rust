use super::Event;
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
        p.finish();
        return OK;
    }
    ERR
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
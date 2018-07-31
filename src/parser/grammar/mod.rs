//! This is the actual "grammar" of the Rust language.
//!
//! Each function in this module and its children corresponds
//! to a production of the format grammar. Submodules roughly
//! correspond to different *areas* of the grammar. By convention,
//! each submodule starts with `use super::*` import and exports
//! "public" productions via `pub(super)`.
//!
//! See docs for `Parser` to learn about API, available to the grammar,
//! and see docs for `Event` to learn how this actually manages to
//! produce parse trees.
//!
//! Code in this module also contains inline tests, which start with
//! `// test name-of-the-test` comment and look like this:
//!
//! ```
//! // test fn_item_with_zero_parameters
//! // fn foo() {}
//! ```
//!
//! After adding a new inline-test, run `cargo collect-tests` to extract
//! it as a standalone text-fixture into `tests/data/parser/inline`, and
//! run `cargo test` once to create the "gold" value.
mod attributes;
mod expressions;
mod items;
mod paths;
mod patterns;
mod params;
mod type_params;
mod type_args;
mod types;

use {
    parser::{
        parser::{CompletedMarker, Parser},
        token_set::TokenSet,
    },
    SyntaxKind::{self, *},
};

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
                // test crate_visibility
                // pub(crate) struct S;
                // pub(self) struct S;
                // pub(self) struct S;
                // pub(self) struct S;
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
        name(p);
        alias.complete(p, ALIAS);
    }
    true //FIXME: return false if three are errors
}

fn abi(p: &mut Parser) {
    assert!(p.at(EXTERN_KW));
    let abi = p.start();
    p.bump();
    match p.current() {
        STRING | RAW_STRING => p.bump(),
        _ => (),
    }
    abi.complete(p, ABI);
}

fn fn_ret_type(p: &mut Parser) -> bool {
    if p.at(THIN_ARROW) {
        p.bump();
        types::type_(p);
        true
    } else {
        false
    }
}

fn name(p: &mut Parser) {
    if p.at(IDENT) {
        let m = p.start();
        p.bump();
        m.complete(p, NAME);
    } else {
        p.error("expected a name");
    }
}

fn name_ref(p: &mut Parser) {
    if p.at(IDENT) {
        let m = p.start();
        p.bump();
        m.complete(p, NAME_REF);
    } else {
        p.error("expected identifier");
    }
}

fn error_block(p: &mut Parser, message: &str) {
    assert!(p.at(L_CURLY));
    let err = p.start();
    p.error(message);
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

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
//! // test function_with_zero_parameters
//! // fn foo() {}
//! ```
//!
//! After adding a new inline-test, run `cargo collect-tests` to extract
//! it as a standalone text-fixture into `tests/data/parser/inline`, and
//! run `cargo test` once to create the "gold" value.
//!
//! Coding convention: rules like `where_clause` always produce either a
//! node or an error, rules like `opt_where_clause` may produce nothing.
//! Non-opt rules typically start with `assert!(p.at(FIRST_TOKEN))`, the
//! caller is responsible for branching on the first token.
mod attributes;
mod expressions;
mod items;
mod params;
mod paths;
mod patterns;
mod type_args;
mod type_params;
mod types;

use crate::{
    SyntaxKind::{self, *},
    TokenSet,
    parser::{CompletedMarker, Marker, Parser},
};

pub(crate) fn root(p: &mut Parser) {
    let m = p.start();
    p.eat(SHEBANG);
    items::mod_contents(p, false);
    m.complete(p, SOURCE_FILE);
}

pub(crate) fn reparser(
    node: SyntaxKind,
    first_child: Option<SyntaxKind>,
    parent: Option<SyntaxKind>,
) -> Option<fn(&mut Parser)> {
    let res = match node {
        BLOCK => expressions::block,
        NAMED_FIELD_DEF_LIST => items::named_field_def_list,
        NAMED_FIELD_LIST => items::named_field_list,
        ENUM_VARIANT_LIST => items::enum_variant_list,
        MATCH_ARM_LIST => items::match_arm_list,
        USE_TREE_LIST => items::use_tree_list,
        EXTERN_ITEM_LIST => items::extern_item_list,
        TOKEN_TREE if first_child? == L_CURLY => items::token_tree,
        ITEM_LIST => match parent? {
            IMPL_BLOCK => items::impl_item_list,
            TRAIT_DEF => items::trait_item_list,
            MODULE => items::mod_item_list,
            _ => return None,
        },
        _ => return None,
    };
    Some(res)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockLike {
    Block,
    NotBlock,
}

impl BlockLike {
    fn is_block(self) -> bool {
        self == BlockLike::Block
    }
}

fn opt_visibility(p: &mut Parser) {
    match p.current() {
        PUB_KW => {
            let m = p.start();
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
            m.complete(p, VISIBILITY);
        }
        // test crate_keyword_vis
        // crate fn main() { }
        CRATE_KW => {
            let m = p.start();
            p.bump();
            m.complete(p, VISIBILITY);
        }
        _ => (),
    }
}

fn opt_alias(p: &mut Parser) {
    if p.at(AS_KW) {
        let m = p.start();
        p.bump();
        name(p);
        m.complete(p, ALIAS);
    }
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

fn opt_fn_ret_type(p: &mut Parser) -> bool {
    if p.at(THIN_ARROW) {
        let m = p.start();
        p.bump();
        types::type_(p);
        m.complete(p, RET_TYPE);
        true
    } else {
        false
    }
}

fn name_r(p: &mut Parser, recovery: TokenSet) {
    if p.at(IDENT) {
        let m = p.start();
        p.bump();
        m.complete(p, NAME);
    } else {
        p.err_recover("expected a name", recovery);
    }
}

fn name(p: &mut Parser) {
    name_r(p, TokenSet::empty())
}

fn name_ref(p: &mut Parser) {
    if p.at(IDENT) {
        let m = p.start();
        p.bump();
        m.complete(p, NAME_REF);
    } else {
        p.err_and_bump("expected identifier");
    }
}

fn error_block(p: &mut Parser, message: &str) {
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.error(message);
    p.bump();
    expressions::expr_block_contents(p);
    p.eat(R_CURLY);
    m.complete(p, ERROR);
}

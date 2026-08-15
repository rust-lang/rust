use super::*;

// test use_item
// use std::collections;
pub(super) fn use_(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![use]);
    use_tree(p, true);
    p.expect(T![;]);
    m.complete(p, USE);
}

// test use_tree
// use outer::tree::{inner::tree};
fn use_tree(p: &mut Parser<'_>, top_level: bool) -> bool {
    let m = p.start();
    match p.current() {
        // test use_tree_star
        // use *;
        // use std::{*};
        T![*] => p.bump(T![*]),
        // test use_tree_abs_star
        // use ::*;
        // use std::{::*};
        T![:] if p.at(T![::]) && p.nth(2) == T![*] => {
            p.bump(T![::]);
            p.bump(T![*]);
        }
        T!['{'] => use_tree_list(p),
        T![:] if p.at(T![::]) && p.nth(2) == T!['{'] => {
            p.bump(T![::]);
            use_tree_list(p);
        }

        // test use_tree_path
        // use ::std;
        // use std::collections;
        //
        // use self::m;
        // use super::m;
        // use crate::m;
        _ if paths::is_use_path_start(p) => {
            paths::use_path(p);
            match p.current() {
                // test use_tree_alias
                // use std as stdlib;
                // use Trait as _;
                T![as] => opt_rename(p),
                T![:] if p.at(T![::]) => {
                    p.bump(T![::]);
                    match p.current() {
                        // test use_tree_path_star
                        // use std::*;
                        T![*] => p.bump(T![*]),
                        // test use_tree_path_use_tree
                        // use std::{collections};
                        T!['{'] => use_tree_list(p),
                        _ => p.error("expected `{` or `*`"),
                    }
                }
                _ => (),
            }
        }
        _ => {
            m.abandon(p);
            let msg = "expected one of `*`, `::`, `{`, `self`, `super` or an identifier";
            if top_level {
                p.err_recover(msg, ITEM_RECOVERY_SET);
            } else {
                // if we are parsing a nested tree, we have to eat a token to
                // main balanced `{}`
                p.err_and_bump(msg);
            }
            return false;
        }
    }
    m.complete(p, USE_TREE);
    true
}

pub(super) const USE_TREE_LIST_RECOVERY_SET: TokenSet =
    TokenSet::new(&[T![;], T![,], T![.], T![ident]]).union(ITEM_RECOVERY_SET);

pub(super) const USE_TREE_LIST_FIRST_SET: TokenSet = TokenSet::new(&[T!['{'], T![ident]]);

// test use_tree_list
// use {a, b, c};
pub(crate) fn use_tree_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();

    // test_err use_tree_list_err_recovery
    // use {a;
    // use b;
    // struct T;
    // fn test() {}
    // use {a ,, b};
    delimited(
        p,
        T!['{'],
        T!['}'],
        T![,],
        || "expected use tree".into(),
        USE_TREE_LIST_FIRST_SET,
        |p: &mut Parser<'_>| use_tree(p, false) || p.at_ts(USE_TREE_LIST_RECOVERY_SET),
    );

    m.complete(p, USE_TREE_LIST);
}

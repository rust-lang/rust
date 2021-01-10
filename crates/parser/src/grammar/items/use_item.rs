//! FIXME: write short doc here

use super::*;

pub(super) fn use_(p: &mut Parser, m: Marker) {
    assert!(p.at(T![use]));
    p.bump(T![use]);
    use_tree(p, true);
    p.expect(T![;]);
    m.complete(p, USE);
}

/// Parse a use 'tree', such as `some::path` in `use some::path;`
/// Note that this is called both by `use_item` and `use_tree_list`,
/// so handles both `some::path::{inner::path}` and `inner::path` in
/// `use some::path::{inner::path};`
fn use_tree(p: &mut Parser, top_level: bool) {
    let m = p.start();
    match p.current() {
        // Finish the use_tree for cases of e.g.
        // `use some::path::{self, *};` or `use *;`
        // This does not handle cases such as `use some::path::*`
        // N.B. in Rust 2015 `use *;` imports all from crate root
        // however in Rust 2018 `use *;` errors: ('cannot glob-import all possible crates')
        // FIXME: Add this error (if not out of scope)

        // test use_star
        // use *;
        // use ::*;
        // use some::path::{*};
        // use some::path::{::*};
        T![*] => p.bump(T![*]),
        T![:] if p.at(T![::]) && p.nth(2) == T![*] => {
            // Parse `use ::*;`, which imports all from the crate root in Rust 2015
            // This is invalid inside a use_tree_list, (e.g. `use some::path::{::*}`)
            // but still parses and errors later: ('crate root in paths can only be used in start position')
            // FIXME: Add this error (if not out of scope)
            // In Rust 2018, it is always invalid (see above)
            p.bump(T![::]);
            p.bump(T![*]);
        }
        // Open a use tree list
        // Handles cases such as `use {some::path};` or `{inner::path}` in
        // `use some::path::{{inner::path}, other::path}`

        // test use_tree_list
        // use {crate::path::from::root, or::path::from::crate_name}; // Rust 2018 (with a crate named `or`)
        // use {path::from::root}; // Rust 2015
        // use ::{some::arbitrary::path}; // Rust 2015
        // use ::{{{root::export}}}; // Nonsensical but perfectly legal nesting
        T!['{'] => {
            use_tree_list(p);
        }
        T![:] if p.at(T![::]) && p.nth(2) == T!['{'] => {
            p.bump(T![::]);
            use_tree_list(p);
        }
        // Parse a 'standard' path.
        // Also handles aliases (e.g. `use something as something_else`)

        // test use_path
        // use ::crate_name; // Rust 2018 - All flavours
        // use crate_name; // Rust 2018 - Anchored paths
        // use item_in_scope_or_crate_name; // Rust 2018 - Uniform Paths
        //
        // use self::module::Item;
        // use crate::Item;
        // use self::some::Struct;
        // use crate_name::some_item;
        _ if paths::is_use_path_start(p) => {
            paths::use_path(p);
            match p.current() {
                T![as] => {
                    // test use_alias
                    // use some::path as some_name;
                    // use some::{
                    //  other::path as some_other_name,
                    //  different::path as different_name,
                    //  yet::another::path,
                    //  running::out::of::synonyms::for_::different::*
                    // };
                    // use Trait as _;
                    opt_rename(p);
                }
                T![:] if p.at(T![::]) => {
                    p.bump(T![::]);
                    match p.current() {
                        T![*] => {
                            p.bump(T![*]);
                        }
                        // test use_tree_list_after_path
                        // use crate::{Item};
                        // use self::{Item};
                        T!['{'] => use_tree_list(p),
                        _ => {
                            // is this unreachable?
                            p.error("expected `{` or `*`");
                        }
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
            return;
        }
    }
    m.complete(p, USE_TREE);
}

pub(crate) fn use_tree_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    while !p.at(EOF) && !p.at(T!['}']) {
        use_tree(p, false);
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, USE_TREE_LIST);
}

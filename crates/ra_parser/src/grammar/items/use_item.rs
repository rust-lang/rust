use super::*;

pub(super) fn use_item(p: &mut Parser, m: Marker) {
    assert!(p.at(T![use]));
    p.bump();
    use_tree(p);
    p.expect(T![;]);
    m.complete(p, USE_ITEM);
}

/// Parse a use 'tree', such as `some::path` in `use some::path;`
/// Note that this is called both by `use_item` and `use_tree_list`,
/// so handles both `some::path::{inner::path}` and `inner::path` in
/// `use some::path::{inner::path};`
fn use_tree(p: &mut Parser) {
    let la = p.nth(1);
    let m = p.start();
    match (p.current(), la) {
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
        (T![*], _) => p.bump(),
        (T![::], T![*]) => {
            // Parse `use ::*;`, which imports all from the crate root in Rust 2015
            // This is invalid inside a use_tree_list, (e.g. `use some::path::{::*}`)
            // but still parses and errors later: ('crate root in paths can only be used in start position')
            // FIXME: Add this error (if not out of scope)
            // In Rust 2018, it is always invalid (see above)
            p.bump();
            p.bump();
        }
        // Open a use tree list
        // Handles cases such as `use {some::path};` or `{inner::path}` in
        // `use some::path::{{inner::path}, other::path}`

        // test use_tree_list
        // use {crate::path::from::root, or::path::from::crate_name}; // Rust 2018 (with a crate named `or`)
        // use {path::from::root}; // Rust 2015
        // use ::{some::arbritrary::path}; // Rust 2015
        // use ::{{{crate::export}}}; // Nonsensical but perfectly legal nestnig
        (T!['{'], _) | (T![::], T!['{']) => {
            if p.at(T![::]) {
                p.bump();
            }
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
                    opt_alias(p);
                }
                T![::] => {
                    p.bump();
                    match p.current() {
                        T![*] => {
                            p.bump();
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
            p.err_and_bump("expected one of `*`, `::`, `{`, `self`, `super` or an indentifier");
            return;
        }
    }
    m.complete(p, USE_TREE);
}

pub(crate) fn use_tree_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(T!['}']) {
        use_tree(p);
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, USE_TREE_LIST);
}

use super::*;

pub(super) fn use_item(p: &mut Parser) {
    assert!(p.at(USE_KW));
    p.bump();
    use_tree(p);
    p.expect(SEMI);
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
        // TODO: Add this error (if not out of scope)

        // test use_star
        // use *;
        // use ::*;
        // use some::path::{*};
        // use some::path::{::*};
        (STAR, _) => p.bump(),
        (COLONCOLON, STAR) => {
            // Parse `use ::*;`, which imports all from the crate root in Rust 2015
            // This is invalid inside a use_tree_list, (e.g. `use some::path::{::*}`)
            // but still parses and errors later: ('crate root in paths can only be used in start position')
            // TODO: Add this error (if not out of scope)
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
        (L_CURLY, _) | (COLONCOLON, L_CURLY) => {
            if p.at(COLONCOLON) {
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
        _ if paths::is_path_start(p) => {
            paths::use_path(p);
            match p.current() {
                AS_KW => {
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
                COLONCOLON => {
                    p.bump();
                    match p.current() {
                        STAR => {
                            p.bump();
                        }
                        // test use_tree_list_after_path
                        // use crate::{Item};
                        // use self::{Item};
                        L_CURLY => use_tree_list(p),
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
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        use_tree(p);
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
    m.complete(p, USE_TREE_LIST);
}

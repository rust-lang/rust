# Deny-by-default lints

These lints are all set to the 'deny' level by default.

                          exceeding-bitshifts  deny     shift exceeds the type's number of bits
                   invalid-type-param-default  deny     type parameter default erroneously allowed in invalid location
                legacy-constructor-visibility  deny     detects use of struct constructors that would be invisible with new visibility rules
                   legacy-directory-ownership  deny     non-inline, non-`#[path]` modules (e.g. `mod foo;`) were erroneously allowed in some files not named `mod.rs`
                               legacy-imports  deny     detects names that resolve to ambiguous glob imports with RFC 1560
                   missing-fragment-specifier  deny     detects missing fragment specifiers in unused `macro_rules!` patterns
                           mutable-transmutes  deny     mutating transmuted &mut T from &T may cause undefined behavior
                        no-mangle-const-items  deny     const items will not have their symbols exported
    parenthesized-params-in-types-and-modules  deny     detects parenthesized generic parameters in type and module names
              pub-use-of-private-extern-crate  deny     detect public re-exports of private extern crates
                          safe-extern-statics  deny     safe access to extern statics was erroneously allowed
                          unknown-crate-types  deny     unknown crate type found in #[crate_type] directive
//@ check-pass
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

// Regression test for issue #75930
// Tests that we cfg-strip all targets before invoking
// a derive macro
// FIXME: We currently lose spans here (see issue #43081)
#![warn(legacy_derive_helpers)]
#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

// Note: the expected output contains this sequence:
// ```
// Punct {
//     ch: '<',
//     spacing: Joint,
//     span: $DIR/issue-75930-derive-cfg.rs:25:11: 25:12 (#0),
// },
// Ident {
//     ident: "B",
//     span: $DIR/issue-75930-derive-cfg.rs:25:29: 25:30 (#0),
// },
// ```
// It's surprising to see a `Joint` token tree followed by an `Ident` token
// tree, because `Joint` is supposed to only be used if the following token is
// `Punct`.
//
// It is because of this code from below:
// ```
// struct Foo<#[cfg(false)] A, B>
// ```
// When the token stream is formed during parsing, `<` is followed immediately
// by `#`, which is punctuation, so it is marked `Joint`. But before being
// passed to the proc macro it is rewritten to this:
// ```
// struct Foo<B>
// ```
// But the `Joint` marker on the `<` is not updated. Perhaps it should be
// corrected before being passed to the proc macro? But a prior attempt to do
// that kind of correction caused the problem seen in #76399, so maybe not.

#[print_helper(a)] //~ WARN derive helper attribute is used before it is introduced
                   //~| WARN derive helper attribute is used before it is introduced
                   //~| WARN this was previously accepted
                   //~| WARN this was previously accepted
#[cfg_attr(not(FALSE), allow(dead_code))]
#[print_attr]
#[derive(Print)]
#[print_helper(b)]
struct Foo<#[cfg(false)] A, B> {
    #[cfg(false)] first: String,
    #[cfg_attr(FALSE, deny(warnings))] second: bool,
    third: [u8; {
        #[cfg(false)] struct Bar;
        #[cfg(not(FALSE))] struct Inner;
        #[cfg(false)] let a = 25;
        match true {
            #[cfg(false)] true => {},
            #[cfg_attr(not(FALSE), allow(warnings))] false => {},
            _ => {}
        };

        #[print_helper(should_be_removed)]
        fn removed_fn() {
            #![cfg(false)]
        }

        #[print_helper(c)] #[cfg(not(FALSE))] fn kept_fn() {
            #![cfg(not(FALSE))]
            let my_val = true;
        }

        enum TupleEnum {
            Foo(
                #[cfg(false)] u8,
                #[cfg(false)] bool,
                #[cfg(not(FALSE))] i32,
                #[cfg(false)] String, u8
            )
        }

        struct TupleStruct(
            #[cfg(false)] String,
            #[cfg(not(FALSE))] i32,
            #[cfg(false)] bool,
            u8
        );

        fn plain_removed_fn() {
            #![cfg_attr(not(FALSE), cfg(false))]
        }

        0
    }],
    #[print_helper(d)]
    fourth: B
}

fn main() {}

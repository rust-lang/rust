//@ proc-macro: amputate-span.rs
//@ run-rustfix
//@ edition:2018
//@ compile-flags: --extern amputate_span

// This test has been crafted to ensure the following things:
//
// 1. There's a resolution error that prompts the compiler to suggest
//    adding a `use` item.
//
// 2. There are no `use` or `extern crate` items in the source
//    code. In fact, there is only one item, the `fn main`
//    declaration.
//
// 3. The single `fn main` declaration has an attribute attached to it
//    that just deletes the first token from the given item.
//
// You need all of these conditions to hold in order to replicate the
// scenario that yielded issue 87613, where the compiler's suggestion
// looks like:
//
// ```
// help: consider importing this struct
//    |
// 47 | hey */ async use std::process::Command;
//    |              ++++++++++++++++++++++++++
// ```
//
// The first condition is necessary to force the compiler issue a
// suggestion. The second condition is necessary to force the
// suggestion to be issued at a span associated with the sole
// `fn`-item of this crate. The third condition is necessary in order
// to yield the weird state where the associated span of the `fn`-item
// does not actually cover all of the original source code of the
// `fn`-item (which is why we are calling it an "amputated" span
// here).
//
// Note that satisfying conditions 2 and 3 requires the use of the
// `--extern` compile flag.
//
// You might ask yourself: What code would do such a thing?  The
// answer is: the #[tokio::main] attribute does *exactly* this (as
// well as injecting some other code into the `fn main` that it
// constructs).

#[amputate_span::drop_first_token]
/* what the
hey */ async fn main() {
    Command::new("git"); //~ ERROR [E0433]
}

// (The /* ... */ comment in the above is not part of the original
// bug. It is just meant to illustrate one particular facet of the
// original non-ideal behavior, where we were transcribing the
// trailing comment as part of the emitted suggestion, for better or
// for worse.)

#[allow(dead_code)]
mod inner {
    #[amputate_span::drop_first_token]
        /* another interesting
    case */ async fn foo() {
        Command::new("git"); //~ ERROR [E0433]
    }
}

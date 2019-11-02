// Regression issue for rustdoc ICE encountered in PR #65252.

#![feature(decl_macro)]

fn main() {
    || {
        macro m() {}
    };
}

//! Regression test for <https://github.com/rust-lang/rust/issues/160931>.
//@ check-pass
//@ revisions: e2018 e2024
//@[e2018] edition: 2018
//@[e2024] edition: 2024
#![feature(try_blocks)]

fn diverging_body() {
    None = try {
        return;
    };
}

fn question_mark_then_diverge() {
    None = try {
        None?;
        return;
    };
}

fn main() {}

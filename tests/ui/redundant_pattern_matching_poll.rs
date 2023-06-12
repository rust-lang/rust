//@run-rustfix

#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(
    unused_must_use,
    clippy::needless_bool,
    clippy::needless_if,
    clippy::match_like_matches_macro,
    clippy::equatable_if_let,
    clippy::if_same_then_else
)]

use std::task::Poll::{self, Pending, Ready};

fn main() {
    if let Pending = Pending::<()> {}

    if let Ready(_) = Ready(42) {}

    if let Ready(_) = Ready(42) {
        foo();
    } else {
        bar();
    }

    while let Ready(_) = Ready(42) {}

    while let Pending = Ready(42) {}

    while let Pending = Pending::<()> {}

    if Pending::<i32>.is_pending() {}

    if Ready(42).is_ready() {}

    match Ready(42) {
        Ready(_) => true,
        Pending => false,
    };

    match Pending::<()> {
        Ready(_) => false,
        Pending => true,
    };

    let _ = match Pending::<()> {
        Ready(_) => false,
        Pending => true,
    };

    let poll = Ready(false);
    let _ = if let Ready(_) = poll { true } else { false };

    poll_const();

    let _ = if let Ready(_) = gen_poll() {
        1
    } else if let Pending = gen_poll() {
        2
    } else {
        3
    };
}

fn gen_poll() -> Poll<()> {
    Pending
}

fn foo() {}

fn bar() {}

const fn poll_const() {
    if let Ready(_) = Ready(42) {}

    if let Pending = Pending::<()> {}

    while let Ready(_) = Ready(42) {}

    while let Pending = Pending::<()> {}

    match Ready(42) {
        Ready(_) => true,
        Pending => false,
    };

    match Pending::<()> {
        Ready(_) => false,
        Pending => true,
    };
}

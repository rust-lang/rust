//@ build-fail
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "thread 'rustc' panicked.*\n" -> ""
//@ normalize-stderr: "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ rustc-env:RUST_BACKTRACE=0
//@ known-bug: #144293
//@ failure-status: 101
// Same as recursion-etc but eggs LLVM emission into giving indirect arguments.
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

use std::hint::black_box;

struct U64Wrapper {
    pub x: u64,
    pub arbitrary: String,
}

fn count(curr: U64Wrapper, top: U64Wrapper) -> U64Wrapper {
    if black_box(curr.x) >= top.x {
        curr
    } else {
        become count(
            U64Wrapper {
                x: curr.x + 1,
                arbitrary: curr.arbitrary,
            },
            top,
        )
    }
}

fn main() {
    println!(
        "{}",
        count(
            U64Wrapper {
                x: 0,
                arbitrary: "hello!".into()
            },
            black_box(U64Wrapper {
                x: 1000000,
                arbitrary: "goodbye!".into()
            })
        )
        .x
    );
}

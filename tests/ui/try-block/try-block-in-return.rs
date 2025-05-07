//@ run-pass
//@ edition: 2018

#![feature(try_blocks)]

fn issue_76271() -> Option<i32> {
    return try { 4 }
}

fn main() {
    assert_eq!(issue_76271(), Some(4));
}

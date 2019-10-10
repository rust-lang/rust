// failure-status: 101
// normalize-stderr-test "note: rustc 1.* running on .*" -> "note: rustc VERSION running on TARGET"
// normalize-stderr-test "note: compiler flags: .*" -> "note: compiler flags: FLAGS"

#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

#[track_caller]
fn f() {}

fn main() {
    f();
}

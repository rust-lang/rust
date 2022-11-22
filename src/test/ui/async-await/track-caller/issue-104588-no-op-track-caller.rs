// check-pass
// edition:2021

#[track_caller] //~ WARN `#[track_caller]` on async functions is a no-op, unless the `closure_track_caller` feature is enabled
async fn foo() {}

fn main() {
    foo();
}

// check-pass
// edition:2021

#[track_caller] //~ WARN `#[track_caller]` on async functions is a no-op
async fn foo() {}

fn main() {
    foo();
}

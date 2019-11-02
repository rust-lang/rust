// check-fail

#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

trait Trait {
    fn unwrap(&self);
}

impl Trait for u64 {
    #[track_caller] //~ ERROR: `#[track_caller]` may not be used on trait methods
    fn unwrap(&self) {}
}

struct S;

impl S {
    #[track_caller] // ok
    fn foo() {}
}

fn main() {}

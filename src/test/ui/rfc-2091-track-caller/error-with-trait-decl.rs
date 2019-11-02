#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

trait Trait {
    #[track_caller] //~ ERROR: `#[track_caller]` may not be used on trait methods
    fn unwrap(&self);
}

impl Trait for u64 {
    fn unwrap(&self) {}
}

fn main() {}

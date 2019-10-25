#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

trait Trait {
    fn unwrap(&self);
}

impl Trait for u64 {
    #[track_caller]
    fn unwrap(&self) {}
    //~^^ ERROR: `#[track_caller]` is not supported in traits yet.
}

fn main() {}

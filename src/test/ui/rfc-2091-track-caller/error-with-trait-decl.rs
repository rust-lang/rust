#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

trait Trait {
    #[track_caller]
    fn unwrap(&self);
    //~^^ ERROR: `#[track_caller]` is not supported in trait declarations.
}

impl Trait for u64 {
    fn unwrap(&self) {}
}

fn main() {}

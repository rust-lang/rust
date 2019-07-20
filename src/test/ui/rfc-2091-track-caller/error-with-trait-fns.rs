#![feature(track_caller)]

trait Trait {
    #[track_caller]
    fn unwrap(&self);
    //~^^ ERROR: `#[track_caller]` is not supported for trait items yet.
}

impl Trait for u64 {
    fn unwrap(&self) {}
}

fn main() {}

#![warn(clippy::repeat_once)]

trait Repeat {
    fn repeat(&self) {}
}

impl Repeat for usize {
    fn repeat(&self) {}
}

fn main() {
    let _ = 42.repeat();
}

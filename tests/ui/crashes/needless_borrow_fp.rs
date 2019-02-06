#[deny(clippy::all)]
#[derive(Debug)]
pub enum Error {
    Type(&'static str),
}

fn main() {}

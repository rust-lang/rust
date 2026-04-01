//@ check-pass

#[derive(Debug)]
pub enum Error {
    Type(&'static str),
}

fn main() {}

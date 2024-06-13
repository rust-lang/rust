//@ check-pass

macro_rules! t {
    ($lt:lifetime) => {};
}

t!('fn);

fn main() {}

//@ edition: 2015
//@ check-pass
// Ensure that we parse `'r#lt` as three tokens in edition 2015.

macro_rules! ed2015 {
    ('r # lt) => {};
    ($lt:lifetime) => { compile_error!() };
}

ed2015!('r#lt);

fn main() {}

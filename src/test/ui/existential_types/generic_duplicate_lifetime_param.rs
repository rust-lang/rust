#![feature(existential_type)]

fn main() {}

existential type Two<'a, 'b>: std::fmt::Debug;

fn one<'a>(t: &'a ()) -> Two<'a, 'a> { //~ ERROR non-defining existential type use
    t
}

//@ run-rustfix
pub struct X<const N: usize = {
    let s: &'static str; s.len() //~ ERROR E0381
}>;

fn main() {}

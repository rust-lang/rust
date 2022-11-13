// aux-crate:aux=assoc-inherent-unstable.rs
// edition: 2021

type Data = aux::Owner::Data; //~ ERROR use of unstable library feature 'data'

fn main() {}

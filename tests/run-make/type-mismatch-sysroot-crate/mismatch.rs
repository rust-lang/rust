#![feature(rustc_private)]
extern crate hashbrown;

fn main() {
    uses_hashbrown::foo(hashbrown::HashMap::default())
}

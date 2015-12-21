#![feature(plugin)]
#![plugin(clippy)]

#![deny(out_of_bounds_indexing)]

fn main() {
    let x = [1,2,3,4];
    x[0];
    x[3];
    x[4]; //~ERROR: const index-expr is out of bounds
    x[1 << 3]; //~ERROR: const index-expr is out of bounds
}

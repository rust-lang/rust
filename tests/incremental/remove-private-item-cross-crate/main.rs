// Test that we are able to reuse `main` even though a private
// item was removed from the root module of crate`a`.

// revisions:rpass1 rpass2
// aux-build:a.rs
// compile-flags: -Zquery-dep-graph

#![feature(rustc_attrs)]
#![crate_type = "bin"]

#![rustc_partition_reused(module="main", cfg="rpass2")]

extern crate a;

pub fn main() {
    let vec: Vec<u8> = vec![0, 1, 2, 3];
    for &b in &vec {
        println!("{}", a::foo(b));
    }
}

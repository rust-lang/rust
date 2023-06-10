// run-pass
// compile-flags: -Zmir-opt-level=2 -Zinline-mir
// aux-build:tricky_mir.rs

extern crate tricky_mir;

fn main() {
    println!("{}", *tricky_mir::get_static());
}

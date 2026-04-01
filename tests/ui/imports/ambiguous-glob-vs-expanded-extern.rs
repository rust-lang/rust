//@ check-pass
//@ aux-crate: glob_vs_expanded=glob-vs-expanded.rs

fn main() {
    glob_vs_expanded::mac!(); // OK
}

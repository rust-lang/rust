#![feature(extern_in_paths)]

fn main() {
    let extern = 0; //~ ERROR cannot find unit struct/variant or constant `extern` in this scope
}

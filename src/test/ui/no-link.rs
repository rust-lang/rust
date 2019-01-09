// aux-build:empty-struct.rs

#[no_link]
extern crate empty_struct;

fn main() {
    empty_struct::XEmpty1; //~ ERROR cannot find value `XEmpty1` in module `empty_struct`
}

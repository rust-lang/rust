#![feature(extern_in_paths)]

fn main() {
    let s = extern::xcrate::S; //~ ERROR can't find crate for `xcrate`
}

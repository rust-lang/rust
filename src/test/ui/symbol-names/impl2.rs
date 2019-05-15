#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Foo {
    fn baz();
}

impl Foo for [u8; 1 + 2] {
    #[rustc_def_path] //~ ERROR def-path(<[u8; _] as Foo>::baz)
    fn baz() { }
}

fn main() {
}

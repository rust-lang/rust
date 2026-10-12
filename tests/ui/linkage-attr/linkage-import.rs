//@ build-pass
//@ aux-build:def_weak.rs
//@ ignore-i686-pc-windows-gnu weak symbols broken with MinGW linker
//@ ignore-x86_64-pc-windows-gnu weak symbols broken with MinGW linker

extern crate def_weak as dep;

fn main() {
    println!("{:p}", &dep::WEAK);
}

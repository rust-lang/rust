#![feature(extern_prelude)]

fn main() {
    let s = ep_lib::S; // It works
    s.external();
}

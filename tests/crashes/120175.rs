//@ known-bug: #120175
//@ needs-rustc-debug-assertions

#![feature(extern_types)]
#![feature(raw_dylib_elf)]

#[link(name = "bar", kind = "raw-dylib")]
extern "C" {
    pub type CrossCrate;
}

fn main() {}

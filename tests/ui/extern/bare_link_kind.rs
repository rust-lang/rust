//@ run-pass
//@ aux-build:bare_link_kind_cdylib.rs

#![feature(bare_link_kind)]

#[link(kind = "dylib")]
extern "C" {
    static FOO: u32;
}

#[cfg_attr(not(target_env = "msvc"), link(name = "bare_link_kind_cdylib", kind = "dylib"))]
#[cfg_attr(target_env = "msvc", link(name = "bare_link_kind_cdylib.dll", kind = "dylib"))]
extern "C" {}

fn main() {
    unsafe {
        assert_eq!(FOO, 0xFEDCBA98);
    }
}

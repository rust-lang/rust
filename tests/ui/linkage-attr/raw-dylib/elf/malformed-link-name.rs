//@ only-elf
//@ needs-dynamic-linking
//@ check-fail

#![feature(raw_dylib_elf)]
#![allow(incomplete_features)]

#[link(name = "libc.so.6", kind = "raw-dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    #[link_name = "exit@"]
    pub safe fn exit_0(status: i32) -> !; //~ ERROR link name must be well-formed if link kind is `raw-dylib`
    #[link_name = "@GLIBC_2.2.5"]
    pub safe fn exit_1(status: i32) -> !; //~ ERROR link name must be well-formed if link kind is `raw-dylib`
    #[link_name = "ex\0it@GLIBC_2.2.5"]
    pub safe fn exit_2(status: i32) -> !; //~ ERROR link name must be well-formed if link kind is `raw-dylib`
    #[link_name = "exit@@GLIBC_2.2.5"]
    pub safe fn exit_3(status: i32) -> !; //~ ERROR link name must be well-formed if link kind is `raw-dylib`
}

fn main() {}

//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ ignore-backends: gcc
//@ add-minicore
//@ edition: 2018..
#![feature(no_core)]
#![no_core]
#![feature(mut_restriction)]

extern crate minicore;
use minicore::*;

pub mod foo {
    pub mod bar {
        pub struct Bar(
            pub mut(self) u8,
            pub mut(super) u8,
            pub mut(crate) u8,
        );

        impl Bar {
            pub fn new() -> Self {
                Bar(0, 0, 0)
            }
        }
    }
}

fn change_bar_asm(bar: &mut foo::bar::Bar) {
    unsafe {
        asm!( //~ ERROR field `0` cannot be mutated outside `crate::foo::bar`
            //~^ ERROR field `1` cannot be mutated outside `crate::foo`
            "add {0}, 5",
            "mov {1}, 42",
            "inc {2}",
            inout(reg_byte) bar.0,
            out(reg_byte) bar.1,
            inout(reg_byte) bar.2,
        );
    }
}

fn main() {
    change_bar_asm(&mut foo::bar::Bar::new());
}

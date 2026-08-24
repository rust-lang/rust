//@ add-minicore
//@ revisions: x86_64 aarch64
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64
//@ ignore-backends: gcc
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]
#![deny(unused)]

extern crate minicore;

pub trait Trait {
    extern "C" fn trait_associated(a: usize, b: usize) -> usize;
    extern "C" fn trait_method(&self, a: usize, b: usize) -> usize;
}

pub mod normal {
    use minicore::asm;

    pub extern "C" fn function(a: usize, b: usize) -> usize {
        //~^ ERROR unused variable: `a`
        //~| ERROR unused variable: `b`
        unsafe {
            asm!("", options(noreturn));
        }
    }

    pub struct Normal;

    impl Normal {
        pub extern "C" fn associated(a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe {
                asm!("", options(noreturn));
            }
        }

        pub extern "C" fn method(&self, a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe {
                asm!("", options(noreturn));
            }
        }
    }

    impl super::Trait for Normal {
        extern "C" fn trait_associated(a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe {
                asm!("", options(noreturn));
            }
        }

        extern "C" fn trait_method(&self, a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe {
                asm!("", options(noreturn));
            }
        }
    }
}

pub mod naked {
    use minicore::naked_asm;

    #[unsafe(naked)]
    pub extern "C" fn function(a: usize, b: usize) -> usize {
        naked_asm!("")
    }

    pub struct Naked;

    impl Naked {
        #[unsafe(naked)]
        pub extern "C" fn associated(a: usize, b: usize) -> usize {
            naked_asm!("")
        }

        #[unsafe(naked)]
        pub extern "C" fn method(&self, a: usize, b: usize) -> usize {
            naked_asm!("")
        }
    }

    impl super::Trait for Naked {
        #[unsafe(naked)]
        extern "C" fn trait_associated(a: usize, b: usize) -> usize {
            naked_asm!("")
        }

        #[unsafe(naked)]
        extern "C" fn trait_method(&self, a: usize, b: usize) -> usize {
            naked_asm!("")
        }
    }
}

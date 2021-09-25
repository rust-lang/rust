// revisions: x86_64 aarch64
//[x86_64] only-x86_64
//[aarch64] only-aarch64
#![deny(unused)]
#![feature(asm)]
#![feature(naked_functions)]
#![crate_type = "lib"]

pub trait Trait {
    extern "C" fn trait_associated(a: usize, b: usize) -> usize;
    extern "C" fn trait_method(&self, a: usize, b: usize) -> usize;
}

pub mod normal {
    pub extern "C" fn function(a: usize, b: usize) -> usize {
        //~^ ERROR unused variable: `a`
        //~| ERROR unused variable: `b`
        unsafe { asm!("", options(noreturn)); }
    }

    pub struct Normal;

    impl Normal {
        pub extern "C" fn associated(a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }

        pub extern "C" fn method(&self, a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }
    }

    impl super::Trait for Normal {
        extern "C" fn trait_associated(a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }

        extern "C" fn trait_method(&self, a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }
    }
}

pub mod naked {
    #[naked]
    pub extern "C" fn function(a: usize, b: usize) -> usize {
        unsafe { asm!("", options(noreturn)); }
    }

    pub struct Naked;

    impl Naked {
        #[naked]
        pub extern "C" fn associated(a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }

        #[naked]
        pub extern "C" fn method(&self, a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }
    }

    impl super::Trait for Naked {
        #[naked]
        extern "C" fn trait_associated(a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }

        #[naked]
        extern "C" fn trait_method(&self, a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }
    }
}

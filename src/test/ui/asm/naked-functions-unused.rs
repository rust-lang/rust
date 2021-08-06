// only-x86_64
#![deny(unused)]
#![feature(asm)]
#![feature(naked_functions)]
#![crate_type = "lib"]

pub trait Trait {
    extern "sysv64" fn trait_associated(a: usize, b: usize) -> usize;
    extern "sysv64" fn trait_method(&self, a: usize, b: usize) -> usize;
}

pub mod normal {
    pub extern "sysv64" fn function(a: usize, b: usize) -> usize {
        //~^ ERROR unused variable: `a`
        //~| ERROR unused variable: `b`
        unsafe { asm!("", options(noreturn)); }
    }

    pub struct Normal;

    impl Normal {
        pub extern "sysv64" fn associated(a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }

        pub extern "sysv64" fn method(&self, a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }
    }

    impl super::Trait for Normal {
        extern "sysv64" fn trait_associated(a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }

        extern "sysv64" fn trait_method(&self, a: usize, b: usize) -> usize {
            //~^ ERROR unused variable: `a`
            //~| ERROR unused variable: `b`
            unsafe { asm!("", options(noreturn)); }
        }
    }
}

pub mod naked {
    #[naked]
    pub extern "sysv64" fn function(a: usize, b: usize) -> usize {
        unsafe { asm!("", options(noreturn)); }
    }

    pub struct Naked;

    impl Naked {
        #[naked]
        pub extern "sysv64" fn associated(a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }

        #[naked]
        pub extern "sysv64" fn method(&self, a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }
    }

    impl super::Trait for Naked {
        #[naked]
        extern "sysv64" fn trait_associated(a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }

        #[naked]
        extern "sysv64" fn trait_method(&self, a: usize, b: usize) -> usize {
            unsafe { asm!("", options(noreturn)); }
        }
    }
}

//@ add-core-stubs
//@ revisions: x86_64 aarch64

//@ [x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu

//@ [x86_64] needs-llvm-components: x86
//@ [aarch64] needs-llvm-components: aarch64

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

fn main() {
    let mut foo = 0;
    unsafe {
        asm!("{}");
        //~^ ERROR invalid reference to argument at index 0
        asm!("{1}", in(reg) foo);
        //~^ ERROR invalid reference to argument at index 1
        //~^^ ERROR argument never used
        asm!("{a}");
        //~^ ERROR there is no argument named `a`
        asm!("{}", a = in(reg) foo);
        //~^ ERROR invalid reference to argument at index 0
        //~^^ ERROR argument never used
        asm!("{1}", a = in(reg) foo);
        //~^ ERROR invalid reference to argument at index 1
        //~^^ ERROR named argument never used
        #[cfg(any(x86_64))]
        asm!("{}", in("eax") foo);
        //[x86_64]~^ ERROR invalid reference to argument at index 0
        #[cfg(any(aarch64))]
        asm!("{}", in("x0") foo);
        //[aarch64]~^ ERROR invalid reference to argument at index 0
        asm!("{:foo}", in(reg) foo);
        //~^ ERROR asm template modifier must be a single character
        //~| WARN formatting may not be suitable for sub-register argument [asm_sub_register]
        asm!("", in(reg) 0, in(reg) 1);
        //~^ ERROR multiple unused asm arguments
    }
}

const FOO: i32 = 1;
global_asm!("{}");
//~^ ERROR invalid reference to argument at index 0
global_asm!("{1}", const FOO);
//~^ ERROR invalid reference to argument at index 1
//~^^ ERROR argument never used
global_asm!("{a}");
//~^ ERROR there is no argument named `a`
global_asm!("{}", a = const FOO);
//~^ ERROR invalid reference to argument at index 0
//~^^ ERROR argument never used
global_asm!("{1}", a = const FOO);
//~^ ERROR invalid reference to argument at index 1
//~^^ ERROR named argument never used
global_asm!("{:foo}", const FOO);
//~^ ERROR asm template modifier must be a single character
global_asm!("", const FOO, const FOO);
//~^ ERROR multiple unused asm arguments

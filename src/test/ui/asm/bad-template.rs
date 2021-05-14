// only-x86_64
// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![feature(asm, global_asm)]

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
        asm!("{}", in("eax") foo);
        //~^ ERROR invalid reference to argument at index 0
        asm!("{:foo}", in(reg) foo);
        //~^ ERROR asm template modifier must be a single character
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

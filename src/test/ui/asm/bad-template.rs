// only-x86_64

#![feature(asm)]

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
    }
}

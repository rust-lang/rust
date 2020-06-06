// only-x86_64

#![feature(asm)]

fn main() {
    let mut foo = 0;
    unsafe {
        asm!("{}");
        //~^ ERROR invalid reference to argument at index 0
        asm!("{1}", in(reg) foo);
        //~^ ERROR invalid reference to argument at index 1
        //~^^ WARN asm argument not used in template
        asm!("{a}");
        //~^ ERROR there is no argument named `a`
        asm!("{}", a = in(reg) foo);
        //~^ ERROR invalid reference to argument at index 0
        //~^^ WARN asm argument not used in template
        asm!("{1}", a = in(reg) foo);
        //~^ ERROR invalid reference to argument at index 1
        //~^^ WARN asm argument not used in template
        asm!("{}", in("eax") foo);
        //~^ ERROR invalid reference to argument at index 0
        asm!("{:foo}", in(reg) foo);
        //~^ ERROR asm template modifier must be a single character
        asm!("", in(reg) 0, in(reg) 1);
        //~^ WARN asm arguments not used in template
    }
}

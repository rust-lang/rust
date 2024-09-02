//@ only-x86_64

use std::arch::asm;

fn main() {
    let mut foo = 0;
    let mut bar = 0;
    unsafe {
        asm!("", a = in("eax") foo);
        //~^ ERROR explicit register arguments cannot have names
        asm!("{a}", in("eax") foo, a = const bar);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{a}", in("eax") foo, a = const bar);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{1}", in("eax") foo, const bar);
        //~^ ERROR positional arguments cannot follow named arguments or explicit register arguments
        //~^^ ERROR attempt to use a non-constant value in a constant
    }
}

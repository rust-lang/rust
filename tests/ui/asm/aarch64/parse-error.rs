//@ only-aarch64

use std::arch::asm;

fn main() {
    let mut foo = 0;
    let mut bar = 0;
    unsafe {
        asm!("", a = in("x0") foo);
        //~^ ERROR explicit register arguments cannot have names
        asm!("{a}", in("x0") foo, a = const bar);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{a}", in("x0") foo, a = const bar);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{1}", in("x0") foo, const bar);
        //~^ ERROR positional arguments cannot follow named arguments or explicit register arguments
        //~^^ ERROR attempt to use a non-constant value in a constant
    }
}

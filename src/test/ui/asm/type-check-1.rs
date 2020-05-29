// only-x86_64

#![feature(asm)]

fn main() {
    unsafe {
        // Outputs must be place expressions

        asm!("{}", in(reg) 1 + 2);
        asm!("{}", out(reg) 1 + 2);
        //~^ ERROR invalid asm output
        asm!("{}", inout(reg) 1 + 2);
        //~^ ERROR invalid asm output

        // Operands must be sized

        let v: [u64; 3] = [0, 1, 2];
        asm!("{}", in(reg) v[..]);
        //~^ ERROR the size for values of type `[u64]` cannot be known at compilation time
        asm!("{}", out(reg) v[..]);
        //~^ ERROR the size for values of type `[u64]` cannot be known at compilation time
        asm!("{}", inout(reg) v[..]);
        //~^ ERROR the size for values of type `[u64]` cannot be known at compilation time
    }
}

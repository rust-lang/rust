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

        // Constants must be... constant

        let x = 0;
        const fn const_foo(x: i32) -> i32 {
            x
        }
        const fn const_bar<T>(x: T) -> T {
            x
        }
        asm!("{}", const x);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{}", const const_foo(0));
        asm!("{}", const const_foo(x));
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{}", const const_bar(0));
        asm!("{}", const const_bar(x));
        //~^ ERROR attempt to use a non-constant value in a constant
    }
}

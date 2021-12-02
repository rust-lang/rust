// needs-asm-support
// ignore-nvptx64
// ignore-spirv
// ignore-wasm32

#![feature(asm, global_asm, asm_const)]

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

        // Const operands must be integers and must be constants.

        asm!("{}", const 0);
        asm!("{}", const 0i32);
        asm!("{}", const 0i128);
        asm!("{}", const 0f32);
        //~^ ERROR mismatched types
        asm!("{}", const 0 as *mut u8);
        //~^ ERROR mismatched types
        asm!("{}", const &0);
        //~^ ERROR mismatched types
    }
}

// Const operands must be integers and must be constants.

global_asm!("{}", const 0);
global_asm!("{}", const 0i32);
global_asm!("{}", const 0i128);
global_asm!("{}", const 0f32);
//~^ ERROR mismatched types
global_asm!("{}", const 0 as *mut u8);
//~^ ERROR mismatched types

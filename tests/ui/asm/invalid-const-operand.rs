//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

use std::arch::{asm, global_asm};

// Const operands must be integers and must be constants.

global_asm!("{}", const 0);
global_asm!("{}", const 0i32);
global_asm!("{}", const 0i128);
global_asm!("{}", const 0f32);
//~^ ERROR invalid type for `const` operand
global_asm!("{}", const 0 as *mut u8);
//~^ ERROR invalid type for `const` operand

fn test1() {
    unsafe {
        // Const operands must be integers and must be constants.

        asm!("{}", const 0);
        asm!("{}", const 0i32);
        asm!("{}", const 0i128);
        asm!("{}", const 0f32);
        //~^ ERROR invalid type for `const` operand
        asm!("{}", const 0 as *mut u8);
        //~^ ERROR invalid type for `const` operand
        asm!("{}", const &0);
        //~^ ERROR invalid type for `const` operand
    }
}

fn test2() {
    unsafe {
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

fn main() {}

//@ needs-asm-support

#![feature(global_asm_statement_position)]

use std::arch::global_asm;

fn main() {
    let x = 42;
    global_asm!("{}", in(x));
    //~^ ERROR the `in` operand cannot be used with `global_asm!`
    //~^^ NOTE the `in` operand is not meaningful for global-scoped inline assembly, remove it

    let y = global_asm!("");
    //~^ ERROR non-expression macro in expression position: global_asm
}

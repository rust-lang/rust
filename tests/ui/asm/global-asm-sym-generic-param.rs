// Ensure that we don't ICE when using a type/const parameter in a `sym` operand
// of `global_asm!`. See #156760.

//@ needs-asm-support

use std::arch::global_asm;

trait Trait {
    fn pure();
}

fn fun_ty<T: Trait>() {
    global_asm!("{}", sym<T>::pure);
    //~^ ERROR type parameters are not allowed in `global_asm!` `sym`
}

trait ConstTrait<const N: usize> {
    fn pure();
}

impl<const N: usize> ConstTrait<N> for () {
    fn pure() {}
}

fn fun_const<const N: usize>() {
    global_asm!("{}", sym <() as ConstTrait<N>>::pure);
    //~^ ERROR const parameters are not allowed in `global_asm!` `sym`
}

fn main() {}

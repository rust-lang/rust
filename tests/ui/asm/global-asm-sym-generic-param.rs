// Ensure that we don't ICE when using a type/const parameter in a `sym` operand
// of `global_asm!`. See #156760.

//@ needs-asm-support

use std::arch::global_asm;

trait Trait {
    fn pure();
}

fn fun_ty<T: Trait>() {
    global_asm!("{}", sym <T>::pure);
    //~^ ERROR can't use type parameters from outer item
}

trait TraitWithConstParam<const N: usize> {
    fn pure();
}

impl<const N: usize> TraitWithConstParam<N> for () {
    fn pure() {}
}

fn fun_const<const N: usize>() {
    global_asm!("{}", sym <() as TraitWithConstParam<N>>::pure);
    //~^ ERROR can't use const parameters from outer item
}

fn main() {}

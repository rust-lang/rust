// Ensure that we don't ICE when using a type/const parameter in a `sym` operand
// of `global_asm!`. See #156760.

//@ needs-asm-support

#![feature(global_asm_statement_position, sized_hierarchy)]

use std::arch::global_asm;
use std::marker::{PhantomData, PointeeSized};

trait Trait {
    fn pure();
}

fn fun_ty<T: Trait>() {
    global_asm!("{}", sym<T>::pure);
    //~^ ERROR can't use generic parameters from outer item
}

trait TraitWithConstParam<const N: usize> {
    fn pure();
}

impl<const N: usize> TraitWithConstParam<N> for () {
    fn pure() {}
}

fn fun_const<const N: usize>() {
    global_asm!("{}", sym <() as TraitWithConstParam<N>>::pure);
    //~^ ERROR can't use generic parameters from outer item
    //~| ERROR unresolved item provided when a constant was expected
}

struct S<T>(PhantomData<T>);

impl<T> S<T> {
    fn meow(&self) {
        global_asm!("/* {} */", sym Self::meow);
        //~^ ERROR can't use `Self` from outer item [E0401]
    }
}

struct U<T: PointeeSized>(PhantomData<T>);

impl<T: PointeeSized> U<T> {
    fn meow(&self) {
        global_asm!("/* {} */", sym Self::meow);
        //~^ ERROR can't use `Self` from outer item [E0401]
    }
}

fn main() {}

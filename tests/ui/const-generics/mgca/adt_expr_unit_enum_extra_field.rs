#![feature(min_generic_const_args, adt_const_params)]

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
enum E {
    S {}
}

fn foo<const N: E>() {}

fn main() {
    E::S { x: const { 1 } };
        //~^ ERROR variant `E::S` has no field named `x` [E0559]
}

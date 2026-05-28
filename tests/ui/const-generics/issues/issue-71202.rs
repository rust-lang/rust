#![feature(generic_const_exprs)]
#![allow(incomplete_features, const_evaluatable_unchecked)]

use std::marker::PhantomData;

struct DataHolder<T> {
    item: T,
}

impl<T: Copy> DataHolder<T> {
    const ITEM_IS_COPY: [(); 1 - { //~ ERROR unconstrained generic constant
        trait NotCopy {
            const VALUE: bool = false;
        }

        impl<__Type: ?Sized> NotCopy for __Type {}

        struct IsCopy<__Type: ?Sized>(PhantomData<__Type>);

        impl<__Type> IsCopy<__Type>
        where
            __Type: Sized + Copy,
        {
            const VALUE: bool = true;
        }

        <IsCopy<T>>::VALUE
    } as usize] = [];
    //~^ ERROR unconstrained generic constant
    //~^^ ERROR mismatched types
}

fn main() {}

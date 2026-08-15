#![crate_type = "lib"]
#![feature(transmutability)]

trait A {
    type AssocA;
}

trait B {
    type AssocB: std::mem::TransmuteFrom<()>;
}

impl<T> B for (T, u8)
where
    T: A,
{
    type AssocB = T::AssocA; //~ERROR: the trait bound `<T as A>::AssocA: TransmuteFrom<(), Assume { alignment: false, lifetimes: false, safety: false, validity: false }>` is not satisfied [E0277]
}


impl<T> B for (T, u16)
where
    for<'a> &'a i32: A,
{
    type AssocB = <&'static i32 as A>::AssocA; //~ERROR: `()` cannot be safely transmuted into `<&i32 as A>::AssocA`
}

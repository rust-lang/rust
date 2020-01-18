use std::mem::size_of;

trait Bar {
    const A: usize;
}

fn foo<T: Bar>() {
    const A: usize = size_of::<T>();
    //~^ can't use generic parameters from outer function [E0401]
    const B: usize = T::A;
    //~^ can't use generic parameters from outer function [E0401]
    static C: usize = size_of::<T>();
    //~^ can't use generic parameters from outer function [E0401]
    static D: usize = T::A;
    //~^ can't use generic parameters from outer function [E0401]

    let _ = [0; size_of::<T>()];
    //~^ ERROR type parameters can't appear within an array length expression [E0447]
    let _ = [0; T::A];
    //~^ ERROR type parameters can't appear within an array length expression [E0447]
}

#[repr(usize)]
enum Enum<T: Bar> {
    //~^ ERROR parameter `T` is never used [E0392]
    V1 = size_of::<T>(),
    //~^ ERROR type parameters can't appear within an enum discriminant [E0447]
    V2 = T::A,
    //~^ ERROR type parameters can't appear within an enum discriminant [E0447]
}

fn main() {}

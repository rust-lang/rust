/// The compiler previously did not properly check the bound of `From` when it was used from type
/// of the dyn trait object (use in `copy_any` below). Since the associated type is under user
/// control in this usage, the compiler could be tricked to believe any type implemented any trait.
/// This would ICE, except for pure marker traits like `Copy`. It did not require providing an
/// instance of the dyn trait type, only name said type.
trait Setup {
    type From: Copy;
}

fn copy<U: Setup + ?Sized>(from: &U::From) -> U::From {
    *from
}

pub fn copy_any<T>(t: &T) -> T {
    copy::<dyn Setup<From=T>>(t)
    //~^ ERROR the trait bound `T: Copy` is not satisfied
}

fn main() {}
